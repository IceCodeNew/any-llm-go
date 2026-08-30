// Package gemini provides a Google Gemini provider implementation for any-llm.
package gemini

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	stderrors "errors"
	"fmt"
	"strings"
	"time"

	"google.golang.org/genai"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

// Provider configuration constants.
const (
	envAPIKey       = "GEMINI_API_KEY"
	envAPIKeyGoogle = "GOOGLE_API_KEY"
	envBaseURL      = "GOOGLE_GEMINI_BASE_URL"
	providerName    = "gemini"
)

// Content part types.
const (
	contentPartTypeImageURL = "image_url"
	contentPartTypeFile     = "file"
	contentPartTypeText     = "text"
)

// Gemini role constants.
const (
	roleModel = "model"
	roleUser  = "user"
)

// Object type constants (Gemini doesn't provide these; we set them ourselves).
const (
	objectChatCompletion      = "chat.completion"
	objectChatCompletionChunk = "chat.completion.chunk"
	objectEmbedding           = "embedding"
	objectList                = "list"
	objectModel               = "model"
)

// Response format and tool type constants.
const (
	responseMIMETypeJSON     = "application/json"
	responseFormatJSON       = "json_object"
	responseFormatJSONSchema = "json_schema"
	toolCallFallbackName     = "function"
	toolCallType             = "function"
)

// ID prefix constants for generated identifiers.
const (
	idPrefixCompletion = "gemini-"
	idPrefixToolCall   = "call_"
)

// Extra keys for round-tripping provider metadata.
const (
	extraKeyResponseParts    = "response_parts"
	extraKeyResponse         = "response"
	extraKeyResponseEvents   = "response_events"
	extraKeyThoughtSignature = "thought_signature"
)

// Default MIME type for image URLs when type cannot be determined.
const defaultImageMIMEType = "image/jpeg"

const (
	defaultFileMIMEType = "application/octet-stream"
	inlineSizeLimit     = 20 * 1024 * 1024
)

// NativeToolsExtraKey is the CompletionParams.Extra key for []*genai.Tool.
// It intentionally exposes the official SDK types rather than duplicating them.
const NativeToolsExtraKey = "gemini_native_tools"

// Bypass value for tool calls that lack a real ThoughtSignature.
// See https://ai.google.dev/gemini-api/docs/thought-signatures#faqs
const thoughtSignatureBypass = "skip_thought_signature_validator"

// Error message patterns for 400 error classification.
// The Gemini SDK doesn't expose typed errors for these conditions,
// so we rely on message matching as a pragmatic fallback.
const (
	errMsgContext = "context"
	errMsgToken   = "token"
	errMsgSafety  = "safety"
	errMsgBlock   = "block"
)

// Ensure Provider implements the required interfaces.
var (
	_ providers.CapabilityProvider = (*Provider)(nil)
	_ providers.EmbeddingProvider  = (*Provider)(nil)
	_ providers.ErrorConverter     = (*Provider)(nil)
	_ providers.ModelLister        = (*Provider)(nil)
	_ providers.Provider           = (*Provider)(nil)
)

// Provider implements the providers.Provider interface for Google Gemini.
type Provider struct {
	client *genai.Client
	config *config.Config
}

// New creates a new Gemini provider.
func New(opts ...config.Option) (*Provider, error) {
	cfg, err := config.New(opts...)
	if err != nil {
		return nil, fmt.Errorf("invalid options: %w", err)
	}

	apiKey := cfg.ResolveAPIKey(envAPIKey)
	if apiKey == "" {
		apiKey = cfg.ResolveEnv(envAPIKeyGoogle)
	}
	if apiKey == "" {
		return nil, errors.NewMissingAPIKeyError(providerName, envAPIKey)
	}

	baseURL, err := cfg.ResolveBaseURL(envBaseURL, "")
	if err != nil {
		return nil, fmt.Errorf("resolving Gemini base URL: %w", err)
	}

	clientCfg := &genai.ClientConfig{
		APIKey:     apiKey,
		Backend:    genai.BackendGeminiAPI,
		HTTPClient: cfg.HTTPClient(),
	}
	if baseURL != "" {
		clientCfg.HTTPOptions.BaseURL = baseURL
	}

	client, err := genai.NewClient(context.Background(), clientCfg)
	if err != nil {
		return nil, fmt.Errorf("creating Gemini client: %w", err)
	}

	return &Provider{
		client: client,
		config: cfg,
	}, nil
}

// Capabilities returns the provider's capabilities.
func (p *Provider) Capabilities() providers.Capabilities {
	return providers.Capabilities{
		Batch:               true,
		Completion:          true,
		CompletionImage:     true,
		CompletionPDF:       true,
		CompletionReasoning: true,
		CompletionStreaming: true,
		CompletionTools:     true,
		Embedding:           true,
		ListModels:          true,
	}
}

// Completion performs a chat completion request.
func (p *Provider) Completion(
	ctx context.Context,
	params providers.CompletionParams,
) (*providers.ChatCompletion, error) {
	contents, cfg, err := p.convertParams(params)
	if err != nil {
		return nil, err
	}

	resp, err := p.client.Models.GenerateContent(ctx, params.Model, contents, cfg)
	if err != nil {
		return nil, p.ConvertError(err)
	}

	return convertResponse(resp, params.Model)
}

// CompletionStream performs a streaming chat completion request.
func (p *Provider) CompletionStream(
	ctx context.Context,
	params providers.CompletionParams,
) (<-chan providers.ChatCompletionChunk, <-chan error) {
	chunks := make(chan providers.ChatCompletionChunk)
	errs := make(chan error, 1)

	go func() {
		defer close(chunks)
		defer close(errs)

		contents, cfg, err := p.convertParams(params)
		if err != nil {
			reportStreamError(errs, err)
			return
		}
		state, err := newStreamState(params.Model)
		if err != nil {
			reportStreamError(errs, err)
			return
		}

		for resp, err := range p.client.Models.GenerateContentStream(ctx, params.Model, contents, cfg) {
			if err != nil {
				if ctxErr := ctx.Err(); ctxErr != nil {
					reportStreamError(errs, ctxErr)
				} else {
					reportStreamError(errs, p.ConvertError(err))
				}
				return
			}

			responseChunks, err := state.processResponse(resp)
			if err != nil {
				reportStreamError(errs, err)
				return
			}

			for _, chunk := range responseChunks {
				if !sendStreamChunk(ctx, chunks, errs, chunk) {
					return
				}
			}
		}
		if err := ctx.Err(); err != nil {
			reportStreamError(errs, err)
			return
		}

		// Emit final chunk with finish reason and usage.
		if finalChunk := state.finalChunk(); finalChunk != nil {
			sendStreamChunk(ctx, chunks, errs, *finalChunk)
		}
	}()

	return chunks, errs
}

func reportStreamError(errs chan<- error, err error) {
	select {
	case errs <- err:
	default:
	}
}

func sendStreamChunk(
	ctx context.Context,
	chunks chan<- providers.ChatCompletionChunk,
	errs chan<- error,
	chunk providers.ChatCompletionChunk,
) bool {
	select {
	case chunks <- chunk:
		return true
	case <-ctx.Done():
		reportStreamError(errs, ctx.Err())
		return false
	}
}

// ConvertError converts a Gemini SDK error to a unified error type.
func (p *Provider) ConvertError(err error) error {
	if err == nil {
		return nil
	}

	var apiErr *genai.APIError
	if !stderrors.As(err, &apiErr) {
		return errors.NewProviderError(providerName, err)
	}

	switch apiErr.Code {
	case 401, 403:
		return errors.NewAuthenticationError(providerName, err)
	case 404:
		return errors.NewModelNotFoundError(providerName, err)
	case 429:
		return errors.NewRateLimitError(providerName, err)
	case 400:
		// The Gemini SDK doesn't expose typed errors for context length or content
		// filter violations, so we use message matching as a pragmatic fallback.
		msg := strings.ToLower(apiErr.Message)
		if strings.Contains(msg, errMsgContext) || strings.Contains(msg, errMsgToken) {
			return errors.NewContextLengthError(providerName, err)
		}
		if strings.Contains(msg, errMsgSafety) || strings.Contains(msg, errMsgBlock) {
			return errors.NewContentFilterError(providerName, err)
		}
		return errors.NewInvalidRequestError(providerName, err)
	default:
		return errors.NewProviderError(providerName, err)
	}
}

// Embedding generates embeddings for the given input.
func (p *Provider) Embedding(
	ctx context.Context,
	params providers.EmbeddingParams,
) (*providers.EmbeddingResponse, error) {
	contents, err := convertEmbeddingInputs(params.Input)
	if err != nil {
		return nil, err
	}
	var cfg *genai.EmbedContentConfig
	if params.Dimensions != nil {
		dimensions, conversionErr := positiveInt32(*params.Dimensions, "dimensions")
		if conversionErr != nil {
			return nil, conversionErr
		}
		cfg = &genai.EmbedContentConfig{OutputDimensionality: new(dimensions)}
	}

	resp, err := p.client.Models.EmbedContent(ctx, params.Model, contents, cfg)
	if err != nil {
		return nil, p.ConvertError(err)
	}

	data := make([]providers.EmbeddingData, 0, len(resp.Embeddings))
	for i, emb := range resp.Embeddings {
		values := make([]float64, len(emb.Values))
		for j, v := range emb.Values {
			values[j] = float64(v)
		}
		data = append(data, providers.EmbeddingData{
			Object:    objectEmbedding,
			Embedding: values,
			Index:     i,
		})
	}

	return &providers.EmbeddingResponse{
		Object: objectList,
		Data:   data,
		Model:  params.Model,
	}, nil
}

// ListModels returns available models.
func (p *Provider) ListModels(ctx context.Context) (*providers.ModelsResponse, error) {
	var models []providers.Model

	page, err := p.client.Models.List(ctx, nil)
	if err != nil {
		return nil, p.ConvertError(err)
	}

	for {
		for _, m := range page.Items {
			models = append(models, providers.Model{
				ID:      m.Name,
				Object:  objectModel,
				OwnedBy: "google",
			})
		}

		if page.NextPageToken == "" {
			break
		}

		page, err = page.Next(ctx)
		if stderrors.Is(err, genai.ErrPageDone) {
			break
		}
		if err != nil {
			return nil, p.ConvertError(err)
		}
	}

	return &providers.ModelsResponse{
		Object: objectList,
		Data:   models,
	}, nil
}

// Name returns the provider name.
func (p *Provider) Name() string {
	return providerName
}

// convertParams converts providers.CompletionParams to Gemini request format.
func (p *Provider) convertParams(
	params providers.CompletionParams,
) ([]*genai.Content, *genai.GenerateContentConfig, error) {
	if err := validateCompletionParams(params); err != nil {
		return nil, nil, err
	}
	contents, systemInstruction, err := convertMessages(params.Messages)
	if err != nil {
		return nil, nil, err
	}

	cfg := &genai.GenerateContentConfig{}
	if err := applyGenerationParams(cfg, systemInstruction, params); err != nil {
		return nil, nil, err
	}
	if err := applyCompletionTools(cfg, params); err != nil {
		return nil, nil, err
	}
	if err := applyThinking(cfg, params.Model, params.ReasoningEffort); err != nil {
		return nil, nil, err
	}
	if params.ResponseFormat != nil {
		if err := applyResponseFormat(cfg, params.ResponseFormat); err != nil {
			return nil, nil, err
		}
	}
	if hasThoughtSignatureBypass(contents) {
		cfg.HTTPOptions = &genai.HTTPOptions{ExtrasRequestProvider: rewriteThoughtSignatureBypass}
	}

	return contents, cfg, nil
}

func validateCompletionParams(params providers.CompletionParams) error {
	if params.Model == "" {
		return errors.NewInvalidRequestError(providerName, fmt.Errorf("model is required"))
	}
	if len(params.Messages) == 0 {
		return errors.NewInvalidRequestError(providerName, fmt.Errorf("messages are required"))
	}
	if params.ParallelToolCalls != nil {
		return errors.NewUnsupportedParamError(providerName, "parallel_tool_calls")
	}
	if err := validateMessages(params.Messages); err != nil {
		return err
	}
	return nil
}

func applyCompletionTools(cfg *genai.GenerateContentConfig, params providers.CompletionParams) error {
	if len(params.Tools) > 0 {
		cfg.Tools = convertTools(params.Tools)
	}
	if native, ok := params.Extra[NativeToolsExtraKey]; ok {
		nativeTools, ok := native.([]*genai.Tool)
		if !ok {
			return errors.NewInvalidRequestError(
				providerName,
				fmt.Errorf("%s must be []*genai.Tool", NativeToolsExtraKey),
			)
		}
		cfg.Tools = append(cfg.Tools, nativeTools...)
	}

	if params.ToolChoice != nil {
		var err error
		cfg.ToolConfig, err = convertToolChoice(params.ToolChoice)
		if err != nil {
			return err
		}
	}
	return nil
}

func applyGenerationParams(
	cfg *genai.GenerateContentConfig,
	systemInstruction *genai.Content,
	params providers.CompletionParams,
) error {
	cfg.SystemInstruction = systemInstruction
	applySamplingParams(cfg, params)

	if params.MaxTokens != nil {
		maxTokens, err := positiveInt32(*params.MaxTokens, "max_tokens")
		if err != nil {
			return err
		}
		cfg.MaxOutputTokens = maxTokens
	}
	if params.Seed != nil {
		seed, err := int32Value(*params.Seed, "seed")
		if err != nil {
			return err
		}
		cfg.Seed = new(seed)
	}
	if params.ServiceTier != "" {
		if err := applyServiceTier(cfg, params.ServiceTier); err != nil {
			return err
		}
	}
	return nil
}

func applySamplingParams(cfg *genai.GenerateContentConfig, params providers.CompletionParams) {
	if params.Temperature != nil {
		cfg.Temperature = new(float32(*params.Temperature))
	}
	if params.TopP != nil {
		cfg.TopP = new(float32(*params.TopP))
	}
	if params.FrequencyPenalty != nil {
		cfg.FrequencyPenalty = new(float32(*params.FrequencyPenalty))
	}
	if params.PresencePenalty != nil {
		cfg.PresencePenalty = new(float32(*params.PresencePenalty))
	}
	if len(params.Stop) > 0 {
		cfg.StopSequences = params.Stop
	}
}

func applyServiceTier(cfg *genai.GenerateContentConfig, serviceTier string) error {
	switch serviceTier {
	case string(genai.ServiceTierFlex), string(genai.ServiceTierStandard), string(genai.ServiceTierPriority):
		cfg.ServiceTier = genai.ServiceTier(serviceTier)
		return nil
	default:
		return errors.NewInvalidRequestError(
			providerName,
			fmt.Errorf("unsupported service_tier %q", serviceTier),
		)
	}
}

// applyResponseFormat configures the response format on the config.
// For json_schema, Gemini requires both responseMimeType application/json and
// responseJsonSchema (see https://ai.google.dev/gemini-api/docs/structured-output).
func applyResponseFormat(cfg *genai.GenerateContentConfig, format *providers.ResponseFormat) error {
	if format == nil {
		return nil
	}
	switch format.Type {
	case responseFormatJSONSchema:
		if format.JSONSchema == nil || format.JSONSchema.Schema == nil {
			return errors.NewInvalidRequestError(
				providerName,
				fmt.Errorf("json_schema response format requires a schema"),
			)
		}
		cfg.ResponseMIMEType = responseMIMETypeJSON
		cfg.ResponseJsonSchema = format.JSONSchema.Schema
	case responseFormatJSON:
		cfg.ResponseMIMEType = responseMIMETypeJSON
	default:
		return errors.NewInvalidRequestError(providerName, fmt.Errorf("unsupported response format %q", format.Type))
	}
	return nil
}

// extractResponseContent extracts content, reasoning, tool calls, and finish reason from a Gemini response.
func extractResponseContent(
	resp *genai.GenerateContentResponse,
) (string, *providers.Reasoning, []providers.ToolCall, string, error) {
	if len(resp.Candidates) == 0 {
		if promptWasBlocked(resp) {
			return "", nil, nil, providers.FinishReasonContentFilter, nil
		}
		return "", nil, nil, "", nil
	}

	candidate := resp.Candidates[0]
	finishReason := convertFinishReason(candidate.FinishReason)

	if candidate.Content == nil {
		return "", nil, nil, finishReason, nil
	}

	var contentBuilder strings.Builder
	var reasoningBuilder strings.Builder
	var toolCalls []providers.ToolCall
	hasReasoning := false

	for _, part := range candidate.Content.Parts {
		switch {
		case part.FunctionCall != nil:
			toolCall, err := convertFunctionCallToToolCall(part.FunctionCall)
			if err != nil {
				return "", nil, nil, "", err
			}

			// Preserve the thought signature so callers can echo it back on the next turn.
			if len(part.ThoughtSignature) > 0 {
				setProviderExtra(&toolCall, providerName, extraKeyThoughtSignature,
					base64.StdEncoding.EncodeToString(part.ThoughtSignature))
			}
			toolCalls = append(toolCalls, toolCall)
		case part.Thought:
			hasReasoning = true
			reasoningBuilder.WriteString(part.Text)
		case part.Text != "":
			contentBuilder.WriteString(part.Text)
		}
	}

	var reasoning *providers.Reasoning
	if hasReasoning {
		reasoning = &providers.Reasoning{Content: reasoningBuilder.String()}
	}

	return contentBuilder.String(), reasoning, toolCalls, finishReason, nil
}

// convertResponse converts a Gemini response to providers format.
func convertResponse(resp *genai.GenerateContentResponse, model string) (*providers.ChatCompletion, error) {
	content, reasoning, toolCalls, finishReason, err := extractResponseContent(resp)
	if err != nil {
		return nil, err
	}

	if len(toolCalls) > 0 && finishReason == providers.FinishReasonStop {
		finishReason = providers.FinishReasonToolCalls
	}
	extra, err := responseExtra(resp)
	if err != nil {
		return nil, err
	}

	message := providers.Message{
		Role:      providers.RoleAssistant,
		Content:   content,
		ToolCalls: toolCalls,
		Reasoning: reasoning,
		Extra:     extra,
	}

	id, err := generateID(idPrefixCompletion)
	if err != nil {
		return nil, err
	}

	completion := &providers.ChatCompletion{
		ID:      id,
		Object:  objectChatCompletion,
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []providers.Choice{{
			Index:        0,
			Message:      message,
			FinishReason: finishReason,
		}},
	}

	if resp.UsageMetadata != nil {
		completion.Usage = &providers.Usage{
			PromptTokens:     int(resp.UsageMetadata.PromptTokenCount),
			CompletionTokens: int(resp.UsageMetadata.CandidatesTokenCount),
			TotalTokens:      usageTotal(resp),
			ReasoningTokens:  int(resp.UsageMetadata.ThoughtsTokenCount),
			CachedTokens:     int(resp.UsageMetadata.CachedContentTokenCount),
		}
	}

	return completion, nil
}

func usageTotal(resp *genai.GenerateContentResponse) int {
	usage := resp.UsageMetadata
	if rawTotal, ok := rawGeminiTotalTokenCount(resp); ok {
		return rawTotal
	}
	if usage.TotalTokenCount != 0 {
		return int(usage.TotalTokenCount)
	}
	return int(usage.PromptTokenCount + usage.CandidatesTokenCount)
}

func rawGeminiTotalTokenCount(resp *genai.GenerateContentResponse) (int, bool) {
	if resp.SDKHTTPResponse == nil || !json.Valid([]byte(resp.SDKHTTPResponse.Body)) {
		return 0, false
	}
	var envelope struct {
		UsageMetadata *struct {
			TotalTokenCount *int32 `json:"totalTokenCount"`
		} `json:"usageMetadata"`
	}
	if err := json.Unmarshal([]byte(resp.SDKHTTPResponse.Body), &envelope); err != nil ||
		envelope.UsageMetadata == nil || envelope.UsageMetadata.TotalTokenCount == nil {
		return 0, false
	}
	return int(*envelope.UsageMetadata.TotalTokenCount), true
}

func responseTextExtra(resp *genai.GenerateContentResponse) map[string]providers.ProviderData {
	if len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil {
		return nil
	}
	for _, part := range resp.Candidates[0].Content.Parts {
		if !part.Thought {
			if extra := thoughtSignatureExtra(part); extra != nil {
				return extra
			}
		}
	}
	return nil
}

func responseExtra(resp *genai.GenerateContentResponse) (map[string]providers.ProviderData, error) {
	extra := responseTextExtra(resp)
	rawResponse, err := rawGeminiResponse(resp)
	if err != nil {
		return nil, err
	}
	extra = withGeminiExtra(extra, extraKeyResponse, rawResponse)
	if len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil {
		return extra, nil
	}

	// Gemini requires callers to replay model parts in their original order, including
	// thought signatures and server tool data that normalized fields cannot represent.
	// https://ai.google.dev/gemini-api/docs/thought-signatures
	parts := resp.Candidates[0].Content.Parts
	raw, err := json.Marshal(parts)
	if err != nil {
		return nil, fmt.Errorf("serializing Gemini response parts: %w", err)
	}
	return withResponseParts(extra, raw), nil
}

func withResponseParts(
	extra map[string]providers.ProviderData,
	raw json.RawMessage,
) map[string]providers.ProviderData {
	return withGeminiExtra(extra, extraKeyResponseParts, raw)
}

func withGeminiExtra(
	extra map[string]providers.ProviderData,
	key string,
	value any,
) map[string]providers.ProviderData {
	if extra == nil {
		extra = make(map[string]providers.ProviderData)
	}
	if extra[providerName] == nil {
		extra[providerName] = make(providers.ProviderData)
	}
	extra[providerName][key] = value
	return extra
}

func rawGeminiResponse(resp *genai.GenerateContentResponse) (json.RawMessage, error) {
	// Google exposes the processed response body through SDKHTTPResponse. Prefer it
	// because generated SDK structs use omitempty and cannot retain the presence of
	// explicit false, zero, or empty metadata fields.
	// https://pkg.go.dev/google.golang.org/genai#GenerateContentResponse
	if resp.SDKHTTPResponse != nil && json.Valid([]byte(resp.SDKHTTPResponse.Body)) {
		return json.RawMessage(resp.SDKHTTPResponse.Body), nil
	}
	raw, err := json.Marshal(resp)
	if err != nil {
		return nil, fmt.Errorf("serializing Gemini response metadata: %w", err)
	}
	return raw, nil
}

func promptWasBlocked(resp *genai.GenerateContentResponse) bool {
	return resp.PromptFeedback != nil && resp.PromptFeedback.BlockReason != genai.BlockedReasonUnspecified
}

// generateID generates a random ID with the given prefix.
func generateID(prefix string) (string, error) {
	b := make([]byte, 12)
	if _, err := rand.Read(b); err != nil {
		return "", fmt.Errorf("generating ID: %w", err)
	}
	return prefix + hex.EncodeToString(b), nil
}

// setProviderExtra safely sets a key in a ToolCall's provider-specific Extra data.
// Initialises the maps if nil, and preserves existing keys.
// NOTE: This lives in the gemini package for now; lift to providers if other providers need it.
func setProviderExtra(tc *providers.ToolCall, provider string, key string, value any) {
	if tc.Extra == nil {
		tc.Extra = make(map[string]providers.ProviderData)
	}
	if tc.Extra[provider] == nil {
		tc.Extra[provider] = make(providers.ProviderData)
	}
	tc.Extra[provider][key] = value
}

// thoughtSignatureFromExtra extracts and base64-decodes a ThoughtSignature
// from ToolCall Extra data. Returns nil if not present or invalid.
