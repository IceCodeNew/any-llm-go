// Package ollama provides an Ollama provider implementation for any-llm.
package ollama

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	stderrors "errors"
	"fmt"
	"net/url"
	"strings"
	"time"

	"github.com/ollama/ollama/api"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

// Provider configuration constants.
const (
	defaultBaseURL = "http://localhost:11434"
	defaultNumCtx  = 32000
	envBaseURL     = "OLLAMA_HOST"
	providerName   = "ollama"
)

// Ollama done reasons.
const (
	doneReasonLength = "length"
	doneReasonStop   = "stop"
)

// Ollama option keys.
const (
	optionNumCtx      = "num_ctx"
	optionNumPredict  = "num_predict"
	optionSeed        = "seed"
	optionStop        = "stop"
	optionTemperature = "temperature"
	optionTopP        = "top_p"
)

// JSON schema keys and types.
const (
	schemaKeyDescription = "description"
	schemaKeyProperties  = "properties"
	schemaKeyRequired    = "required"
	schemaKeyType        = "type"
	schemaTypeObject     = "object"
)

// Tool and response format constants.
const (
	emptyJSONObject      = "{}"
	ollamaFormatJSON     = "json"
	responseFormatJSON   = "json_object"
	responseFormatSchema = "json_schema"
	toolCallIDFormat     = "call_%d"
	toolTypeFunction     = "function"
)

// Object type constants.
const (
	objectChatCompletion      = "chat.completion"
	objectChatCompletionChunk = "chat.completion.chunk"
	objectEmbedding           = "embedding"
	objectList                = "list"
	objectModel               = "model"
)

// Thinking tag constants.
const (
	thinkingTagClose = "</think>"
	thinkingTagOpen  = "<think>"
)

// Content part constants.
const (
	contentTypeImageURL = "image_url"
	dataImagePrefix     = "data:image/"
)

// Ensure Provider implements the required interfaces.
var (
	_ providers.CapabilityProvider = (*Provider)(nil)
	_ providers.EmbeddingProvider  = (*Provider)(nil)
	_ providers.ErrorConverter     = (*Provider)(nil)
	_ providers.ModelLister        = (*Provider)(nil)
	_ providers.Provider           = (*Provider)(nil)
)

// Provider implements the providers.Provider interface for Ollama.
type Provider struct {
	client *api.Client
	config *config.Config
}

// streamState tracks accumulated state during streaming.
type streamState struct {
	id        string
	model     string
	created   int64
	content   strings.Builder
	reasoning strings.Builder
}

// New creates a new Ollama provider.
func New(opts ...config.Option) (*Provider, error) {
	cfg, err := config.New(opts...)
	if err != nil {
		return nil, fmt.Errorf("invalid options: %w", err)
	}

	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = cfg.ResolveEnv(envBaseURL)
	}
	if baseURL == "" {
		baseURL = defaultBaseURL
	}

	parsedURL, err := url.Parse(baseURL)
	if err != nil {
		return nil, fmt.Errorf("invalid base URL: %w", err)
	}

	client := api.NewClient(parsedURL, cfg.HTTPClient())

	return &Provider{
		client: client,
		config: cfg,
	}, nil
}

// Capabilities returns the provider's capabilities.
func (p *Provider) Capabilities() providers.Capabilities {
	return providers.Capabilities{
		Completion:          true,
		CompletionImage:     true,
		CompletionPDF:       false,
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
	req, err := p.convertParams(params)
	if err != nil {
		return nil, err
	}

	// Disable streaming for non-stream requests.
	stream := false
	req.Stream = &stream

	var response api.ChatResponse
	err = p.client.Chat(ctx, req, func(resp api.ChatResponse) error {
		response = resp
		return nil
	})
	if err != nil {
		return nil, p.ConvertError(err)
	}

	return convertResponse(&response), nil
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

		req, err := p.convertParams(params)
		if err != nil {
			errs <- err
			return
		}
		state := newStreamState()

		err = p.client.Chat(ctx, req, func(resp api.ChatResponse) error {
			chunk := state.handleChunk(&resp)
			select {
			case chunks <- chunk:
				return nil
			case <-ctx.Done():
				return ctx.Err()
			}
		})
		if err != nil {
			errs <- p.ConvertError(err)
		}
	}()

	return chunks, errs
}

// ConvertError converts Ollama errors to unified error types.
// Implements providers.ErrorConverter.
func (p *Provider) ConvertError(err error) error {
	if err == nil {
		return nil
	}

	// Check for authorization error (401).
	var authErr api.AuthorizationError
	if stderrors.As(err, &authErr) {
		return errors.NewAuthenticationError(providerName, err)
	}

	// Check for HTTP status errors.
	var statusErr api.StatusError
	if stderrors.As(err, &statusErr) {
		switch statusErr.StatusCode {
		case 401:
			return errors.NewAuthenticationError(providerName, err)
		case 404:
			return errors.NewModelNotFoundError(providerName, err)
		case 429:
			return errors.NewRateLimitError(providerName, err)
		case 400:
			if strings.Contains(statusErr.ErrorMessage, "context") {
				return errors.NewContextLengthError(providerName, err)
			}
			return errors.NewInvalidRequestError(providerName, err)
		}
	}

	// Network-level errors (connection refused, etc.) - string check acceptable here.
	if strings.Contains(err.Error(), "connection refused") {
		return errors.NewProviderError(providerName, fmt.Errorf("ollama server not running: %w", err))
	}

	return errors.NewProviderError(providerName, err)
}

// Embedding generates embeddings for the given input.
func (p *Provider) Embedding(
	ctx context.Context,
	params providers.EmbeddingParams,
) (*providers.EmbeddingResponse, error) {
	req := &api.EmbedRequest{
		Model: params.Model,
		Input: params.Input,
	}

	resp, err := p.client.Embed(ctx, req)
	if err != nil {
		return nil, p.ConvertError(err)
	}

	return convertEmbeddingResponse(resp, params.Model), nil
}

// ListModels returns a list of available models.
func (p *Provider) ListModels(ctx context.Context) (*providers.ModelsResponse, error) {
	resp, err := p.client.List(ctx)
	if err != nil {
		return nil, p.ConvertError(err)
	}

	return convertModelsResponse(resp), nil
}

// Name returns the provider name.
func (p *Provider) Name() string {
	return providerName
}

// convertParams converts providers.CompletionParams to Ollama ChatRequest.
func (p *Provider) convertParams(params providers.CompletionParams) (*api.ChatRequest, error) {
	messages, err := convertMessages(params.Messages)
	if err != nil {
		return nil, err
	}

	req := &api.ChatRequest{
		Model:    params.Model,
		Messages: messages,
		Options:  make(map[string]any),
	}

	// Set default context size.
	req.Options[optionNumCtx] = defaultNumCtx

	if params.Temperature != nil {
		req.Options[optionTemperature] = *params.Temperature
	}

	if params.TopP != nil {
		req.Options[optionTopP] = *params.TopP
	}

	if len(params.Stop) > 0 {
		req.Options[optionStop] = params.Stop
	}

	if params.MaxTokens != nil {
		req.Options[optionNumPredict] = *params.MaxTokens
	}

	if params.Seed != nil {
		req.Options[optionSeed] = *params.Seed
	}

	if len(params.Tools) > 0 {
		req.Tools = convertTools(params.Tools)
	}

	if params.ResponseFormat != nil {
		format, err := convertResponseFormat(params.ResponseFormat)
		if err != nil {
			return nil, err
		}
		req.Format = format
	}

	// Ollama accepts a boolean disable switch or an explicit supported level.
	// https://docs.ollama.com/api/chat
	switch params.ReasoningEffort {
	case "", providers.ReasoningEffortAuto:
	case providers.ReasoningEffortNone:
		req.Think = &api.ThinkValue{Value: false}
	case providers.ReasoningEffortLow,
		providers.ReasoningEffortMedium,
		providers.ReasoningEffortHigh,
		providers.ReasoningEffortMax:
		req.Think = &api.ThinkValue{Value: string(params.ReasoningEffort)}
	case providers.ReasoningEffortMinimal, providers.ReasoningEffortXHigh:
		return nil, errors.NewUnsupportedParamError(providerName, "reasoning_effort")
	default:
		return nil, errors.NewUnsupportedParamError(providerName, "reasoning_effort")
	}

	return req, nil
}

// newStreamState creates a new stream state.
func newStreamState() *streamState {
	return &streamState{
		id:      generateID(),
		created: time.Now().Unix(),
	}
}

// chunk creates a ChatCompletionChunk with common fields populated.
func (s *streamState) chunk() providers.ChatCompletionChunk {
	return providers.ChatCompletionChunk{
		ID:      s.id,
		Object:  objectChatCompletionChunk,
		Created: s.created,
		Model:   s.model,
		Choices: []providers.ChunkChoice{{Index: 0}},
	}
}

// handleChunk processes a streaming response and returns a chunk.
func (s *streamState) handleChunk(resp *api.ChatResponse) providers.ChatCompletionChunk {
	s.updateMetadata(resp)

	chunk := s.chunk()
	chunk.Choices[0].Delta = s.buildDelta(resp)

	if resp.Done {
		s.handleDone(resp, &chunk)
	}

	return chunk
}

// updateMetadata updates stream state metadata from response.
func (s *streamState) updateMetadata(resp *api.ChatResponse) {
	if s.model == "" {
		s.model = resp.Model
	}
	if resp.CreatedAt.Unix() > 0 {
		s.created = resp.CreatedAt.Unix()
	}
}

// buildDelta constructs the delta content from a response.
func (s *streamState) buildDelta(resp *api.ChatResponse) providers.ChunkDelta {
	delta := providers.ChunkDelta{}

	// Handle content.
	if resp.Message.Content != "" {
		s.content.WriteString(resp.Message.Content)
		delta.Content = resp.Message.Content
	}

	// Handle thinking/reasoning.
	if resp.Message.Thinking != "" {
		s.reasoning.WriteString(resp.Message.Thinking)
		delta.Reasoning = &providers.Reasoning{Content: resp.Message.Thinking}
	}

	// Handle tool calls.
	if len(resp.Message.ToolCalls) > 0 {
		delta.ToolCalls = convertToolCalls(resp.Message.ToolCalls)
	}

	return delta
}

// handleDone processes the final chunk when streaming is complete.
func (s *streamState) handleDone(resp *api.ChatResponse, chunk *providers.ChatCompletionChunk) {
	finishReason := providers.FinishReasonToolCalls
	if len(resp.Message.ToolCalls) == 0 {
		finishReason = convertDoneReason(resp.DoneReason)
	}

	chunk.Choices[0].FinishReason = finishReason
	chunk.Usage = &providers.Usage{
		PromptTokens:     resp.PromptEvalCount,
		CompletionTokens: resp.EvalCount,
		TotalTokens:      resp.PromptEvalCount + resp.EvalCount,
	}
}

// convertAssistantMessage converts an assistant message to Ollama format.
func convertAssistantMessage(msg providers.Message) (*api.Message, error) {
	ollamaMsg := &api.Message{
		Role:    msg.Role,
		Content: msg.ContentString(),
	}

	if len(msg.ToolCalls) > 0 {
		toolCalls := make([]api.ToolCall, 0, len(msg.ToolCalls))
		for _, tc := range msg.ToolCalls {
			argsMap := make(map[string]any)
			if tc.Function.Arguments != "" {
				if err := json.Unmarshal([]byte(tc.Function.Arguments), &argsMap); err != nil || argsMap == nil {
					return nil, errors.NewInvalidRequestError(
						providerName,
						fmt.Errorf("tool call arguments must be a JSON object"),
					)
				}
			}

			args := api.NewToolCallFunctionArguments()
			for k, v := range argsMap {
				args.Set(k, v)
			}

			toolCalls = append(toolCalls, api.ToolCall{
				ID: tc.ID,
				Function: api.ToolCallFunction{
					Name:      tc.Function.Name,
					Arguments: args,
				},
			})
		}
		ollamaMsg.ToolCalls = toolCalls
	}

	return ollamaMsg, nil
}

// convertDoneReason converts Ollama done reason to OpenAI finish reason.
func convertDoneReason(reason string) string {
	switch reason {
	case doneReasonLength:
		return providers.FinishReasonLength
	default:
		return providers.FinishReasonStop
	}
}

// convertEmbeddingResponse converts an Ollama embedding response to provider format.
func convertEmbeddingResponse(resp *api.EmbedResponse, model string) *providers.EmbeddingResponse {
	data := make([]providers.EmbeddingData, 0, len(resp.Embeddings))

	for i, embedding := range resp.Embeddings {
		// Convert []float32 to []float64.
		floats := make([]float64, len(embedding))
		for j, f := range embedding {
			floats[j] = float64(f)
		}

		data = append(data, providers.EmbeddingData{
			Object:    objectEmbedding,
			Embedding: floats,
			Index:     i,
		})
	}

	return &providers.EmbeddingResponse{
		Object: objectList,
		Data:   data,
		Model:  model,
		Usage: &providers.EmbeddingUsage{
			PromptTokens: resp.PromptEvalCount,
			TotalTokens:  resp.PromptEvalCount,
		},
	}
}

// convertMessage converts a single message to Ollama format.
func convertMessage(msg providers.Message) (*api.Message, error) {
	switch msg.Role {
	case providers.RoleTool:
		return convertToolMessage(msg), nil
	case providers.RoleAssistant:
		return convertAssistantMessage(msg)
	case providers.RoleUser:
		return convertUserMessage(msg)
	default:
		// System and other roles.
		return &api.Message{
			Role:    msg.Role,
			Content: msg.ContentString(),
		}, nil
	}
}

// convertMessages converts provider messages to Ollama format.
func convertMessages(messages []providers.Message) ([]api.Message, error) {
	result := make([]api.Message, 0, len(messages))

	for _, msg := range messages {
		ollamaMsg, err := convertMessage(msg)
		if err != nil {
			return nil, err
		}
		if ollamaMsg != nil {
			result = append(result, *ollamaMsg)
		}
	}

	return result, nil
}

// convertModelsResponse converts an Ollama list response to provider format.
func convertModelsResponse(resp *api.ListResponse) *providers.ModelsResponse {
	models := make([]providers.Model, 0, len(resp.Models))

	for _, m := range resp.Models {
		models = append(models, providers.Model{
			ID:      m.Model,
			Object:  objectModel,
			Created: m.ModifiedAt.Unix(),
			OwnedBy: providerName,
		})
	}

	return &providers.ModelsResponse{
		Object: objectList,
		Data:   models,
	}
}

// convertResponse converts an Ollama response to provider format.
func convertResponse(resp *api.ChatResponse) *providers.ChatCompletion {
	content, reasoning := extractThinking(resp.Message.Content, resp.Message.Thinking)

	message := providers.Message{
		Role:      providers.RoleAssistant,
		Content:   content,
		Reasoning: reasoning,
	}

	// Handle tool calls.
	if len(resp.Message.ToolCalls) > 0 {
		message.ToolCalls = convertToolCalls(resp.Message.ToolCalls)
	}

	finishReason := providers.FinishReasonToolCalls
	if len(resp.Message.ToolCalls) == 0 {
		finishReason = convertDoneReason(resp.DoneReason)
	}

	return &providers.ChatCompletion{
		ID:      generateID(),
		Object:  objectChatCompletion,
		Created: resp.CreatedAt.Unix(),
		Model:   resp.Model,
		Choices: []providers.Choice{{
			Index:        0,
			Message:      message,
			FinishReason: finishReason,
		}},
		Usage: &providers.Usage{
			PromptTokens:     resp.PromptEvalCount,
			CompletionTokens: resp.EvalCount,
			TotalTokens:      resp.PromptEvalCount + resp.EvalCount,
		},
	}
}

// convertResponseFormat converts a response format to Ollama JSON schema.
func convertResponseFormat(format *providers.ResponseFormat) (json.RawMessage, error) {
	if format == nil {
		return nil, nil
	}

	if format.Type == responseFormatJSON {
		return json.RawMessage(`"` + ollamaFormatJSON + `"`), nil
	}

	if format.Type == responseFormatSchema {
		if format.JSONSchema == nil || format.JSONSchema.Schema == nil {
			return nil, errors.NewInvalidRequestError(
				providerName,
				fmt.Errorf("json_schema response format requires a schema"),
			)
		}
		schemaBytes, err := json.Marshal(format.JSONSchema.Schema)
		if err != nil {
			return nil, errors.NewInvalidRequestError(
				providerName,
				fmt.Errorf("marshaling response format schema: %w", err),
			)
		}
		return schemaBytes, nil
	}

	return nil, errors.NewInvalidRequestError(
		providerName,
		fmt.Errorf("unsupported response_format.type %q", format.Type),
	)
}

// convertToolCalls converts Ollama tool calls to provider format.
func convertToolCalls(toolCalls []api.ToolCall) []providers.ToolCall {
	result := make([]providers.ToolCall, 0, len(toolCalls))

	for i, tc := range toolCalls {
		args := emptyJSONObject
		argsMap := tc.Function.Arguments.ToMap()
		if len(argsMap) > 0 {
			if argsBytes, err := json.Marshal(argsMap); err == nil {
				args = string(argsBytes)
			}
		}

		id := tc.ID
		if id == "" {
			id = fmt.Sprintf(toolCallIDFormat, i)
		}
		result = append(result, providers.ToolCall{
			ID:   id,
			Type: toolTypeFunction,
			Function: providers.FunctionCall{
				Name:      tc.Function.Name,
				Arguments: args,
			},
		})
	}

	return result
}

// convertToolMessage converts a tool message to Ollama format.
func convertToolMessage(msg providers.Message) *api.Message {
	// Ollama uses user role for tool results.
	return &api.Message{
		Role:    providers.RoleUser,
		Content: msg.ContentString(),
	}
}

// convertTools converts provider tools to Ollama format.
func convertTools(tools []providers.Tool) api.Tools {
	result := make(api.Tools, 0, len(tools))

	for _, tool := range tools {
		params := api.ToolFunctionParameters{
			Type: schemaTypeObject,
		}

		// Convert properties.
		if props, ok := tool.Function.Parameters[schemaKeyProperties].(map[string]any); ok {
			propsMap := api.NewToolPropertiesMap()
			for name, prop := range props {
				if propMap, ok := prop.(map[string]any); ok {
					tp := api.ToolProperty{}
					if t, ok := propMap[schemaKeyType].(string); ok {
						tp.Type = api.PropertyType{t}
					}
					if d, ok := propMap[schemaKeyDescription].(string); ok {
						tp.Description = d
					}
					propsMap.Set(name, tp)
				}
			}
			params.Properties = propsMap
		}

		// Convert required fields.
		if req, ok := tool.Function.Parameters[schemaKeyRequired].([]any); ok {
			for _, r := range req {
				if s, ok := r.(string); ok {
					params.Required = append(params.Required, s)
				}
			}
		}

		ollamaTool := api.Tool{
			Type: toolTypeFunction,
			Function: api.ToolFunction{
				Name:        tool.Function.Name,
				Description: tool.Function.Description,
				Parameters:  params,
			},
		}

		result = append(result, ollamaTool)
	}

	return result
}

// convertUserMessage converts a user message to Ollama format.
func convertUserMessage(msg providers.Message) (*api.Message, error) {
	ollamaMsg := &api.Message{
		Role:    msg.Role,
		Content: msg.ContentString(),
	}

	// Handle multi-modal messages with images.
	if msg.IsMultiModal() {
		images, err := extractImages(msg)
		if err != nil {
			return nil, err
		}
		if len(images) > 0 {
			ollamaMsg.Images = images
		}
	}

	return ollamaMsg, nil
}

// extractImages extracts base64 image data from a multi-modal message.
func extractImages(msg providers.Message) ([]api.ImageData, error) {
	var images []api.ImageData

	for _, part := range msg.ContentParts() {
		if part.Type == "text" {
			continue
		}
		if part.Type != contentTypeImageURL {
			return nil, errors.NewInvalidRequestError(
				providerName,
				fmt.Errorf("unsupported content part type %q", part.Type),
			)
		}
		if part.ImageURL == nil {
			return nil, errors.NewInvalidRequestError(
				providerName,
				fmt.Errorf("image_url content part is missing image_url"),
			)
		}

		imgURL := part.ImageURL.URL
		if !strings.HasPrefix(imgURL, dataImagePrefix) {
			return nil, errors.NewInvalidRequestError(
				providerName,
				fmt.Errorf("images require base64 data URLs"),
			)
		}

		// Ollama's message.images contract accepts base64 strings, not URLs.
		// https://docs.ollama.com/api/chat
		header, data, ok := strings.Cut(imgURL, ",")
		if !ok || !strings.HasSuffix(header, ";base64") || data == "" {
			return nil, errors.NewInvalidRequestError(providerName, fmt.Errorf("malformed base64 image data URL"))
		}
		if _, err := base64.StdEncoding.DecodeString(data); err != nil {
			return nil, errors.NewInvalidRequestError(providerName, fmt.Errorf("invalid base64 image data: %w", err))
		}

		images = append(images, api.ImageData(data))
	}

	return images, nil
}

// extractThinking extracts thinking content from response.
// It checks the dedicated Thinking field first, then falls back to parsing <think> tags.
func extractThinking(content, thinking string) (string, *providers.Reasoning) {
	// Check for dedicated thinking content first.
	if thinking != "" {
		return content, &providers.Reasoning{Content: thinking}
	}

	// Fall back to parsing <think> tags in content.
	if !strings.Contains(content, thinkingTagOpen) || !strings.Contains(content, thinkingTagClose) {
		return content, nil
	}

	parts := strings.SplitN(content, thinkingTagOpen, 2)
	if len(parts) != 2 {
		return content, nil
	}

	thinkParts := strings.SplitN(parts[1], thinkingTagClose, 2)
	if len(thinkParts) != 2 {
		return content, nil
	}

	reasoning := &providers.Reasoning{Content: thinkParts[0]}
	cleanContent := strings.TrimSpace(parts[0] + thinkParts[1])

	return cleanContent, reasoning
}

// generateID generates a unique ID for responses using crypto/rand.
func generateID() string {
	b := make([]byte, 8)
	_, _ = rand.Read(b)
	return fmt.Sprintf("chatcmpl-%d-%s", time.Now().UnixNano(), hex.EncodeToString(b))
}
