// Package anthropic provides an Anthropic provider implementation for any-llm.
package anthropic

import (
	"context"
	stderrors "errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

// Provider configuration constants.
const (
	defaultMaxTokens = 4096
	envAPIKey        = "ANTHROPIC_API_KEY"
	envBaseURL       = "ANTHROPIC_BASE_URL"
	providerName     = "anthropic"
)

// Anthropic content block types.
const (
	blockTypeText             = "text"
	blockTypeThinking         = "thinking"
	blockTypeRedactedThinking = "redacted_thinking"
	blockTypeToolUse          = "tool_use"
)

// Anthropic delta types.
const (
	deltaTypeCitations = "citations_delta"
	deltaTypeInputJSON = "input_json_delta"
	deltaTypeSignature = "signature_delta"
	deltaTypeText      = "text_delta"
	deltaTypeThinking  = "thinking_delta"
)

// Anthropic error response patterns (checked in raw JSON).
const (
	errorPatternContextLength = "context_length"
	errorPatternToken         = "token"
	errorPatternContent       = "content"
	errorPatternSafety        = "safety"
)

// Anthropic streaming event types.
const (
	eventContentBlockDelta = "content_block_delta"
	eventContentBlockStart = "content_block_start"
	eventMessageDelta      = "message_delta"
	eventMessageStart      = "message_start"
)

// Anthropic stop reasons.
const (
	stopReasonEndTurn      = "end_turn"
	stopReasonMaxTokens    = "max_tokens"
	stopReasonStopSequence = "stop_sequence"
	stopReasonToolUse      = "tool_use"
	stopReasonRefusal      = "refusal"
	stopReasonContextLimit = "model_context_window_exceeded"
)

// JSON schema field names.
const (
	schemaFieldProperties = "properties"
	schemaFieldRequired   = "required"
)

// Response format types.
const (
	responseFormatJSONObject = "json_object"
	responseFormatJSONSchema = "json_schema"
)

// Ensure Provider implements the required interfaces.
var (
	_ providers.CapabilityProvider = (*Provider)(nil)
	_ providers.BatchProvider      = (*Provider)(nil)
	_ providers.ErrorConverter     = (*Provider)(nil)
	_ providers.ModelLister        = (*Provider)(nil)
	_ providers.Provider           = (*Provider)(nil)
)

// Provider implements the providers.Provider interface for Anthropic.
type Provider struct {
	client *anthropic.Client
	config *config.Config
}

// New creates a new Anthropic provider.
func New(opts ...config.Option) (*Provider, error) {
	cfg, err := config.New(opts...)
	if err != nil {
		return nil, fmt.Errorf("invalid options: %w", err)
	}

	apiKey := cfg.ResolveAPIKey(envAPIKey)
	if apiKey == "" {
		return nil, errors.NewMissingAPIKeyError(providerName, envAPIKey)
	}

	baseURL, err := cfg.ResolveBaseURL(envBaseURL, "")
	if err != nil {
		return nil, err
	}

	clientOpts := []option.RequestOption{
		option.WithAPIKey(apiKey),
		option.WithHTTPClient(cfg.HTTPClient()),
	}

	if baseURL != "" {
		clientOpts = append(clientOpts, option.WithBaseURL(baseURL))
	}

	client := anthropic.NewClient(clientOpts...)

	return &Provider{
		client: &client,
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
		Embedding:           false,
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

	resp, err := p.client.Messages.New(ctx, req)
	if err != nil {
		return nil, p.ConvertError(err)
	}

	return convertResponse(resp)
}

// convertParams converts providers.CompletionParams to Anthropic request parameters.
func (p *Provider) convertParams(params providers.CompletionParams) (anthropic.MessageNewParams, error) {
	if params.Model == "" {
		return anthropic.MessageNewParams{}, errors.NewInvalidRequestError(
			providerName,
			fmt.Errorf("model is required"),
		)
	}
	if len(params.Messages) == 0 {
		return anthropic.MessageNewParams{}, errors.NewInvalidRequestError(
			providerName,
			fmt.Errorf("messages are required"),
		)
	}
	messages, system, err := convertMessages(params.Messages)
	if err != nil {
		return anthropic.MessageNewParams{}, errors.NewInvalidRequestError(providerName, err)
	}

	maxTokens := int64(defaultMaxTokens)
	if params.MaxTokens != nil {
		maxTokens = int64(*params.MaxTokens)
	}

	req := anthropic.MessageNewParams{
		Model:     anthropic.Model(params.Model),
		Messages:  messages,
		MaxTokens: maxTokens,
	}

	if len(system) > 0 {
		req.System = system
	}

	// Anthropic retains these fields for older models but deprecates or rejects them for
	// models after Claude Opus 4.6. The SDK still models them, so forward caller intent and
	// let the API validate the selected model: https://platform.claude.com/docs/en/api/messages
	if params.Temperature != nil {
		req.Temperature = anthropic.Float(*params.Temperature)
	}

	if params.TopP != nil {
		req.TopP = anthropic.Float(*params.TopP)
	}
	if params.TopK != nil {
		req.TopK = anthropic.Int(int64(*params.TopK))
	}

	if len(params.Stop) > 0 {
		req.StopSequences = params.Stop
	}

	if len(params.Tools) > 0 {
		tools := make([]anthropic.ToolUnionParam, 0, len(params.Tools))
		for _, tool := range params.Tools {
			converted, err := convertTool(tool)
			if err != nil {
				return anthropic.MessageNewParams{}, err
			}
			tools = append(tools, converted)
		}
		req.Tools = tools
	}

	if params.ToolChoice != nil || params.ParallelToolCalls != nil {
		choice, err := convertToolChoice(params.ToolChoice, params.ParallelToolCalls, len(params.Tools) > 0)
		if err != nil {
			return anthropic.MessageNewParams{}, errors.NewInvalidRequestError(providerName, err)
		}
		req.ToolChoice = choice
	}

	if err := applyResponseFormat(&req, params.ResponseFormat); err != nil {
		return anthropic.MessageNewParams{}, errors.NewInvalidRequestError(providerName, err)
	}

	if err := applyThinking(&req, params.ReasoningEffort); err != nil {
		return anthropic.MessageNewParams{}, err
	}

	return req, nil
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

		stream := p.client.Messages.NewStreaming(ctx, req)
		defer func() {
			if err := stream.Close(); err != nil {
				reportStreamError(errs, p.ConvertError(err))
			}
		}()
		state := newStreamState()

		for stream.Next() {
			event := stream.Current()

			switch event.Type {
			case eventMessageStart:
				if !sendChunk(ctx, chunks, state.handleMessageStart(event.AsMessageStart())) {
					errs <- ctx.Err()
					return
				}

			case eventContentBlockStart:
				if chunk := state.handleContentBlockStart(event.AsContentBlockStart()); chunk != nil {
					if !sendChunk(ctx, chunks, *chunk) {
						errs <- ctx.Err()
						return
					}
				}

			case eventContentBlockDelta:
				if chunk := state.handleContentBlockDelta(event.AsContentBlockDelta()); chunk != nil {
					if !sendChunk(ctx, chunks, *chunk) {
						errs <- ctx.Err()
						return
					}
				}

			case eventMessageDelta:
				if !sendChunk(ctx, chunks, state.handleMessageDelta(event.AsMessageDelta())) {
					errs <- ctx.Err()
					return
				}
			}
		}

		if err := stream.Err(); err != nil {
			errs <- p.ConvertError(err)
		}
	}()

	return chunks, errs
}

func sendChunk(
	ctx context.Context,
	chunks chan<- providers.ChatCompletionChunk,
	chunk providers.ChatCompletionChunk,
) bool {
	select {
	case chunks <- chunk:
		return true
	case <-ctx.Done():
		return false
	}
}

func reportStreamError(errs chan<- error, err error) {
	select {
	case errs <- err:
	default:
	}
}

// ListModels returns available Anthropic models.
func (p *Provider) ListModels(ctx context.Context) (*providers.ModelsResponse, error) {
	var models []providers.Model
	pager := p.client.Models.ListAutoPaging(ctx, anthropic.ModelListParams{})
	for pager.Next() {
		info := pager.Current()
		created := int64(0)
		if !info.CreatedAt.IsZero() {
			created = info.CreatedAt.Unix()
		}
		models = append(models, providers.Model{
			ID:      info.ID,
			Object:  "model",
			Created: created,
			OwnedBy: providerName,
		})
	}
	if err := pager.Err(); err != nil {
		return nil, p.ConvertError(err)
	}

	return &providers.ModelsResponse{
		Object: "list",
		Data:   models,
	}, nil
}

// Name returns the provider name.
func (p *Provider) Name() string {
	return providerName
}

// ConvertError converts an Anthropic SDK error to a unified error type.
// Implements providers.ErrorConverter.
func (p *Provider) ConvertError(err error) error {
	if err == nil {
		return nil
	}

	// Extract the Anthropic API error type from the error chain.
	// If it's not an API error (e.g., network error), wrap as generic provider error.
	apiErr, ok := stderrors.AsType[*anthropic.Error](err)
	if !ok {
		return errors.NewProviderError(providerName, err)
	}

	// Classify by HTTP status code.
	switch apiErr.StatusCode {
	case http.StatusUnauthorized:
		return errors.NewAuthenticationError(providerName, err)
	case http.StatusPaymentRequired:
		return errors.NewInsufficientFundsError(providerName, err)
	case http.StatusTooManyRequests:
		return errors.NewRateLimitError(providerName, err)
	case http.StatusNotFound:
		return errors.NewModelNotFoundError(providerName, err)
	case http.StatusRequestEntityTooLarge:
		return errors.NewContextLengthError(providerName, err)
	case http.StatusBadRequest, http.StatusUnprocessableEntity:
		// Anthropic uses 400 for various client errors.
		// Check the raw JSON for context length indicators.
		rawJSON := apiErr.RawJSON()
		if strings.Contains(rawJSON, errorPatternContextLength) || strings.Contains(rawJSON, errorPatternToken) {
			return errors.NewContextLengthError(providerName, err)
		}
		return errors.NewInvalidRequestError(providerName, err)
	case http.StatusForbidden:
		// Forbidden - could be content filter or permission issue.
		rawJSON := apiErr.RawJSON()
		if strings.Contains(rawJSON, errorPatternContent) || strings.Contains(rawJSON, errorPatternSafety) {
			return errors.NewContentFilterError(providerName, err)
		}
		return errors.NewAuthenticationError(providerName, err)
	default:
		return errors.NewProviderError(providerName, err)
	}
}
