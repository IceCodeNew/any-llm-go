// Package openairesponses implements the provider-neutral Responses contract
// for providers backed by the official OpenAI Go SDK.
package openairesponses

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/responses"

	anyerrors "github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

var (
	errInputRequired   = errors.New("at least one input item is required")
	errModelRequired   = errors.New("model is required")
	errUnsupportedRole = errors.New("unsupported responses role")
)

// Create sends a provider-neutral Responses request through an OpenAI SDK
// client and normalizes the result.
func Create(
	ctx context.Context,
	client *openai.Client,
	providerName string,
	convertError func(error) error,
	params providers.ResponsesParams,
) (*providers.ResponsesResult, error) {
	err := validateParams(providerName, params)
	if err != nil {
		return nil, err
	}

	req, err := convertParams(providerName, params)
	if err != nil {
		return nil, err
	}

	resp, err := client.Responses.New(ctx, req)
	if err != nil {
		return nil, convertError(err)
	}

	// Empty OutputText is valid: tool calls, reasoning-only turns, and refusals
	// can succeed without assistant text. OutputItems and Status let callers
	// distinguish those turns from an actually empty response.
	outputItems, err := convertOutputItems(resp.Output)
	if err != nil {
		return nil, err
	}

	result := &providers.ResponsesResult{
		ID:          resp.ID,
		Model:       resp.Model,
		Status:      string(resp.Status),
		Output:      resp.OutputText(),
		OutputItems: outputItems,
		ProviderRaw: json.RawMessage(resp.RawJSON()),
	}
	if resp.JSON.Usage.Valid() {
		result.Usage = &providers.Usage{
			PromptTokens:     int(resp.Usage.InputTokens),
			CompletionTokens: int(resp.Usage.OutputTokens),
			TotalTokens:      int(resp.Usage.TotalTokens),
			ReasoningTokens:  int(resp.Usage.OutputTokensDetails.ReasoningTokens),
		}
	}

	return result, nil
}

func convertOutputItems(items []responses.ResponseOutputItemUnion) ([]providers.ResponsesOutputItem, error) {
	result := make([]providers.ResponsesOutputItem, 0, len(items))
	for _, item := range items {
		providerRaw := json.RawMessage(item.RawJSON())
		if len(providerRaw) == 0 {
			var rawValue any = item.AsAny()
			if rawValue == nil {
				rawValue = item
			}
			var err error
			providerRaw, err = json.Marshal(rawValue)
			if err != nil {
				return nil, fmt.Errorf("serializing OpenAI Responses output item: %w", err)
			}
		}

		out := providers.ResponsesOutputItem{
			Type:        item.Type,
			ID:          item.ID,
			Status:      item.Status,
			ProviderRaw: providerRaw,
		}
		switch item.Type {
		case "message":
			for _, content := range item.Content {
				switch content.Type {
				case "output_text":
					out.Content += content.Text
				case "refusal":
					out.Refusal += content.Refusal
				}
			}
		case "function_call":
			out.Name = item.Name
			out.CallID = item.CallID
			out.Arguments = item.Arguments.OfString
		case "reasoning":
			for _, summary := range item.Summary {
				out.Summary += summary.Text
			}
		}

		result = append(result, out)
	}

	return result, nil
}

func convertParams(providerName string, params providers.ResponsesParams) (responses.ResponseNewParams, error) {
	items := make(responses.ResponseInputParam, 0, len(params.Input))
	for _, item := range params.Input {
		role, err := responseRole(item.Role)
		if err != nil {
			return responses.ResponseNewParams{}, anyerrors.NewInvalidRequestError(providerName, err)
		}

		items = append(items, responses.ResponseInputItemParamOfMessage(item.Content, role))
	}

	req := responses.ResponseNewParams{
		Model: params.Model,
		Input: responses.ResponseNewParamsInputUnion{
			OfInputItemList: items,
		},
	}

	if params.Instructions != nil {
		req.Instructions = openai.String(*params.Instructions)
	}

	if params.MaxTokens != nil {
		req.MaxOutputTokens = openai.Int(int64(*params.MaxTokens))
	}

	if params.Reasoning != "" && params.Reasoning != providers.ReasoningEffortAuto {
		if !supportedReasoning(params.Reasoning) {
			return responses.ResponseNewParams{}, anyerrors.NewUnsupportedParamError(providerName, "reasoning_effort")
		}
		req.Reasoning = responses.ReasoningParam{
			Effort: responses.ReasoningEffort(params.Reasoning),
		}
	}

	return req, nil
}

func supportedReasoning(effort providers.ReasoningEffort) bool {
	// OpenAI documents model-specific subsets, so keep this SDK-wide and let the
	// API validate model aliases and future models:
	// https://developers.openai.com/api/docs/guides/reasoning
	switch effort {
	case providers.ReasoningEffortNone,
		providers.ReasoningEffortMinimal,
		providers.ReasoningEffortLow,
		providers.ReasoningEffortMedium,
		providers.ReasoningEffortHigh,
		providers.ReasoningEffortXHigh,
		providers.ReasoningEffortMax:
		return true
	case providers.ReasoningEffortAuto:
		return false
	default:
		return false
	}
}

func responseRole(role string) (responses.EasyInputMessageRole, error) {
	switch role {
	case providers.RoleUser:
		return responses.EasyInputMessageRoleUser, nil
	case providers.RoleAssistant:
		return responses.EasyInputMessageRoleAssistant, nil
	case providers.RoleDeveloper:
		return responses.EasyInputMessageRoleDeveloper, nil
	case providers.RoleSystem:
		return responses.EasyInputMessageRoleSystem, nil
	default:
		return "", fmt.Errorf("%w %q", errUnsupportedRole, role)
	}
}

func validateParams(providerName string, params providers.ResponsesParams) error {
	if params.Model == "" {
		return anyerrors.NewInvalidRequestError(providerName, errModelRequired)
	}

	if len(params.Input) == 0 {
		return anyerrors.NewInvalidRequestError(providerName, errInputRequired)
	}

	return nil
}
