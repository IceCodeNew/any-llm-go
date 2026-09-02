package openai

import (
	"fmt"

	"github.com/openai/openai-go/v3"

	"github.com/mozilla-ai/any-llm-go/providers"
)

// Content part types.
const (
	contentTypeImageURL = "image_url"
	contentTypeText     = "text"
)

// convertAssistantMessage converts an assistant message to OpenAI format.
func convertAssistantMessage(msg providers.Message) openai.ChatCompletionMessageParamUnion {
	if len(msg.ToolCalls) > 0 {
		toolCalls := make([]openai.ChatCompletionMessageToolCallUnionParam, 0, len(msg.ToolCalls))
		for _, tc := range msg.ToolCalls {
			toolCalls = append(toolCalls, openai.ChatCompletionMessageToolCallUnionParam{
				OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
					ID: tc.ID,
					Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				},
			})
		}
		return openai.ChatCompletionMessageParamUnion{
			OfAssistant: &openai.ChatCompletionAssistantMessageParam{
				Content: openai.ChatCompletionAssistantMessageParamContentUnion{
					OfString: openai.String(msg.ContentString()),
				},
				ToolCalls: toolCalls,
			},
		}
	}
	return openai.AssistantMessage(msg.ContentString())
}

// convertMessage converts a single message to OpenAI format.
func convertMessage(msg providers.Message) (openai.ChatCompletionMessageParamUnion, error) {
	switch msg.Role {
	case providers.RoleAssistant:
		return convertAssistantMessage(msg), nil
	case providers.RoleSystem:
		return openai.SystemMessage(msg.ContentString()), nil
	case providers.RoleTool:
		return openai.ToolMessage(msg.ContentString(), msg.ToolCallID), nil
	case providers.RoleUser:
		return convertUserMessage(msg), nil
	default:
		return openai.ChatCompletionMessageParamUnion{}, fmt.Errorf("unknown message role: %q", msg.Role)
	}
}

// convertMessages converts provider messages to OpenAI format.
func convertMessages(messages []providers.Message) ([]openai.ChatCompletionMessageParamUnion, error) {
	result := make([]openai.ChatCompletionMessageParamUnion, 0, len(messages))
	for _, msg := range messages {
		converted, err := convertMessage(msg)
		if err != nil {
			return nil, err
		}
		result = append(result, converted)
	}
	return result, nil
}

// convertUserMessage converts a user message to OpenAI format.
func convertUserMessage(msg providers.Message) openai.ChatCompletionMessageParamUnion {
	if msg.IsMultiModal() {
		parts := make([]openai.ChatCompletionContentPartUnionParam, 0, len(msg.ContentParts()))
		for _, part := range msg.ContentParts() {
			switch part.Type {
			case contentTypeText:
				parts = append(parts, openai.TextContentPart(part.Text))
			case contentTypeImageURL:
				if part.ImageURL != nil {
					parts = append(parts, openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{
						URL: part.ImageURL.URL,
					}))
				}
			}
		}
		return openai.UserMessage(parts)
	}
	return openai.UserMessage(msg.ContentString())
}
