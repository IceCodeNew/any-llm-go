package anthropic

import (
	"context"
	"encoding/json"
	stderrors "errors"
	"net/http"
	"net/url"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/internal/testutil"
	"github.com/mozilla-ai/any-llm-go/providers"
)

func TestNew(t *testing.T) {
	t.Run("creates provider with API key", func(t *testing.T) {
		provider, err := New(config.WithAPIKey("test-api-key"))
		require.NoError(t, err)
		require.NotNil(t, provider)
		require.Equal(t, "anthropic", provider.Name())
	})

	t.Run("creates provider from environment variable", func(t *testing.T) {
		t.Setenv("ANTHROPIC_API_KEY", "env-api-key")

		provider, err := New()
		require.NoError(t, err)
		require.NotNil(t, provider)
	})

	t.Run("returns error when API key is missing", func(t *testing.T) {
		t.Setenv("ANTHROPIC_API_KEY", "")

		provider, err := New()
		require.Nil(t, provider)
		require.Error(t, err)

		var missingKeyErr *errors.MissingAPIKeyError
		require.ErrorAs(t, err, &missingKeyErr)
		require.Equal(t, "anthropic", missingKeyErr.Provider)
		require.Equal(t, "ANTHROPIC_API_KEY", missingKeyErr.EnvVar)
	})
}

func TestCapabilities(t *testing.T) {
	t.Parallel()

	provider, err := New(config.WithAPIKey("test-key"))
	require.NoError(t, err)

	caps := provider.Capabilities()

	require.True(t, caps.Batch)
	require.True(t, caps.Completion)
	require.True(t, caps.CompletionImage)
	require.True(t, caps.CompletionPDF)
	require.True(t, caps.CompletionReasoning)
	require.True(t, caps.CompletionStreaming)
	require.True(t, caps.CompletionTools)
	require.False(t, caps.Embedding) // Anthropic doesn't support embeddings.
	require.True(t, caps.ListModels)
}

func TestConvertMessages(t *testing.T) {
	t.Parallel()

	t.Run("extracts system message", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleSystem, Content: "You are a helpful assistant."},
			{Role: providers.RoleUser, Content: "Hello"},
		}

		result, system, err := convertMessages(messages)
		require.NoError(t, err)

		require.Equal(t, "You are a helpful assistant.", system[0].Text)
		require.Len(t, result, 1) // Only user message.
	})

	t.Run("concatenates multiple system messages", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleSystem, Content: "First part."},
			{Role: providers.RoleSystem, Content: "Second part."},
			{Role: providers.RoleUser, Content: "Hello"},
		}

		result, system, err := convertMessages(messages)
		require.NoError(t, err)

		require.Equal(t, []string{"First part.", "Second part."}, []string{system[0].Text, system[1].Text})
		require.Len(t, result, 1)
	})

	t.Run("converts user message", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleUser, Content: "Hello"},
		}

		result, system, err := convertMessages(messages)
		require.NoError(t, err)

		require.Empty(t, system)
		require.Len(t, result, 1)
	})

	t.Run("converts assistant message", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleUser, Content: "Hello"},
			{Role: providers.RoleAssistant, Content: "Hi there!"},
		}

		result, system, err := convertMessages(messages)
		require.NoError(t, err)

		require.Empty(t, system)
		require.Len(t, result, 2)
	})

	t.Run("converts assistant message with tool calls", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleUser, Content: "What's the weather?"},
			{
				Role:    providers.RoleAssistant,
				Content: "",
				ToolCalls: []providers.ToolCall{
					{
						ID:   "call_123",
						Type: "function",
						Function: providers.FunctionCall{
							Name:      "get_weather",
							Arguments: `{"location": "Paris"}`,
						},
					},
				},
			},
		}

		result, _, err := convertMessages(messages)
		require.NoError(t, err)

		require.Len(t, result, 2)
	})

	t.Run("converts tool result to user message", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleUser, Content: "What's the weather?"},
			{
				Role:    providers.RoleAssistant,
				Content: "",
				ToolCalls: []providers.ToolCall{
					{
						ID:       "call_123",
						Type:     "function",
						Function: providers.FunctionCall{Name: "get_weather", Arguments: `{"location": "Paris"}`},
					},
				},
			},
			{Role: providers.RoleTool, Content: "sunny, 22°C", ToolCallID: "call_123"},
		}

		result, _, err := convertMessages(messages)
		require.NoError(t, err)

		require.Len(t, result, 3)
	})
}

func TestConvertImagePart(t *testing.T) {
	t.Parallel()

	t.Run("converts URL image", func(t *testing.T) {
		t.Parallel()

		img := &providers.ImageURL{URL: "https://example.com/image.png"}
		result, err := convertImagePart(img)
		require.NoError(t, err)
		require.NotNil(t, result)
	})

	t.Run("converts base64 image", func(t *testing.T) {
		t.Parallel()

		img := &providers.ImageURL{URL: "data:image/jpeg;base64,/9j/4AAQSkZJRg=="}
		result, err := convertImagePart(img)
		require.NoError(t, err)
		require.NotNil(t, result)
	})
}

func TestConvertStopReason(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "end_turn",
			input:    "end_turn",
			expected: providers.FinishReasonStop,
		},
		{
			name:     "max_tokens",
			input:    "max_tokens",
			expected: providers.FinishReasonLength,
		},
		{
			name:     "tool_use",
			input:    "tool_use",
			expected: providers.FinishReasonToolCalls,
		},
		{
			name:     "stop_sequence",
			input:    "stop_sequence",
			expected: providers.FinishReasonStop,
		},
		{
			name:     "unknown",
			input:    "unknown",
			expected: providers.FinishReasonStop,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result := convertStopReason(tc.input)
			require.Equal(t, tc.expected, result)
		})
	}
}

func TestNewStreamState(t *testing.T) {
	t.Parallel()

	state := newStreamState()
	require.NotNil(t, state)
	require.Equal(t, -1, state.currentToolIdx)
	require.Empty(t, state.messageID)
	require.Empty(t, state.model)
	require.Nil(t, state.toolCalls)
}

func TestStreamStateHandleTextDelta(t *testing.T) {
	t.Parallel()

	state := newStreamState()
	state.messageID = "msg_123"
	state.model = "claude-3"

	chunk := state.handleTextDelta("Hello ")
	require.NotNil(t, chunk)
	require.Equal(t, "msg_123", chunk.ID)
	require.Equal(t, "claude-3", chunk.Model)
	require.Equal(t, "chat.completion.chunk", chunk.Object)
	require.Len(t, chunk.Choices, 1)
	require.Equal(t, "Hello ", chunk.Choices[0].Delta.Content)

	// Verify content is accumulated.
	chunk2 := state.handleTextDelta("world!")
	require.NotNil(t, chunk2)
	require.Equal(t, "world!", chunk2.Choices[0].Delta.Content)
	require.Equal(t, "Hello world!", state.content.String())
}

func TestStreamStateHandleThinkingDelta(t *testing.T) {
	t.Parallel()

	state := newStreamState()
	state.messageID = "msg_123"
	state.model = "claude-3"

	chunk := state.handleThinkingDelta("Let me think...")
	require.NotNil(t, chunk)
	require.Equal(t, "msg_123", chunk.ID)
	require.Len(t, chunk.Choices, 1)
	require.NotNil(t, chunk.Choices[0].Delta.Reasoning)
	require.Equal(t, "Let me think...", chunk.Choices[0].Delta.Reasoning.Content)

	// Verify reasoning is accumulated.
	require.Equal(t, "Let me think...", state.reasoning.String())
}

func TestStreamStateHandleToolUseStart(t *testing.T) {
	t.Parallel()

	const toolCallID = "call_1"

	state := newStreamState()
	state.messageID = "msg_123"
	state.model = "claude-3"

	chunk := state.handleContentBlockStart(anthropic.ContentBlockStartEvent{
		ContentBlock: anthropic.ContentBlockStartEventContentBlockUnion{
			Type: blockTypeToolUse,
			ID:   toolCallID,
			Name: "get_weather",
		},
	})

	expected := providers.ToolCall{
		ID:   toolCallID,
		Type: "function",
		Function: providers.FunctionCall{
			Name: "get_weather",
		},
	}

	require.NotNil(t, chunk)
	require.Equal(t, 0, state.currentToolIdx)
	require.Equal(t, []providers.ToolCall{expected}, state.toolCalls)
	require.Equal(t, []providers.ToolCall{expected}, chunk.Choices[0].Delta.ToolCalls)
}

func TestStreamStatePreservesServerToolInputForReplay(t *testing.T) {
	t.Parallel()

	state := newStreamState()
	var start anthropic.ContentBlockStartEvent
	require.NoError(t, json.Unmarshal([]byte(`{
		"type":"content_block_start","index":0,
		"content_block":{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{}}
	}`), &start))
	startChunk := state.handleContentBlockStart(start)
	require.NotNil(t, startChunk)
	require.NotNil(t, startChunk.Choices[0].Delta.Extra[providerName]["response_blocks"])

	var delta anthropic.ContentBlockDeltaEvent
	require.NoError(t, json.Unmarshal([]byte(`{
		"type":"content_block_delta","index":0,
		"delta":{"type":"input_json_delta","partial_json":"{\"query\":\"any-llm\"}"}
	}`), &delta))
	inputChunk := state.handleContentBlockDelta(delta)
	require.NotNil(t, inputChunk)

	message := providers.Message{Role: providers.RoleAssistant, Extra: inputChunk.Choices[0].Delta.Extra}
	replayed, err := convertAssistantMessage(message)
	require.NoError(t, err)
	wire, err := json.Marshal(replayed)
	require.NoError(t, err)
	require.Contains(t, string(wire), `"type":"server_tool_use"`)
	require.Contains(t, string(wire), `"query":"any-llm"`)
}

func TestStreamStateHandleInputJSONDelta(t *testing.T) {
	t.Parallel()

	t.Run("returns nil when no tool calls", func(t *testing.T) {
		t.Parallel()

		state := newStreamState()
		chunk := state.handleInputJSONDelta(`{"key":`)
		require.Nil(t, chunk)
	})

	t.Run("returns nil when tool index out of bounds", func(t *testing.T) {
		t.Parallel()

		state := newStreamState()
		state.currentToolIdx = 5 // Out of bounds.
		state.toolCalls = []providers.ToolCall{
			{ID: "call_1", Type: "function", Function: providers.FunctionCall{Name: "get_weather", Arguments: ""}},
		}
		chunk := state.handleInputJSONDelta(`{"key":`)
		require.Nil(t, chunk)
	})

	t.Run("appends to current tool call arguments", func(t *testing.T) {
		t.Parallel()

		state := newStreamState()
		state.messageID = "msg_123"
		state.model = "claude-3"
		state.currentToolIdx = 0
		state.toolCalls = []providers.ToolCall{
			{ID: "call_1", Type: "function", Function: providers.FunctionCall{Name: "get_weather", Arguments: ""}},
		}

		chunk := state.handleInputJSONDelta(`{"location":`)
		require.NotNil(t, chunk)
		require.Equal(t, `{"location":`, state.toolCalls[0].Function.Arguments)
		require.Equal(t, `{"location":`, chunk.Choices[0].Delta.ToolCalls[0].Function.Arguments)

		chunk2 := state.handleInputJSONDelta(`"Paris"}`)
		require.NotNil(t, chunk2)
		require.Equal(t, `{"location":"Paris"}`, state.toolCalls[0].Function.Arguments)
		require.Equal(t, `"Paris"}`, chunk2.Choices[0].Delta.ToolCalls[0].Function.Arguments)
	})
}

func TestStreamStateHandleCitationDelta(t *testing.T) {
	t.Parallel()

	var event anthropic.ContentBlockDeltaEvent
	require.NoError(t, json.Unmarshal([]byte(`{
		"type":"content_block_delta",
		"index":2,
		"delta":{"type":"citations_delta","citation":{
			"type":"char_location","cited_text":"evidence","document_index":0,
			"document_title":"source","start_char_index":1,"end_char_index":9
		}}
	}`), &event))

	chunk := newStreamState().handleContentBlockDelta(event)
	require.NotNil(t, chunk)
	data := chunk.Choices[0].Delta.Extra[providerName]
	require.Equal(t, int64(2), data["index"])
	citations, ok := data["citations"].([]any)
	require.True(t, ok)
	require.Len(t, citations, 1)
	citation, ok := citations[0].(anthropic.CitationsDeltaCitationUnion)
	require.True(t, ok)
	require.Equal(t, "char_location", citation.Type)
	require.Equal(t, "evidence", citation.CitedText)
	require.Equal(t, int64(1), citation.StartCharIndex)
	require.Equal(t, int64(9), citation.EndCharIndex)
}

func TestApplyThinking(t *testing.T) {
	t.Parallel()

	t.Run("empty effort leaves thinking unset", func(t *testing.T) {
		t.Parallel()

		req := &anthropic.MessageNewParams{MaxTokens: 1000}
		require.NoError(t, applyThinking(req, ""))
		require.Equal(t, int64(1000), req.MaxTokens)
		require.Nil(t, req.Thinking.OfDisabled)
		require.Nil(t, req.Thinking.OfAdaptive)
		require.Empty(t, req.OutputConfig.Effort)
	})

	t.Run("ReasoningEffortNone disables thinking", func(t *testing.T) {
		t.Parallel()

		req := &anthropic.MessageNewParams{MaxTokens: 1000}
		require.NoError(t, applyThinking(req, providers.ReasoningEffortNone))
		require.NotNil(t, req.Thinking.OfDisabled)
		require.Empty(t, req.OutputConfig.Effort)
	})

	t.Run("ReasoningEffortAuto leaves thinking unset", func(t *testing.T) {
		t.Parallel()

		req := &anthropic.MessageNewParams{MaxTokens: 1000}
		require.NoError(t, applyThinking(req, providers.ReasoningEffortAuto))
		require.Nil(t, req.Thinking.OfDisabled)
		require.Nil(t, req.Thinking.OfAdaptive)
		require.Empty(t, req.OutputConfig.Effort)
	})

	t.Run("invalid effort returns UnsupportedParamError", func(t *testing.T) {
		t.Parallel()

		req := &anthropic.MessageNewParams{MaxTokens: 1000}
		err := applyThinking(req, "invalid")
		require.Error(t, err)
		require.ErrorIs(t, err, errors.ErrUnsupportedParam)
		require.Equal(t, int64(1000), req.MaxTokens)
	})

	t.Run("low effort uses adaptive thinking and does not change max tokens", func(t *testing.T) {
		t.Parallel()

		req := &anthropic.MessageNewParams{MaxTokens: 1000}
		require.NoError(t, applyThinking(req, providers.ReasoningEffortLow))
		require.Equal(t, int64(1000), req.MaxTokens)
		require.NotNil(t, req.Thinking.OfAdaptive)
		require.Equal(t, anthropic.OutputConfigEffortLow, req.OutputConfig.Effort)
	})

	t.Run("medium effort maps to medium", func(t *testing.T) {
		t.Parallel()

		req := &anthropic.MessageNewParams{}
		require.NoError(t, applyThinking(req, providers.ReasoningEffortMedium))
		require.NotNil(t, req.Thinking.OfAdaptive)
		require.Equal(t, anthropic.OutputConfigEffortMedium, req.OutputConfig.Effort)
	})

	t.Run("high effort maps to high", func(t *testing.T) {
		t.Parallel()

		req := &anthropic.MessageNewParams{}
		require.NoError(t, applyThinking(req, providers.ReasoningEffortHigh))
		require.Equal(t, anthropic.OutputConfigEffortHigh, req.OutputConfig.Effort)
	})

	t.Run("xhigh stays xhigh", func(t *testing.T) {
		t.Parallel()

		req := &anthropic.MessageNewParams{}
		require.NoError(t, applyThinking(req, "xhigh"))
		require.NotNil(t, req.Thinking.OfAdaptive)
		require.Equal(t, anthropic.OutputConfigEffortXhigh, req.OutputConfig.Effort)
	})

	t.Run("max stays max", func(t *testing.T) {
		t.Parallel()

		req := &anthropic.MessageNewParams{}
		require.NoError(t, applyThinking(req, "max"))
		require.Equal(t, anthropic.OutputConfigEffortMax, req.OutputConfig.Effort)
	})

	t.Run("preserves existing OutputConfig format", func(t *testing.T) {
		t.Parallel()

		schema := map[string]any{"type": "object"}
		req := &anthropic.MessageNewParams{
			OutputConfig: anthropic.OutputConfigParam{
				Format: anthropic.JSONOutputFormatParam{Schema: schema},
			},
		}
		require.NoError(t, applyThinking(req, providers.ReasoningEffortLow))
		require.Equal(t, schema, req.OutputConfig.Format.Schema)
		require.Equal(t, anthropic.OutputConfigEffortLow, req.OutputConfig.Effort)
	})
}

func TestConvertMessage(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		msg       providers.Message
		expectNil bool
	}{
		{
			name:      "system role returns nil",
			msg:       providers.Message{Role: providers.RoleSystem, Content: "System prompt"},
			expectNil: true,
		},
		{
			name:      "unknown role returns nil",
			msg:       providers.Message{Role: "unknown", Content: "Content"},
			expectNil: true,
		},
		{
			name:      "user role converts",
			msg:       providers.Message{Role: providers.RoleUser, Content: "Hello"},
			expectNil: false,
		},
		{
			name:      "assistant role converts",
			msg:       providers.Message{Role: providers.RoleAssistant, Content: "Hi there!"},
			expectNil: false,
		},
		{
			name:      "tool role converts",
			msg:       providers.Message{Role: providers.RoleTool, Content: "Result", ToolCallID: "call_123"},
			expectNil: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result, err := convertMessage(tc.msg)
			if tc.expectNil {
				require.Error(t, err)
				require.Nil(t, result)
			} else {
				require.NoError(t, err)
				require.NotNil(t, result)
			}
		})
	}
}

func TestConvertToolCall(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		toolCall  providers.ToolCall
		wantInput map[string]any
		wantErr   bool
	}{
		{
			name: "valid JSON arguments",
			toolCall: providers.ToolCall{
				ID:   "call_123",
				Type: "function",
				Function: providers.FunctionCall{
					Name:      "get_weather",
					Arguments: `{"location": "Paris"}`,
				},
			},
			wantInput: map[string]any{"location": "Paris"},
			wantErr:   false,
		},
		{
			name: "invalid JSON arguments return an error",
			toolCall: providers.ToolCall{
				ID:   "call_456",
				Type: "function",
				Function: providers.FunctionCall{
					Name:      "get_weather",
					Arguments: `{invalid json`,
				},
			},
			wantInput: nil,
			wantErr:   true,
		},
		{
			name: "empty arguments become an empty object",
			toolCall: providers.ToolCall{
				ID:   "call_789",
				Type: "function",
				Function: providers.FunctionCall{
					Name:      "get_weather",
					Arguments: "",
				},
			},
			wantInput: map[string]any{},
			wantErr:   false,
		},
		{
			name: "whitespace arguments become an empty object",
			toolCall: providers.ToolCall{
				ID:   "call_whitespace",
				Type: "function",
				Function: providers.FunctionCall{
					Name:      "get_weather",
					Arguments: " \n\t ",
				},
			},
			wantInput: map[string]any{},
			wantErr:   false,
		},
		{
			name: "null arguments return an error",
			toolCall: providers.ToolCall{
				ID:   "call_null",
				Type: "function",
				Function: providers.FunctionCall{
					Name:      "get_weather",
					Arguments: "null",
				},
			},
			wantInput: nil,
			wantErr:   true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result, err := convertToolCall(tc.toolCall)
			if tc.wantErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			require.NotNil(t, result.OfToolUse)
			require.Equal(t, tc.toolCall.ID, result.OfToolUse.ID)
			require.Equal(t, tc.toolCall.Function.Name, result.OfToolUse.Name)
			require.Equal(t, "tool_use", string(result.OfToolUse.Type))
			require.Equal(t, tc.wantInput, result.OfToolUse.Input)
		})
	}
}

func TestConvertTool(t *testing.T) {
	t.Parallel()

	t.Run("converts tool with properties and required fields", func(t *testing.T) {
		t.Parallel()

		tool := testutil.WeatherTool()
		result, err := convertTool(tool)

		require.NoError(t, err)
		require.NotNil(t, result.OfTool)
		require.Equal(t, "get_weather", result.OfTool.Name)
		require.Equal(t, "Get the current weather for a location.", result.OfTool.Description.Value)
		require.Equal(t, "object", string(result.OfTool.InputSchema.Type))

		// Verify properties are preserved.
		props, ok := result.OfTool.InputSchema.Properties.(map[string]any)
		require.True(t, ok, "properties should be a map")
		require.Contains(t, props, "location")

		locationProp, ok := props["location"].(map[string]any)
		require.True(t, ok, "location property should be a map")
		require.Equal(t, "string", locationProp["type"])
		require.Equal(t, "The city name, e.g. 'Paris, France'", locationProp["description"])

		// Verify required fields are preserved.
		require.Contains(t, result.OfTool.InputSchema.Required, "location")
	})

	t.Run("converts tool with multiple parameters", func(t *testing.T) {
		t.Parallel()

		tool := testutil.NewTestCalculatorTool(t)
		result, err := convertTool(tool)

		require.NoError(t, err)
		require.NotNil(t, result.OfTool)
		require.Equal(t, "calculate", result.OfTool.Name)
		require.Equal(t, "object", string(result.OfTool.InputSchema.Type))

		// Verify all properties are preserved.
		props, ok := result.OfTool.InputSchema.Properties.(map[string]any)
		require.True(t, ok, "properties should be a map")
		require.Contains(t, props, "a")
		require.Contains(t, props, "b")
		require.Contains(t, props, "operation")

		// Verify property types.
		aProp, ok := props["a"].(map[string]any)
		require.True(t, ok)
		require.Equal(t, "number", aProp["type"])

		bProp, ok := props["b"].(map[string]any)
		require.True(t, ok)
		require.Equal(t, "number", bProp["type"])

		opProp, ok := props["operation"].(map[string]any)
		require.True(t, ok)
		require.Equal(t, "string", opProp["type"])

		// Verify enum values are preserved.
		enum, ok := opProp["enum"].([]string)
		require.True(t, ok, "enum should be a string slice")
		require.ElementsMatch(t, []string{"add", "subtract", "multiply", "divide"}, enum)

		// Verify all required fields are preserved.
		require.Len(t, result.OfTool.InputSchema.Required, 3)
		require.Contains(t, result.OfTool.InputSchema.Required, "a")
		require.Contains(t, result.OfTool.InputSchema.Required, "b")
		require.Contains(t, result.OfTool.InputSchema.Required, "operation")
	})

	t.Run("converts tool with no required fields", func(t *testing.T) {
		t.Parallel()

		tool := providers.Tool{
			Type: "function",
			Function: providers.Function{
				Name:        "optional_params",
				Description: "A tool with optional parameters.",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"optional_field": map[string]any{
							"type":        "string",
							"description": "An optional field",
						},
					},
					// No "required" field.
				},
			},
		}
		result, err := convertTool(tool)

		require.NoError(t, err)
		require.NotNil(t, result.OfTool)
		require.Equal(t, "optional_params", result.OfTool.Name)
		require.Empty(t, result.OfTool.InputSchema.Required)
	})

	t.Run("converts tool with empty parameters", func(t *testing.T) {
		t.Parallel()

		tool := testutil.DateTool()
		result, err := convertTool(tool)

		require.NoError(t, err)
		require.NotNil(t, result.OfTool)
		require.Equal(t, "get_current_date", result.OfTool.Name)
		require.Equal(t, "object", string(result.OfTool.InputSchema.Type))
	})

	t.Run("returns error for invalid required field type", func(t *testing.T) {
		t.Parallel()

		tool := providers.Tool{
			Type: "function",
			Function: providers.Function{
				Name:        "bad_tool",
				Description: "A tool with invalid required field.",
				Parameters: map[string]any{
					"type":       "object",
					"properties": map[string]any{},
					"required":   123, // Invalid type.
				},
			},
		}
		_, err := convertTool(tool)

		require.Error(t, err)
		require.Contains(t, err.Error(), "bad_tool")
		require.Contains(t, err.Error(), "invalid required field")
	})

	t.Run("returns error for non-string element in required array", func(t *testing.T) {
		t.Parallel()

		tool := providers.Tool{
			Type: "function",
			Function: providers.Function{
				Name:        "mixed_required",
				Description: "A tool with mixed types in required.",
				Parameters: map[string]any{
					"type":       "object",
					"properties": map[string]any{},
					"required":   []any{"valid", 42, "also_valid"}, // Mixed types.
				},
			},
		}
		_, err := convertTool(tool)

		require.Error(t, err)
		require.Contains(t, err.Error(), "mixed_required")
		require.Contains(t, err.Error(), "element 1")
	})
}

func TestThinkingEffort(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		effort   providers.ReasoningEffort
		expected anthropic.OutputConfigEffort
		ok       bool
	}{
		{name: "minimal maps to low", effort: "minimal", expected: anthropic.OutputConfigEffortLow, ok: true},
		{name: "low", effort: providers.ReasoningEffortLow, expected: anthropic.OutputConfigEffortLow, ok: true},
		{
			name:     "medium",
			effort:   providers.ReasoningEffortMedium,
			expected: anthropic.OutputConfigEffortMedium,
			ok:       true,
		},
		{name: "high", effort: providers.ReasoningEffortHigh, expected: anthropic.OutputConfigEffortHigh, ok: true},
		{name: "xhigh stays xhigh", effort: "xhigh", expected: anthropic.OutputConfigEffortXhigh, ok: true},
		{name: "max stays max", effort: "max", expected: anthropic.OutputConfigEffortMax, ok: true},
		{name: "none", effort: providers.ReasoningEffortNone, ok: false},
		{name: "invalid", effort: "invalid", ok: false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			level, ok := thinkingEffort(tc.effort)
			require.Equal(t, tc.ok, ok)
			require.Equal(t, tc.expected, level)
		})
	}
}

func TestToStringSlice(t *testing.T) {
	t.Parallel()

	t.Run("returns []string input unchanged", func(t *testing.T) {
		t.Parallel()

		input := []string{"a", "b", "c"}
		result, err := toStringSlice(input)

		require.NoError(t, err)
		require.Equal(t, input, result)
	})

	t.Run("converts []any with all strings", func(t *testing.T) {
		t.Parallel()

		input := []any{"x", "y", "z"}
		result, err := toStringSlice(input)

		require.NoError(t, err)
		require.Equal(t, []string{"x", "y", "z"}, result)
	})

	t.Run("converts empty []any to empty []string", func(t *testing.T) {
		t.Parallel()

		input := []any{}
		result, err := toStringSlice(input)

		require.NoError(t, err)
		require.Empty(t, result)
	})

	t.Run("returns error for []any with non-string element", func(t *testing.T) {
		t.Parallel()

		input := []any{"valid", 42, "another"}
		_, err := toStringSlice(input)

		require.Error(t, err)
		require.Contains(t, err.Error(), "element 1")
		require.Contains(t, err.Error(), "expected string")
		require.Contains(t, err.Error(), "int")
	})

	t.Run("returns error for unexpected type", func(t *testing.T) {
		t.Parallel()

		_, err := toStringSlice(123)

		require.Error(t, err)
		require.Contains(t, err.Error(), "expected []string or []any")
		require.Contains(t, err.Error(), "int")
	})

	t.Run("returns error for nil input", func(t *testing.T) {
		t.Parallel()

		_, err := toStringSlice(nil)

		require.Error(t, err)
		require.Contains(t, err.Error(), "expected []string or []any")
	})
}

// Integration tests - only run if API key is available.

func TestIntegrationCompletion(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey("anthropic") {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:    testutil.TestModel("anthropic"),
		Messages: testutil.SimpleMessages(),
	}

	resp, err := provider.Completion(ctx, params)
	require.NoError(t, err)

	require.NotEmpty(t, resp.ID)
	require.Equal(t, "chat.completion", resp.Object)
	require.Len(t, resp.Choices, 1)
	require.NotEmpty(t, resp.Choices[0].Message.Content)
	require.Equal(t, providers.RoleAssistant, resp.Choices[0].Message.Role)
	require.NotNil(t, resp.Usage)
	require.Greater(t, resp.Usage.TotalTokens, 0)
}

func TestIntegrationCompletionWithSystemMessage(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey("anthropic") {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:    testutil.TestModel("anthropic"),
		Messages: testutil.MessagesWithSystem(),
	}

	resp, err := provider.Completion(ctx, params)
	require.NoError(t, err)

	require.NotEmpty(t, resp.ID)
	require.Len(t, resp.Choices, 1)
	require.NotEmpty(t, resp.Choices[0].Message.Content)
}

func TestIntegrationCompletionStream(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey("anthropic") {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:    testutil.TestModel("anthropic"),
		Messages: testutil.SimpleMessages(),
		Stream:   true,
	}

	chunks, errs := provider.CompletionStream(ctx, params)

	var content strings.Builder
	chunkCount := 0

	for chunk := range chunks {
		chunkCount++
		require.Equal(t, "chat.completion.chunk", chunk.Object)
		if len(chunk.Choices) > 0 {
			content.WriteString(chunk.Choices[0].Delta.Content)
		}
	}

	err = <-errs
	require.NoError(t, err)

	require.Greater(t, chunkCount, 0)
	require.NotEmpty(t, content.String())
}

func TestIntegrationCompletionWithTools(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey("anthropic") {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:      testutil.TestModel("anthropic"),
		Messages:   testutil.ToolCallMessages(),
		Tools:      []providers.Tool{testutil.WeatherTool()},
		ToolChoice: "auto",
	}

	resp, err := provider.Completion(ctx, params)
	require.NoError(t, err)

	require.NotEmpty(t, resp.ID)
	require.Len(t, resp.Choices, 1)

	// The model should call the weather tool.
	if len(resp.Choices[0].Message.ToolCalls) > 0 {
		tc := resp.Choices[0].Message.ToolCalls[0]
		require.Equal(t, "get_weather", tc.Function.Name)
		require.Contains(t, strings.ToLower(tc.Function.Arguments), "paris")
		require.Equal(t, providers.FinishReasonToolCalls, resp.Choices[0].FinishReason)
	}
}

func TestIntegrationCompletionWithToolsParallelDisabled(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey("anthropic") {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	parallel := false
	ctx := context.Background()
	params := providers.CompletionParams{
		Model: testutil.TestModel("anthropic"),
		Messages: []providers.Message{
			{Role: providers.RoleUser, Content: "Get the weather in Paris and London"},
		},
		Tools:             []providers.Tool{testutil.WeatherTool()},
		ToolChoice:        "auto",
		ParallelToolCalls: &parallel,
	}

	resp, err := provider.Completion(ctx, params)
	require.NoError(t, err)

	require.NotEmpty(t, resp.ID)
	require.Len(t, resp.Choices, 1)
}

func TestIntegrationAgentLoop(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey("anthropic") {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	tools := []providers.Tool{testutil.WeatherTool()}

	// Step 1: Send initial message asking about weather.
	messages := []providers.Message{
		{Role: providers.RoleUser, Content: "What is the weather in Paris? Use the get_weather tool."},
	}

	resp, err := provider.Completion(ctx, providers.CompletionParams{
		Model:      testutil.TestModel("anthropic"),
		Messages:   messages,
		Tools:      tools,
		ToolChoice: "auto",
	})
	require.NoError(t, err)
	require.Len(t, resp.Choices, 1)

	// Step 2: Verify the model called the tool.
	require.NotEmpty(t, resp.Choices[0].Message.ToolCalls, "expected model to call get_weather tool")
	require.Equal(t, providers.FinishReasonToolCalls, resp.Choices[0].FinishReason)

	tc := resp.Choices[0].Message.ToolCalls[0]
	require.Equal(t, "get_weather", tc.Function.Name)
	require.NotEmpty(t, tc.ID)

	// Step 3: Parse the arguments - this verifies parameters were sent correctly.
	var args struct {
		Location string `json:"location"`
	}
	err = json.Unmarshal([]byte(tc.Function.Arguments), &args)
	require.NoError(t, err, "tool arguments should be valid JSON")
	require.NotEmpty(t, args.Location, "location argument should be present")
	require.Contains(t, strings.ToLower(args.Location), "paris")

	// Step 4: Add assistant message with tool call and tool result.
	messages = append(messages, resp.Choices[0].Message)
	messages = append(messages, providers.Message{
		Role:       providers.RoleTool,
		Content:    testutil.MockWeatherResult(t, args.Location),
		ToolCallID: tc.ID,
	})

	// Step 5: Continue conversation with tool result.
	resp, err = provider.Completion(ctx, providers.CompletionParams{
		Model:    testutil.TestModel("anthropic"),
		Messages: messages,
		Tools:    tools,
	})
	require.NoError(t, err)
	require.Len(t, resp.Choices, 1)

	// Step 6: Verify the model produced a final response.
	require.Equal(t, providers.FinishReasonStop, resp.Choices[0].FinishReason)
	contentStr, ok := resp.Choices[0].Message.Content.(string)
	require.True(t, ok, "expected string content in final response")
	require.NotEmpty(t, contentStr)
}

func TestIntegrationAgentLoopMultipleParams(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey("anthropic") {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	tools := []providers.Tool{testutil.NewTestCalculatorTool(t)}

	// Ask the model to use the calculator with specific values.
	messages := []providers.Message{
		{Role: providers.RoleUser, Content: "Use the calculate tool to add 15 and 27 together."},
	}

	resp, err := provider.Completion(ctx, providers.CompletionParams{
		Model:      testutil.TestModel("anthropic"),
		Messages:   messages,
		Tools:      tools,
		ToolChoice: "auto",
	})
	require.NoError(t, err)
	require.Len(t, resp.Choices, 1)

	// Verify the model called the tool with correct parameters.
	require.NotEmpty(t, resp.Choices[0].Message.ToolCalls, "expected model to call calculate tool")

	tc := resp.Choices[0].Message.ToolCalls[0]
	require.Equal(t, "calculate", tc.Function.Name)

	// Parse and verify all required parameters are present.
	var args struct {
		A         float64 `json:"a"`
		B         float64 `json:"b"`
		Operation string  `json:"operation"`
	}
	err = json.Unmarshal([]byte(tc.Function.Arguments), &args)
	require.NoError(t, err, "tool arguments should be valid JSON")

	// Verify the parameters - this catches "wrong order" bugs.
	require.Equal(t, 15.0, args.A, "first operand should be 15")
	require.Equal(t, 27.0, args.B, "second operand should be 27")
	require.Equal(t, "add", args.Operation, "operation should be 'add'")

	// Complete the agent loop with tool result.
	messages = append(messages, resp.Choices[0].Message)
	messages = append(messages, providers.Message{
		Role:       providers.RoleTool,
		Content:    testutil.MockCalculatorResult(t, args.A, args.B, args.Operation),
		ToolCallID: tc.ID,
	})

	resp, err = provider.Completion(ctx, providers.CompletionParams{
		Model:    testutil.TestModel("anthropic"),
		Messages: messages,
		Tools:    tools,
	})
	require.NoError(t, err)

	// Verify final response mentions the result.
	contentStr, ok := resp.Choices[0].Message.Content.(string)
	require.True(t, ok)
	require.Contains(t, contentStr, "42")
}

func TestIntegrationCompletionConversation(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey("anthropic") {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:    testutil.TestModel("anthropic"),
		Messages: testutil.ConversationMessages(),
	}

	resp, err := provider.Completion(ctx, params)
	require.NoError(t, err)

	require.NotEmpty(t, resp.ID)
	require.Len(t, resp.Choices, 1)

	// The model should remember the name "Alice".
	contentStr, ok := resp.Choices[0].Message.Content.(string)
	require.True(t, ok, "expected string content")
	require.Contains(t, strings.ToLower(contentStr), "alice")
}

func TestIntegrationCompletionReasoning(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey("anthropic") {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	model := testutil.ReasoningModel("anthropic")
	if model == "" {
		t.Skip("No reasoning model configured for anthropic")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model: model,
		Messages: []providers.Message{
			{Role: providers.RoleUser, Content: "Please say hello! Think very briefly before you respond."},
		},
		ReasoningEffort: providers.ReasoningEffortLow,
	}

	resp, err := provider.Completion(ctx, params)
	require.NoError(t, err)

	require.NotEmpty(t, resp.ID)
	require.Len(t, resp.Choices, 1)
	require.NotEmpty(t, resp.Choices[0].Message.Content)

	// With reasoning effort, we should get reasoning content.
	if resp.Choices[0].Message.Reasoning != nil {
		require.NotEmpty(t, resp.Choices[0].Message.Reasoning.Content)
	}
}

func TestIntegrationAgentLoopContinuation(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey("anthropic") {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()

	// Start with the agent loop messages (user asks, assistant calls tool, tool returns).
	messages := testutil.AgentLoopMessages()

	params := providers.CompletionParams{
		Model:    testutil.TestModel("anthropic"),
		Messages: messages,
		Tools:    []providers.Tool{testutil.WeatherTool()},
	}

	// The model should respond with the weather information.
	resp, err := provider.Completion(ctx, params)
	require.NoError(t, err)

	require.NotEmpty(t, resp.ID)
	require.Len(t, resp.Choices, 1)

	// Should have a content response (not another tool call).
	if contentStr, ok := resp.Choices[0].Message.Content.(string); ok && contentStr != "" {
		content := strings.ToLower(contentStr)
		// Should mention the weather or sunny.
		require.True(
			t,
			strings.Contains(content, "sunny") || strings.Contains(content, "weather") ||
				strings.Contains(content, "salvaterra"),
		)
	}
}

func TestIntegrationAuthenticationError(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey("anthropic") {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	provider, err := New(config.WithAPIKey("invalid-api-key"))
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:    "claude-3-5-haiku-latest",
		Messages: testutil.SimpleMessages(),
	}

	_, err = provider.Completion(ctx, params)
	require.Error(t, err)

	// Check that it's converted to an authentication error.
	var authErr *errors.AuthenticationError
	require.ErrorAs(t, err, &authErr)
}

func TestIntegrationCompletionWithStructuredOutput(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey("anthropic") {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"answer": map[string]any{"type": "string"},
		},
		"required": []any{"answer"},
	}

	ctx := context.Background()
	result, err := provider.Completion(ctx, providers.CompletionParams{
		Model: testutil.TestModel("anthropic"),
		Messages: []providers.Message{
			{Role: providers.RoleUser, Content: "What is 2+2? Respond using the provided schema."},
		},
		ResponseFormat: &providers.ResponseFormat{
			Type: responseFormatJSONSchema,
			JSONSchema: &providers.JSONSchema{
				Name:   "answer_schema",
				Schema: schema,
			},
		},
	})
	require.NoError(t, err)
	require.NotEmpty(t, result.ID)
	require.Len(t, result.Choices, 1)
	require.NotEmpty(t, result.Choices[0].Message.Content)

	contentStr, ok := result.Choices[0].Message.Content.(string)
	require.True(t, ok, "expected string content")

	var response map[string]any
	err = json.Unmarshal([]byte(contentStr), &response)
	require.NoError(t, err, "response should be valid JSON")
	require.Contains(t, response, "answer", "response should contain 'answer' key")
}

func TestConvertError(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		err          error
		wantSentinel error
	}{
		{
			name:         "nil error returns nil",
			err:          nil,
			wantSentinel: nil,
		},
		{
			name:         "non-API error becomes ProviderError",
			err:          stderrors.New("network timeout"),
			wantSentinel: errors.ErrProvider,
		},
		{
			name:         "401 status becomes AuthenticationError",
			err:          newTestAPIError(t, 401),
			wantSentinel: errors.ErrAuthentication,
		},
		{
			name:         "429 status becomes RateLimitError",
			err:          newTestAPIError(t, 429),
			wantSentinel: errors.ErrRateLimit,
		},
		{
			name:         "404 status becomes ModelNotFoundError",
			err:          newTestAPIError(t, 404),
			wantSentinel: errors.ErrModelNotFound,
		},
		{
			name:         "400 status becomes InvalidRequestError",
			err:          newTestAPIError(t, 400),
			wantSentinel: errors.ErrInvalidRequest,
		},
		{
			name:         "403 status becomes AuthenticationError",
			err:          newTestAPIError(t, 403),
			wantSentinel: errors.ErrAuthentication,
		},
		{
			name:         "500 status becomes ProviderError",
			err:          newTestAPIError(t, 500),
			wantSentinel: errors.ErrProvider,
		},
		{
			name:         "502 status becomes ProviderError",
			err:          newTestAPIError(t, 502),
			wantSentinel: errors.ErrProvider,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			p := &Provider{}
			result := p.ConvertError(tc.err)

			if tc.wantSentinel == nil {
				require.Nil(t, result)
				return
			}

			require.NotNil(t, result)
			require.True(t, stderrors.Is(result, tc.wantSentinel), "expected error to match %v", tc.wantSentinel)

			// Verify the provider name is set in the error message.
			require.Contains(t, result.Error(), "["+providerName+"]")
		})
	}
}

func TestConvertParams_ResponseFormat(t *testing.T) {
	t.Parallel()

	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"answer": map[string]any{"type": "string"},
		},
		"required": []any{"answer"},
	}

	p, err := New(config.WithAPIKey("test-key"))
	require.NoError(t, err)

	baseParams := func() providers.CompletionParams {
		return providers.CompletionParams{
			Model: "claude-3-5-haiku-20241022",
			Messages: []providers.Message{
				{Role: providers.RoleUser, Content: []providers.ContentPart{{Text: "hello"}}},
			},
		}
	}

	t.Run("omitted MaxTokens preserves the existing default", func(t *testing.T) {
		t.Parallel()

		result, err := p.convertParams(baseParams())
		require.NoError(t, err)
		require.Equal(t, int64(defaultMaxTokens), result.MaxTokens)
	})

	t.Run("nil ResponseFormat leaves OutputConfig unset", func(t *testing.T) {
		t.Parallel()

		params := baseParams()
		params.ResponseFormat = nil

		result, err := p.convertParams(params)
		require.NoError(t, err)
		require.Equal(t, anthropic.OutputConfigParam{}, result.OutputConfig)
		require.Nil(t, result.Thinking.OfDisabled)
		require.Nil(t, result.Thinking.OfAdaptive)
	})

	t.Run("none reasoning effort disables thinking", func(t *testing.T) {
		t.Parallel()

		params := baseParams()
		params.ReasoningEffort = providers.ReasoningEffortNone

		result, err := p.convertParams(params)
		require.NoError(t, err)
		require.NotNil(t, result.Thinking.OfDisabled)
		require.Nil(t, result.Thinking.OfAdaptive)
		require.Empty(t, result.OutputConfig.Effort)
	})

	t.Run("invalid reasoning effort returns UnsupportedParamError", func(t *testing.T) {
		t.Parallel()

		params := baseParams()
		params.ReasoningEffort = "invalid"

		_, err := p.convertParams(params)
		require.Error(t, err)
		require.ErrorIs(t, err, errors.ErrUnsupportedParam)
	})

	t.Run("json_object type is rejected", func(t *testing.T) {
		t.Parallel()

		params := baseParams()
		params.ResponseFormat = &providers.ResponseFormat{Type: responseFormatJSONObject}

		_, err := p.convertParams(params)
		require.Error(t, err)
	})

	t.Run("json_schema with nil JSONSchema is rejected", func(t *testing.T) {
		t.Parallel()

		params := baseParams()
		params.ResponseFormat = &providers.ResponseFormat{Type: responseFormatJSONSchema, JSONSchema: nil}

		_, err := p.convertParams(params)
		require.Error(t, err)
	})

	t.Run("json_schema with valid schema sets OutputConfig", func(t *testing.T) {
		t.Parallel()

		params := baseParams()
		params.ResponseFormat = &providers.ResponseFormat{
			Type: responseFormatJSONSchema,
			JSONSchema: &providers.JSONSchema{
				Name:   "answer_schema",
				Schema: schema,
			},
		}

		result, err := p.convertParams(params)
		require.NoError(t, err)
		require.Equal(t, schema, result.OutputConfig.Format.Schema)
	})

	t.Run("unsupported JSONSchema fields are not forwarded", func(t *testing.T) {
		t.Parallel()

		strict := true
		params := baseParams()
		params.ResponseFormat = &providers.ResponseFormat{
			Type: responseFormatJSONSchema,
			JSONSchema: &providers.JSONSchema{
				Name:        "answer_schema",
				Description: "A schema for answers",
				Strict:      &strict,
				Schema:      schema,
			},
		}

		result, err := p.convertParams(params)
		require.NoError(t, err)
		require.Equal(t, schema, result.OutputConfig.Format.Schema)
		// JSONOutputFormatParam only has Schema and Type fields;
		// Name, Description, and Strict have no destination.
		require.Equal(t, anthropic.JSONOutputFormatParam{Schema: schema}, result.OutputConfig.Format)
	})

	t.Run("reasoning effort merges into existing OutputConfig", func(t *testing.T) {
		t.Parallel()

		params := baseParams()
		params.ReasoningEffort = providers.ReasoningEffortMedium
		params.ResponseFormat = &providers.ResponseFormat{
			Type: responseFormatJSONSchema,
			JSONSchema: &providers.JSONSchema{
				Name:   "answer_schema",
				Schema: schema,
			},
		}

		result, err := p.convertParams(params)
		require.NoError(t, err)
		require.Equal(t, schema, result.OutputConfig.Format.Schema)
		require.Equal(t, anthropic.OutputConfigEffortMedium, result.OutputConfig.Effort)
		require.NotNil(t, result.Thinking.OfAdaptive)
	})

	t.Run("streaming path also receives OutputConfig", func(t *testing.T) {
		t.Parallel()

		params := baseParams()
		params.Stream = true
		params.ResponseFormat = &providers.ResponseFormat{
			Type: responseFormatJSONSchema,
			JSONSchema: &providers.JSONSchema{
				Name:   "answer_schema",
				Schema: schema,
			},
		}

		result, err := p.convertParams(params)
		require.NoError(t, err)
		require.Equal(t, schema, result.OutputConfig.Format.Schema)
	})
}

func TestConvertResponsePreservesThinkingSignature(t *testing.T) {
	t.Parallel()

	resp := &anthropic.Message{
		ID:    "msg_001",
		Model: anthropic.Model("claude-3-5-sonnet"),
		Content: []anthropic.ContentBlockUnion{
			{Type: blockTypeThinking, Thinking: "Let me think...", Signature: "sig_abc123"},
			{Type: blockTypeText, Text: "The answer is 42."},
		},
		StopReason: anthropic.StopReasonEndTurn,
		Usage:      anthropic.Usage{InputTokens: 10, OutputTokens: 20},
	}

	result, err := convertResponse(resp)
	require.NoError(t, err)
	require.NotNil(t, result)
	require.NotNil(t, result.Choices[0].Message.Reasoning)
	require.Equal(t, "Let me think...", result.Choices[0].Message.Reasoning.Content)
	require.Equal(t, "sig_abc123", result.Choices[0].Message.Reasoning.Signature)
	require.Equal(t, "The answer is 42.", result.Choices[0].Message.ContentString())
}

func TestConvertResponsePreservesTextCitations(t *testing.T) {
	t.Parallel()

	resp := &anthropic.Message{
		ID:    "msg_citation",
		Model: anthropic.Model("claude-test"),
		Content: []anthropic.ContentBlockUnion{{
			Type: blockTypeText,
			Text: "Cited answer.",
			Citations: []anthropic.TextCitationUnion{{
				Type:            "page_location",
				CitedText:       "source text",
				DocumentIndex:   0,
				StartPageNumber: 1,
				EndPageNumber:   2,
			}},
		}},
	}

	result, err := convertResponse(resp)
	require.NoError(t, err)
	data := result.Choices[0].Message.Extra[providerName]
	citations, ok := data["citations"].([]any)
	require.True(t, ok)
	require.Len(t, citations, 1)
	citation, ok := citations[0].(anthropic.TextCitationUnion)
	require.True(t, ok)
	require.Equal(t, "page_location", citation.Type)
	require.Equal(t, "source text", citation.CitedText)
	require.Equal(t, int64(1), citation.StartPageNumber)
	require.Equal(t, int64(2), citation.EndPageNumber)
}

func TestConvertResponsePreservesAndReplaysServerToolBlocks(t *testing.T) {
	t.Parallel()

	var response anthropic.Message
	require.NoError(t, json.Unmarshal([]byte(`{
		"id":"msg_server_tool","type":"message","role":"assistant","model":"claude-test",
		"stop_reason":"end_turn","stop_sequence":null,
		"content":[
			{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"any-llm"}},
			{"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":[{
				"type":"web_search_result","title":"any-llm","url":"https://example.test/any-llm",
				"encrypted_content":"encrypted-result"
			}]},
			{"type":"text","text":"Here is the answer."}
		],
		"usage":{"input_tokens":3,"output_tokens":4}
	}`), &response))

	result, err := convertResponse(&response)
	require.NoError(t, err)
	message := result.Choices[0].Message
	require.Equal(t, "Here is the answer.", message.Content)

	encoded, err := json.Marshal(message)
	require.NoError(t, err)
	var restored providers.Message
	require.NoError(t, json.Unmarshal(encoded, &restored))
	replayed, err := convertAssistantMessage(restored)
	require.NoError(t, err)
	require.Len(t, replayed.Content, 3)

	wire, err := json.Marshal(replayed)
	require.NoError(t, err)
	require.Contains(t, string(wire), `"type":"server_tool_use"`)
	require.Contains(t, string(wire), `"encrypted_content":"encrypted-result"`)
}

func TestServerToolBlockReplayRejectsConflictingNormalizedContent(t *testing.T) {
	t.Parallel()

	var response anthropic.Message
	require.NoError(t, json.Unmarshal([]byte(`{
		"id":"msg_123","type":"message","role":"assistant","model":"claude-sonnet-4-6",
		"content":[
			{"type":"server_tool_use","id":"srv_1","name":"web_search","input":{"query":"test"}},
			{"type":"text","text":"original"}
		],
		"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}
	}`), &response))

	completion, err := convertResponse(&response)
	require.NoError(t, err)
	message := completion.Choices[0].Message
	message.Content = "edited"

	_, err = convertAssistantMessage(message)
	require.ErrorIs(t, err, errors.ErrInvalidRequest)
}

func TestConvertResponseAccumulatesThinkingBlocks(t *testing.T) {
	t.Parallel()

	resp := &anthropic.Message{
		ID:    "msg_002",
		Model: anthropic.Model("claude-3-5-sonnet"),
		Content: []anthropic.ContentBlockUnion{
			{Type: blockTypeThinking, Thinking: "First thought.", Signature: "sig_1"},
			{Type: blockTypeThinking, Thinking: " Second thought.", Signature: "sig_2"},
		},
		StopReason: anthropic.StopReasonEndTurn,
		Usage:      anthropic.Usage{InputTokens: 5, OutputTokens: 10},
	}

	result, err := convertResponse(resp)
	require.NoError(t, err)
	require.NotNil(t, result.Choices[0].Message.Reasoning)
	require.Equal(t, "First thought. Second thought.", result.Choices[0].Message.Reasoning.Content)
	// Last non-empty signature wins, matching Python extra_content overwrite.
	require.Equal(t, "sig_2", result.Choices[0].Message.Reasoning.Signature)
}

func TestConvertResponsePreservesThinkingWithoutSignature(t *testing.T) {
	t.Parallel()

	resp := &anthropic.Message{
		ID:    "msg_003",
		Model: anthropic.Model("claude-3-5-sonnet"),
		Content: []anthropic.ContentBlockUnion{
			{Type: blockTypeThinking, Thinking: "No signature here."},
			{Type: blockTypeText, Text: "Response."},
		},
		StopReason: anthropic.StopReasonEndTurn,
		Usage:      anthropic.Usage{InputTokens: 5, OutputTokens: 10},
	}

	result, err := convertResponse(resp)
	require.NoError(t, err)
	require.NotNil(t, result.Choices[0].Message.Reasoning)
	require.Equal(t, "No signature here.", result.Choices[0].Message.Reasoning.Content)
	require.Empty(t, result.Choices[0].Message.Reasoning.Signature)
	require.Empty(t, result.Choices[0].Message.Extra)
}

func TestConvertAssistantMessageReplaysThinkingBeforeText(t *testing.T) {
	t.Parallel()

	msg := providers.Message{
		Role:      providers.RoleAssistant,
		Content:   "The answer.",
		Reasoning: &providers.Reasoning{Content: "I reasoned about this.", Signature: "sig_xyz"},
	}

	result, err := convertAssistantMessage(msg)
	require.NoError(t, err)
	require.NotNil(t, result)
	require.Len(t, result.Content, 2)
	require.NotNil(t, result.Content[0].OfThinking)
	require.Equal(t, "sig_xyz", result.Content[0].OfThinking.Signature)
	require.Equal(t, "I reasoned about this.", result.Content[0].OfThinking.Thinking)
	require.NotNil(t, result.Content[1].OfText)
	require.Equal(t, "The answer.", result.Content[1].OfText.Text)
}

func TestConvertAssistantMessageReplaysThinkingBeforeToolCalls(t *testing.T) {
	t.Parallel()

	msg := providers.Message{
		Role: providers.RoleAssistant,
		ToolCalls: []providers.ToolCall{
			{
				ID:       "call_1",
				Type:     "function",
				Function: providers.FunctionCall{Name: "get_weather", Arguments: `{}`},
			},
		},
		Reasoning: &providers.Reasoning{Content: "Need to check weather.", Signature: "sig_tool"},
	}

	result, err := convertAssistantMessage(msg)
	require.NoError(t, err)
	require.NotNil(t, result)
	require.Len(t, result.Content, 2)
	require.NotNil(t, result.Content[0].OfThinking)
	require.Equal(t, "sig_tool", result.Content[0].OfThinking.Signature)
	require.NotNil(t, result.Content[1].OfToolUse)
}

func TestConvertAssistantMessagePreservesMultimodalContentWithToolCalls(t *testing.T) {
	t.Parallel()

	msg := providers.Message{
		Role: providers.RoleAssistant,
		Content: []providers.ContentPart{
			{Type: blockTypeText, Text: "I found this image."},
			{Type: "image_url", ImageURL: &providers.ImageURL{URL: "https://example.com/result.png"}},
		},
		ToolCalls: []providers.ToolCall{
			{
				ID:       "call_1",
				Type:     "function",
				Function: providers.FunctionCall{Name: "inspect_image", Arguments: `{}`},
			},
		},
		Reasoning: &providers.Reasoning{Content: "Need to inspect the image.", Signature: "sig_image"},
	}

	result, err := convertAssistantMessage(msg)
	require.NoError(t, err)
	require.NotNil(t, result)
	require.Len(t, result.Content, 4)
	require.NotNil(t, result.Content[0].OfThinking)
	require.Equal(t, "I found this image.", result.Content[1].OfText.Text)
	require.Equal(t, "https://example.com/result.png", result.Content[2].OfImage.Source.OfURL.URL)
	require.NotNil(t, result.Content[3].OfToolUse)
}

func TestConvertAssistantMessageOmitsThinkingWithoutReasoning(t *testing.T) {
	t.Parallel()

	msg := providers.Message{
		Role:    providers.RoleAssistant,
		Content: "Plain response.",
	}

	result, err := convertAssistantMessage(msg)
	require.NoError(t, err)
	require.NotNil(t, result)
	require.Len(t, result.Content, 1)
	require.Nil(t, result.Content[0].OfThinking)
	require.NotNil(t, result.Content[0].OfText)
}

func TestConvertAssistantMessageOmitsUnsignedReasoning(t *testing.T) {
	t.Parallel()

	msg := providers.Message{
		Role:      providers.RoleAssistant,
		Content:   "Response.",
		Reasoning: &providers.Reasoning{Content: "Thought without signature."},
	}

	result, err := convertAssistantMessage(msg)
	require.NoError(t, err)
	require.NotNil(t, result)
	require.Len(t, result.Content, 1)
	require.Nil(t, result.Content[0].OfThinking)
	require.NotNil(t, result.Content[0].OfText)
}

func TestConvertAssistantMessageOmitsUnsignedThinkingMetadata(t *testing.T) {
	t.Parallel()

	msg := providers.Message{
		Role:    providers.RoleAssistant,
		Content: "Response.",
		Extra: map[string]providers.ProviderData{providerName: {"thinking_blocks": []any{
			map[string]any{"type": blockTypeThinking, "thinking": "Incomplete stream.", "signature": ""},
		}}},
	}

	result, err := convertAssistantMessage(msg)
	require.NoError(t, err)
	require.Len(t, result.Content, 1)
	require.Nil(t, result.Content[0].OfThinking)
	require.NotNil(t, result.Content[0].OfText)
}

func TestStreamStateHandleSignatureDelta(t *testing.T) {
	t.Parallel()

	state := newStreamState()
	state.messageID = "msg_123"
	state.model = "claude-3"
	state.handleContentBlockStart(anthropic.ContentBlockStartEvent{
		ContentBlock: anthropic.ContentBlockStartEventContentBlockUnion{Type: blockTypeThinking},
	})

	chunk := state.handleContentBlockDelta(anthropic.ContentBlockDeltaEvent{
		Delta: anthropic.RawContentBlockDeltaUnion{
			Type:      deltaTypeSignature,
			Signature: "sig_stream_abc",
		},
	})

	// signature_delta emits the signature at Anthropic's official event boundary.
	require.NotNil(t, chunk)
	blocks, ok := chunk.Choices[0].Delta.Extra[providerName]["thinking_blocks"].([]providers.ProviderData)
	require.True(t, ok)
	require.Equal(t, "sig_stream_abc", blocks[0]["signature"])
}

func TestStreamStateHandleMessageDelta_WithoutSignature(t *testing.T) {
	t.Parallel()

	state := newStreamState()
	state.messageID = "msg_123"
	state.model = "claude-3"

	chunk := state.handleMessageDelta(anthropic.MessageDeltaEvent{
		Delta: anthropic.MessageDeltaEventDelta{StopReason: anthropic.StopReasonEndTurn},
		Usage: anthropic.MessageDeltaUsage{OutputTokens: 42},
	})

	require.Nil(t, chunk.Choices[0].Delta.Reasoning)
}

func TestConvertMessages_AssistantWithThinkingSignatureRoundTrip(t *testing.T) {
	t.Parallel()

	// Simulate a multi-turn conversation: the assistant's previous thinking
	// (with signature) must be replayed so Anthropic maintains thinking continuity.
	messages := []providers.Message{
		{Role: providers.RoleUser, Content: "What is 2+2?"},
		{
			Role:      providers.RoleAssistant,
			Content:   "4",
			Reasoning: &providers.Reasoning{Content: "2 plus 2 equals 4.", Signature: "sig_roundtrip"},
		},
		{Role: providers.RoleUser, Content: "And 3+3?"},
	}

	result, system, err := convertMessages(messages)
	require.NoError(t, err)

	require.Empty(t, system)
	require.Len(t, result, 3)

	// The assistant message (index 1) must start with a thinking block.
	assistantMsg := result[1]
	require.Equal(t, anthropic.MessageParamRoleAssistant, assistantMsg.Role)
	require.NotEmpty(t, assistantMsg.Content)
	require.NotNil(t, assistantMsg.Content[0].OfThinking)
	require.Equal(t, "sig_roundtrip", assistantMsg.Content[0].OfThinking.Signature)
	require.Equal(t, "2 plus 2 equals 4.", assistantMsg.Content[0].OfThinking.Thinking)
}

// newTestAPIError creates an Anthropic API error for testing.
// Note: The raw JSON field is unexported, so we can only test status code based conversion.
func newTestAPIError(t *testing.T, statusCode int) *anthropic.Error {
	t.Helper()

	testURL, _ := url.Parse("https://api.anthropic.com/v1/messages")
	return &anthropic.Error{
		StatusCode: statusCode,
		RequestID:  "req_test123",
		Request:    &http.Request{Method: "POST", URL: testURL},
		Response:   &http.Response{StatusCode: statusCode},
	}
}
