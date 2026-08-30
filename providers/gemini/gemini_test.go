package gemini

import (
	"context"
	"encoding/base64"
	"encoding/json"
	stderrors "errors"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/genai"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/internal/testutil"
	"github.com/mozilla-ai/any-llm-go/providers"
)

const gemini3FlashPreviewModel = "gemini-3-flash-preview"

func TestNew(t *testing.T) {
	t.Run("creates provider with API key", func(t *testing.T) {
		provider, err := New(config.WithAPIKey("test-api-key"))
		require.NoError(t, err)
		require.NotNil(t, provider)
		require.Equal(t, providerName, provider.Name())
	})

	t.Run("creates provider from GEMINI_API_KEY", func(t *testing.T) {
		t.Setenv("GEMINI_API_KEY", "env-api-key")

		provider, err := New()
		require.NoError(t, err)
		require.NotNil(t, provider)
	})

	t.Run("creates provider from GOOGLE_API_KEY fallback", func(t *testing.T) {
		t.Setenv("GEMINI_API_KEY", "")
		t.Setenv("GOOGLE_API_KEY", "google-api-key")

		provider, err := New()
		require.NoError(t, err)
		require.NotNil(t, provider)
	})

	t.Run("returns error when API key is missing", func(t *testing.T) {
		t.Setenv("GEMINI_API_KEY", "")
		t.Setenv("GOOGLE_API_KEY", "")

		provider, err := New()
		require.Nil(t, provider)
		require.Error(t, err)

		var missingKeyErr *errors.MissingAPIKeyError
		require.ErrorAs(t, err, &missingKeyErr)
		require.Equal(t, providerName, missingKeyErr.Provider)
		require.Equal(t, envAPIKey, missingKeyErr.EnvVar)
	})

	t.Run("creates provider with custom base URL", func(t *testing.T) {
		t.Parallel()

		provider, err := New(
			config.WithAPIKey("test-api-key"),
			config.WithBaseURL("https://gemini-proxy.example"),
		)
		require.NoError(t, err)
		require.NotNil(t, provider)
	})
}

func TestInt32Conversions(t *testing.T) {
	t.Parallel()

	value, err := int32Value(math.MaxInt32, "seed")
	require.NoError(t, err)
	require.Equal(t, int32(math.MaxInt32), value)

	if strconv.IntSize == 64 {
		maxInt32 := int64(math.MaxInt32)
		_, err = int32Value(int(maxInt32+1), "seed")
		require.Error(t, err)
	}

	_, err = positiveInt32(0, "max_tokens")
	require.Error(t, err)
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
	require.True(t, caps.Embedding)
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

		require.NotNil(t, system)
		require.Len(t, system.Parts, 1)
		require.Equal(t, "You are a helpful assistant.", system.Parts[0].Text)
		require.Len(t, result, 1)
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

		require.NotNil(t, system)
		require.Equal(t, "First part.", system.Parts[0].Text)
		require.Equal(t, "Second part.", system.Parts[1].Text)
		require.Len(t, result, 1)
	})

	t.Run("preserves ordered system text parts", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{{
			Role: providers.RoleSystem,
			Content: []providers.ContentPart{
				{Type: contentPartTypeText, Text: "first"},
				{Type: contentPartTypeText, Text: "second"},
			},
		}}

		_, system, err := convertMessages(messages)
		require.NoError(t, err)
		require.Equal(t, "first", system.Parts[0].Text)
		require.Equal(t, "second", system.Parts[1].Text)
	})

	for _, part := range []providers.ContentPart{
		{Type: contentPartTypeImageURL, ImageURL: &providers.ImageURL{URL: "https://example.com/image.png"}},
		{Type: contentPartTypeFile, File: &providers.File{FileData: "https://example.com/file.pdf"}},
		{Type: "audio"},
	} {
		t.Run("rejects unsupported system part "+part.Type, func(t *testing.T) {
			t.Parallel()

			_, _, err := convertMessages([]providers.Message{{
				Role:    providers.RoleSystem,
				Content: []providers.ContentPart{part},
			}})
			require.ErrorIs(t, err, errors.ErrInvalidRequest)
		})
	}

	t.Run("converts user message", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleUser, Content: "Hello"},
		}

		result, system, err := convertMessages(messages)
		require.NoError(t, err)

		require.Nil(t, system)
		require.Len(t, result, 1)
		require.Equal(t, "user", result[0].Role)
	})

	t.Run("converts assistant message to model role", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleUser, Content: "Hello"},
			{Role: providers.RoleAssistant, Content: "Hi there!"},
		}

		result, system, err := convertMessages(messages)
		require.NoError(t, err)

		require.Nil(t, system)
		require.Len(t, result, 2)
		require.Equal(t, roleModel, result[1].Role)
		require.Equal(t, "Hi there!", result[1].Parts[0].Text)
	})

	for name, content := range map[string]any{
		"empty string": "",
		"empty list":   []providers.ContentPart{},
	} {
		t.Run("preserves assistant "+name, func(t *testing.T) {
			t.Parallel()

			result, _, err := convertMessages([]providers.Message{{Role: providers.RoleAssistant, Content: content}})
			require.NoError(t, err)
			require.Len(t, result, 1)
			require.Equal(t, roleModel, result[0].Role)
		})
	}

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
		require.Equal(t, roleModel, result[1].Role)
		require.NotNil(t, result[1].Parts[0].FunctionCall)
		require.Equal(t, "get_weather", result[1].Parts[0].FunctionCall.Name)
		require.Equal(t, "call_123", result[1].Parts[0].FunctionCall.ID)
	})

	t.Run("converts tool result message with plain text", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleUser, Content: "Hello"},
			{
				Role:       providers.RoleTool,
				Content:    "sunny, 22°C",
				Name:       "get_weather",
				ToolCallID: "call_123",
			},
		}

		result, _, err := convertMessages(messages)
		require.NoError(t, err)

		require.Len(t, result, 2)
		require.Equal(t, "user", result[1].Role)
		require.NotNil(t, result[1].Parts[0].FunctionResponse)
		require.Equal(t, "get_weather", result[1].Parts[0].FunctionResponse.Name)
		require.Equal(t, "call_123", result[1].Parts[0].FunctionResponse.ID)
		// Plain text is wrapped as {"result": "sunny, 22°C"}.
		require.Equal(t, "sunny, 22°C", result[1].Parts[0].FunctionResponse.Response["result"])
	})

	t.Run("converts tool result message with JSON content", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleUser, Content: "Hello"},
			{Role: providers.RoleTool, Content: `{"temperature": 22, "condition": "sunny"}`, Name: "get_weather"},
		}

		result, _, err := convertMessages(messages)
		require.NoError(t, err)

		require.Len(t, result, 2)
		require.NotNil(t, result[1].Parts[0].FunctionResponse)
		// JSON content is parsed directly.
		require.Equal(t, "sunny", result[1].Parts[0].FunctionResponse.Response["condition"])
	})

	t.Run("converts tool result message with fallback name", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleTool, Content: "result data"},
		}

		result, _, err := convertMessages(messages)
		require.NoError(t, err)

		require.Len(t, result, 1)
		require.Equal(t, "function", result[0].Parts[0].FunctionResponse.Name)
	})

	t.Run("no system returns nil instruction", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleUser, Content: "Hello"},
		}

		_, system, err := convertMessages(messages)
		require.NoError(t, err)
		require.Nil(t, system)
	})

	t.Run("replays thought signature from Extra", func(t *testing.T) {
		t.Parallel()

		sig := base64.StdEncoding.EncodeToString([]byte("real-signature"))
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
						Extra: map[string]providers.ProviderData{
							providerName: {extraKeyThoughtSignature: sig},
						},
					},
				},
			},
		}

		result, _, err := convertMessages(messages)
		require.NoError(t, err)

		require.Len(t, result, 2)
		require.Equal(t, roleModel, result[1].Role)
		require.NotNil(t, result[1].Parts[0].FunctionCall)
		require.Equal(t, []byte("real-signature"), result[1].Parts[0].ThoughtSignature)
	})

	t.Run("preserves assistant content parts beside tool calls", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{{
			Role: providers.RoleAssistant,
			Content: []providers.ContentPart{
				{Type: contentPartTypeText, Text: "I found this image."},
				{Type: contentPartTypeImageURL, ImageURL: &providers.ImageURL{URL: "https://example.com/result.png"}},
			},
			ToolCalls: []providers.ToolCall{{
				ID:       "call_123",
				Type:     "function",
				Function: providers.FunctionCall{Name: "inspect_image", Arguments: `{}`},
			}},
		}}

		result, _, err := convertMessages(messages)
		require.NoError(t, err)
		require.Len(t, result, 1)
		require.Len(t, result[0].Parts, 3)
		require.Equal(t, "I found this image.", result[0].Parts[0].Text)
		require.Equal(t, "https://example.com/result.png", result[0].Parts[1].FileData.FileURI)
		require.Equal(t, "inspect_image", result[0].Parts[2].FunctionCall.Name)
	})

	t.Run("converts tool call with empty arguments", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{{
			Role: providers.RoleAssistant,
			ToolCalls: []providers.ToolCall{{
				ID:       "call_123",
				Type:     "function",
				Function: providers.FunctionCall{Name: "get_weather"},
			}},
		}}

		result, _, err := convertMessages(messages)
		require.NoError(t, err)

		require.Len(t, result, 1)
		require.Empty(t, result[0].Parts[0].FunctionCall.Args)
	})

	t.Run("tool call without signature uses bypass value", func(t *testing.T) {
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
		require.NotNil(t, result[1].Parts[0].FunctionCall)
		require.Equal(t, []byte(thoughtSignatureBypass), result[1].Parts[0].ThoughtSignature)
	})

	t.Run("public conversion rejects unknown role", func(t *testing.T) {
		t.Parallel()

		provider := &Provider{}
		contents, cfg, err := provider.convertParams(providers.CompletionParams{
			Model:    "gemini-2.5-flash",
			Messages: []providers.Message{{Role: "unknown", Content: "Hello"}},
		})
		require.Error(t, err)
		require.Nil(t, contents)
		require.Nil(t, cfg)
	})
}

func TestConvertFinishReason(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		input    genai.FinishReason
		expected string
	}{
		{
			name:     "STOP",
			input:    genai.FinishReasonStop,
			expected: providers.FinishReasonStop,
		},
		{
			name:     "MAX_TOKENS",
			input:    genai.FinishReasonMaxTokens,
			expected: providers.FinishReasonLength,
		},
		{
			name:     "TOO_MANY_TOOL_CALLS",
			input:    genai.FinishReasonTooManyToolCalls,
			expected: providers.FinishReasonToolCalls,
		},
		{
			name:     "SAFETY",
			input:    genai.FinishReasonSafety,
			expected: providers.FinishReasonContentFilter,
		},
		{
			name:     "RECITATION",
			input:    genai.FinishReasonRecitation,
			expected: providers.FinishReasonContentFilter,
		},
		{
			name:     "BLOCKLIST",
			input:    genai.FinishReasonBlocklist,
			expected: providers.FinishReasonContentFilter,
		},
		{
			name:     "PROHIBITED_CONTENT",
			input:    genai.FinishReasonProhibitedContent,
			expected: providers.FinishReasonContentFilter,
		},
		{
			name:     "unknown",
			input:    "UNKNOWN",
			expected: providers.FinishReasonStop,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result := convertFinishReason(tc.input)
			require.Equal(t, tc.expected, result)
		})
	}
}

func TestConvertTools(t *testing.T) {
	t.Parallel()

	tools := []providers.Tool{
		{
			Type: "function",
			Function: providers.Function{
				Name:        "get_weather",
				Description: "Get the weather for a location.",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"location": map[string]any{
							"type":        "string",
							"description": "The city name",
						},
					},
					"required": []string{"location"},
				},
			},
		},
	}

	result := convertTools(tools)

	require.Len(t, result, 1)
	require.Len(t, result[0].FunctionDeclarations, 1)
	require.Equal(t, "get_weather", result[0].FunctionDeclarations[0].Name)
	require.Equal(t, "Get the weather for a location.", result[0].FunctionDeclarations[0].Description)
	require.NotNil(t, result[0].FunctionDeclarations[0].ParametersJsonSchema)
}

func TestConvertToolChoice(t *testing.T) {
	t.Parallel()

	t.Run("auto string", func(t *testing.T) {
		t.Parallel()

		result, err := convertToolChoice("auto")
		require.NoError(t, err)
		require.NotNil(t, result)
		require.Equal(t, genai.FunctionCallingConfigModeAuto, result.FunctionCallingConfig.Mode)
	})

	t.Run("none string", func(t *testing.T) {
		t.Parallel()

		result, err := convertToolChoice("none")
		require.NoError(t, err)
		require.NotNil(t, result)
		require.Equal(t, genai.FunctionCallingConfigModeNone, result.FunctionCallingConfig.Mode)
	})

	t.Run("required string", func(t *testing.T) {
		t.Parallel()

		result, err := convertToolChoice("required")
		require.NoError(t, err)
		require.NotNil(t, result)
		require.Equal(t, genai.FunctionCallingConfigModeAny, result.FunctionCallingConfig.Mode)
	})

	t.Run("specific function", func(t *testing.T) {
		t.Parallel()

		result, err := convertToolChoice(providers.ToolChoice{
			Type:     "function",
			Function: &providers.ToolChoiceFunction{Name: "get_weather"},
		})
		require.NoError(t, err)
		require.NotNil(t, result)
		require.Equal(t, genai.FunctionCallingConfigModeAny, result.FunctionCallingConfig.Mode)
		require.Contains(t, result.FunctionCallingConfig.AllowedFunctionNames, "get_weather")
	})

	t.Run("unknown returns error", func(t *testing.T) {
		t.Parallel()

		result, err := convertToolChoice("unknown_value")
		require.Nil(t, result)
		require.Error(t, err)
	})
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
			err:          &genai.APIError{Code: 401, Message: "unauthorized"},
			wantSentinel: errors.ErrAuthentication,
		},
		{
			name:         "403 status becomes AuthenticationError",
			err:          &genai.APIError{Code: 403, Message: "forbidden"},
			wantSentinel: errors.ErrAuthentication,
		},
		{
			name:         "404 status becomes ModelNotFoundError",
			err:          &genai.APIError{Code: 404, Message: "not found"},
			wantSentinel: errors.ErrModelNotFound,
		},
		{
			name:         "429 status becomes RateLimitError",
			err:          &genai.APIError{Code: 429, Message: "rate limited"},
			wantSentinel: errors.ErrRateLimit,
		},
		{
			name:         "400 status becomes InvalidRequestError",
			err:          &genai.APIError{Code: 400, Message: "bad request"},
			wantSentinel: errors.ErrInvalidRequest,
		},
		{
			name:         "400 with context message becomes ContextLengthError",
			err:          &genai.APIError{Code: 400, Message: "context length exceeded"},
			wantSentinel: errors.ErrContextLength,
		},
		{
			name:         "400 with token message becomes ContextLengthError",
			err:          &genai.APIError{Code: 400, Message: "too many tokens in request"},
			wantSentinel: errors.ErrContextLength,
		},
		{
			name:         "400 with safety message becomes ContentFilterError",
			err:          &genai.APIError{Code: 400, Message: "blocked by safety filters"},
			wantSentinel: errors.ErrContentFilter,
		},
		{
			name:         "500 status becomes ProviderError",
			err:          &genai.APIError{Code: 500, Message: "internal error"},
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
			require.True(
				t,
				stderrors.Is(result, tc.wantSentinel),
				"expected error to match %v, got %v",
				tc.wantSentinel,
				result,
			)
			require.Contains(t, result.Error(), "["+providerName+"]")
		})
	}
}

func TestThinkingBudget(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		effort   providers.ReasoningEffort
		expected int32
		ok       bool
	}{
		{
			name:     "minimal effort",
			effort:   providers.ReasoningEffortMinimal,
			expected: thinkingBudgetMinimal,
			ok:       true,
		},
		{
			name:     "low effort",
			effort:   providers.ReasoningEffortLow,
			expected: thinkingBudgetLow,
			ok:       true,
		},
		{
			name:     "medium effort",
			effort:   providers.ReasoningEffortMedium,
			expected: thinkingBudgetMedium,
			ok:       true,
		},
		{
			name:     "high effort",
			effort:   providers.ReasoningEffortHigh,
			expected: thinkingBudgetHigh,
			ok:       true,
		},
		{
			name:     "xhigh effort",
			effort:   providers.ReasoningEffortXHigh,
			expected: thinkingBudgetXHigh,
			ok:       true,
		},
		{
			name:     "max effort",
			effort:   providers.ReasoningEffortMax,
			expected: thinkingBudgetMax,
			ok:       true,
		},
		{
			name:     "none effort",
			effort:   providers.ReasoningEffortNone,
			expected: 0,
			ok:       false,
		},
		{
			name:     "invalid effort",
			effort:   "invalid",
			expected: 0,
			ok:       false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			budget, ok := thinkingBudget(tc.effort)
			require.Equal(t, tc.ok, ok)
			require.Equal(t, tc.expected, budget)
		})
	}
}

func TestApplyThinking(t *testing.T) {
	t.Parallel()

	t.Run("empty effort does nothing", func(t *testing.T) {
		t.Parallel()

		cfg := &genai.GenerateContentConfig{}
		require.NoError(t, applyThinking(cfg, "gemini-2.5-flash", ""))
		require.Nil(t, cfg.ThinkingConfig)
	})

	t.Run("none effort disables thoughts on Gemini 2.5", func(t *testing.T) {
		t.Parallel()

		cfg := &genai.GenerateContentConfig{}
		require.NoError(t, applyThinking(cfg, "gemini-2.5-flash", providers.ReasoningEffortNone))
		require.NotNil(t, cfg.ThinkingConfig)
		require.False(t, cfg.ThinkingConfig.IncludeThoughts)
		require.NotNil(t, cfg.ThinkingConfig.ThinkingBudget)
		require.Equal(t, int32(0), *cfg.ThinkingConfig.ThinkingBudget)

		wire, err := json.Marshal(cfg)
		require.NoError(t, err)
		require.JSONEq(t, `{"thinkingConfig":{"thinkingBudget":0}}`, string(wire))
	})

	t.Run("none effort is rejected when Gemini 3 cannot disable thinking", func(t *testing.T) {
		t.Parallel()

		cfg := &genai.GenerateContentConfig{}
		err := applyThinking(cfg, gemini3FlashPreviewModel, providers.ReasoningEffortNone)
		require.ErrorIs(t, err, errors.ErrUnsupportedParam)
		require.Nil(t, cfg.ThinkingConfig)
	})

	t.Run("none effort is rejected when Gemini 2.5 Pro cannot disable thinking", func(t *testing.T) {
		t.Parallel()

		cfg := &genai.GenerateContentConfig{}
		err := applyThinking(cfg, "gemini-2.5-pro", providers.ReasoningEffortNone)
		require.ErrorIs(t, err, errors.ErrUnsupportedParam)
		require.Nil(t, cfg.ThinkingConfig)
	})

	t.Run("none effort disables thinking on Gemini 2.5 preview Flash models", func(t *testing.T) {
		t.Parallel()

		for _, model := range []string{
			"gemini-2.5-flash-preview-05-20",
			"models/gemini-2.5-flash-lite-preview-06-17",
		} {
			cfg := &genai.GenerateContentConfig{}
			require.NoError(t, applyThinking(cfg, model, providers.ReasoningEffortNone))
			require.NotNil(t, cfg.ThinkingConfig)
			require.Equal(t, int32(0), *cfg.ThinkingConfig.ThinkingBudget)
		}
	})

	t.Run("auto effort does nothing", func(t *testing.T) {
		t.Parallel()

		cfg := &genai.GenerateContentConfig{}
		require.NoError(t, applyThinking(cfg, "gemini-2.5-flash", providers.ReasoningEffortAuto))
		require.Nil(t, cfg.ThinkingConfig)
	})

	t.Run("low effort sets thinking budget", func(t *testing.T) {
		t.Parallel()

		cfg := &genai.GenerateContentConfig{}
		require.NoError(t, applyThinking(cfg, "gemini-2.5-flash", providers.ReasoningEffortLow))
		require.NotNil(t, cfg.ThinkingConfig)
		require.True(t, cfg.ThinkingConfig.IncludeThoughts)
		require.Equal(t, thinkingBudgetLow, *cfg.ThinkingConfig.ThinkingBudget)
		require.Empty(t, cfg.ThinkingConfig.ThinkingLevel)
	})

	t.Run("minimal effort sets thinking budget", func(t *testing.T) {
		t.Parallel()

		cfg := &genai.GenerateContentConfig{}
		require.NoError(t, applyThinking(cfg, "gemini-2.5-flash", providers.ReasoningEffortMinimal))
		require.NotNil(t, cfg.ThinkingConfig)
		require.Equal(t, thinkingBudgetMinimal, *cfg.ThinkingConfig.ThinkingBudget)
	})

	t.Run("minimal effort respects Gemini 2.5 Flash-Lite lower bound", func(t *testing.T) {
		t.Parallel()

		for _, model := range []string{
			"gemini-2.5-flash-lite",
			"gemini-2.5-flash-lite-preview-06-17",
		} {
			cfg := &genai.GenerateContentConfig{}
			require.NoError(t, applyThinking(cfg, model, providers.ReasoningEffortMinimal))
			require.NotNil(t, cfg.ThinkingConfig)
			require.Equal(t, thinkingBudgetMinimalLite, *cfg.ThinkingConfig.ThinkingBudget)
		}
	})

	t.Run("high effort sets thinking config", func(t *testing.T) {
		t.Parallel()

		cfg := &genai.GenerateContentConfig{}
		require.NoError(t, applyThinking(cfg, "gemini-2.5-flash", providers.ReasoningEffortHigh))
		require.NotNil(t, cfg.ThinkingConfig)
		require.True(t, cfg.ThinkingConfig.IncludeThoughts)
		require.Equal(t, thinkingBudgetHigh, *cfg.ThinkingConfig.ThinkingBudget)
	})

	t.Run("caps maximum effort for Gemini 2.5 Flash models", func(t *testing.T) {
		t.Parallel()

		tests := []struct {
			model  string
			effort providers.ReasoningEffort
		}{
			{model: "gemini-2.5-flash", effort: providers.ReasoningEffortXHigh},
			{
				model:  "projects/p/locations/l/publishers/google/models/gemini-2.5-flash-lite",
				effort: providers.ReasoningEffortMax,
			},
		}

		for _, tc := range tests {
			cfg := &genai.GenerateContentConfig{}
			require.NoError(t, applyThinking(cfg, tc.model, tc.effort))
			require.NotNil(t, cfg.ThinkingConfig)
			require.Equal(t, thinkingBudgetHigh, *cfg.ThinkingConfig.ThinkingBudget)
		}
	})

	t.Run("keeps maximum effort for Gemini 2.5 Pro", func(t *testing.T) {
		t.Parallel()

		cfg := &genai.GenerateContentConfig{}
		require.NoError(t, applyThinking(cfg, "gemini-2.5-pro", providers.ReasoningEffortMax))
		require.NotNil(t, cfg.ThinkingConfig)
		require.Equal(t, thinkingBudgetMax, *cfg.ThinkingConfig.ThinkingBudget)
	})

	t.Run("gemini 3 uses thinking level", func(t *testing.T) {
		t.Parallel()

		cfg := &genai.GenerateContentConfig{}
		require.NoError(t, applyThinking(cfg, gemini3FlashPreviewModel, providers.ReasoningEffortMinimal))
		require.NotNil(t, cfg.ThinkingConfig)
		require.True(t, cfg.ThinkingConfig.IncludeThoughts)
		require.Equal(t, genai.ThinkingLevelMinimal, cfg.ThinkingConfig.ThinkingLevel)
		require.Nil(t, cfg.ThinkingConfig.ThinkingBudget)
	})

	t.Run("unknown effort returns error", func(t *testing.T) {
		t.Parallel()

		cfg := &genai.GenerateContentConfig{}
		err := applyThinking(cfg, "gemini-2.5-flash", "custom")
		require.Error(t, err)
		require.ErrorIs(t, err, errors.ErrUnsupportedParam)
	})
}

func TestSupportsDisabledThinkingBudget(t *testing.T) {
	t.Parallel()

	tests := []struct {
		model string
		want  bool
	}{
		{model: "gemini-2.5-flash", want: true},
		{model: "models/gemini-2.5-flash-lite", want: true},
		{model: "gemini-2.5-flash-preview-05-20", want: true},
		{model: "models/gemini-2.5-flash-lite-preview-06-17", want: true},
		{model: "gemini-2.5-pro", want: false},
		{model: "gemini-2.5-flash-future", want: false},
		{model: gemini3FlashPreviewModel, want: false},
	}

	for _, tc := range tests {
		t.Run(tc.model, func(t *testing.T) {
			t.Parallel()
			require.Equal(t, tc.want, supportsDisabledThinkingBudget(tc.model))
		})
	}
}

func TestUsesThinkingLevel(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		model string
		want  bool
	}{
		{name: "2.5 flash", model: "gemini-2.5-flash", want: false},
		{name: "3.0 flash", model: "gemini-3.0-flash", want: true},
		{name: "3 flash", model: gemini3FlashPreviewModel, want: true},
		{name: "3 pro preview", model: "gemini-3-pro-preview", want: true},
		{name: "3.5 flash", model: "gemini-3.5-flash", want: true},
		{name: "4 pro", model: "models/gemini-4-pro", want: true},
		{name: "unrelated", model: "gpt-4o", want: false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			require.Equal(t, tc.want, usesThinkingLevel(tc.model))
		})
	}
}

func TestConvertImagePart(t *testing.T) {
	t.Parallel()

	t.Run("converts base64 image", func(t *testing.T) {
		t.Parallel()

		img := &providers.ImageURL{URL: "data:image/jpeg;base64,/9j/4AAQSkZJRg=="}
		result, err := convertImagePart(img)
		require.NoError(t, err)
		require.NotNil(t, result)
		require.NotNil(t, result.InlineData)
		require.Equal(t, "image/jpeg", result.InlineData.MIMEType)
	})

	t.Run("converts URL image", func(t *testing.T) {
		t.Parallel()

		img := &providers.ImageURL{URL: "https://example.com/image.png"}
		result, err := convertImagePart(img)
		require.NoError(t, err)
		require.NotNil(t, result)
		require.NotNil(t, result.FileData)
		require.Equal(t, "https://example.com/image.png", result.FileData.FileURI)
	})
}

func TestConvertMessagesReturnsMalformedMediaError(t *testing.T) {
	t.Parallel()

	messages := []providers.Message{{
		Role: providers.RoleUser,
		Content: []providers.ContentPart{{
			Type:     contentPartTypeImageURL,
			ImageURL: &providers.ImageURL{URL: "data:image/png;base64,not-valid-base64"},
		}},
	}}

	contents, system, err := convertMessages(messages)
	require.Error(t, err)
	require.ErrorIs(t, err, errors.ErrInvalidRequest)
	require.Nil(t, contents)
	require.Nil(t, system)
}

func TestConvertEmbeddingInput(t *testing.T) {
	t.Parallel()

	t.Run("string input", func(t *testing.T) {
		t.Parallel()

		results, err := convertEmbeddingInputs("hello world")
		require.NoError(t, err)
		result := results[0]
		require.NotNil(t, result)
		require.Len(t, result.Parts, 1)
		require.Equal(t, "hello world", result.Parts[0].Text)
	})

	t.Run("string slice input", func(t *testing.T) {
		t.Parallel()

		results, err := convertEmbeddingInputs([]string{"hello", "world"})
		require.NoError(t, err)
		require.Len(t, results, 2)
		require.Equal(t, "hello", results[0].Parts[0].Text)
		require.Equal(t, "world", results[1].Parts[0].Text)
	})
}

func TestGenerateID(t *testing.T) {
	t.Parallel()

	t.Run("has correct prefix", func(t *testing.T) {
		t.Parallel()

		id, err := generateID(idPrefixCompletion)
		require.NoError(t, err)
		require.True(t, strings.HasPrefix(id, idPrefixCompletion))
	})

	t.Run("generates unique IDs", func(t *testing.T) {
		t.Parallel()

		id1, err := generateID(idPrefixToolCall)
		require.NoError(t, err)
		id2, err := generateID(idPrefixToolCall)
		require.NoError(t, err)
		require.NotEqual(t, id1, id2)
	})

	t.Run("has expected length", func(t *testing.T) {
		t.Parallel()

		id, err := generateID("test-")
		require.NoError(t, err)
		// prefix (5) + 24 hex chars (12 bytes) = 29.
		require.Len(t, id, 29)
	})
}

func TestNewStreamState(t *testing.T) {
	t.Parallel()

	state, err := newStreamState("gemini-1.5-flash")
	require.NoError(t, err)
	require.NotNil(t, state)
	require.Equal(t, "gemini-1.5-flash", state.model)
	require.True(t, strings.HasPrefix(state.messageID, idPrefixCompletion))
	require.Nil(t, state.toolCalls)
	require.Nil(t, state.usage)
}

func TestSetProviderExtra(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		initial  map[string]providers.ProviderData
		provider string
		key      string
		value    any
		expected map[string]providers.ProviderData
	}{
		{
			name:     "nil Extra initialises both maps",
			initial:  nil,
			provider: providerName,
			key:      "thought_signature",
			value:    "abc123",
			expected: map[string]providers.ProviderData{
				providerName: {"thought_signature": "abc123"},
			},
		},
		{
			name:     "nil provider map initialises inner map",
			initial:  map[string]providers.ProviderData{},
			provider: providerName,
			key:      "thought_signature",
			value:    "abc123",
			expected: map[string]providers.ProviderData{
				providerName: {"thought_signature": "abc123"},
			},
		},
		{
			name: "preserves existing provider keys",
			initial: map[string]providers.ProviderData{
				providerName: {"existing_key": "existing_value"},
			},
			provider: providerName,
			key:      "thought_signature",
			value:    "abc123",
			expected: map[string]providers.ProviderData{
				providerName: {
					"existing_key":      "existing_value",
					"thought_signature": "abc123",
				},
			},
		},
		{
			name: "preserves other providers",
			initial: map[string]providers.ProviderData{
				"other": {"key": "value"},
			},
			provider: providerName,
			key:      "thought_signature",
			value:    "abc123",
			expected: map[string]providers.ProviderData{
				"other":      {"key": "value"},
				providerName: {"thought_signature": "abc123"},
			},
		},
		{
			name: "overwrites existing key",
			initial: map[string]providers.ProviderData{
				providerName: {"thought_signature": "old"},
			},
			provider: providerName,
			key:      "thought_signature",
			value:    "new",
			expected: map[string]providers.ProviderData{
				providerName: {"thought_signature": "new"},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			toolCall := providers.ToolCall{Extra: tc.initial}
			setProviderExtra(&toolCall, tc.provider, tc.key, tc.value)
			require.Equal(t, tc.expected, toolCall.Extra)
		})
	}
}

func TestThoughtSignatureFromExtra(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		extra    map[string]providers.ProviderData
		expected []byte
	}{
		{
			name:     "nil extra returns nil",
			extra:    nil,
			expected: nil,
		},
		{
			name:     "missing provider returns nil",
			extra:    map[string]providers.ProviderData{"other": {"key": "value"}},
			expected: nil,
		},
		{
			name: "missing key returns nil",
			extra: map[string]providers.ProviderData{
				providerName: {"other_key": "value"},
			},
			expected: nil,
		},
		{
			name: "wrong type returns nil",
			extra: map[string]providers.ProviderData{
				providerName: {extraKeyThoughtSignature: 12345},
			},
			expected: nil,
		},
		{
			name: "invalid base64 returns nil",
			extra: map[string]providers.ProviderData{
				providerName: {extraKeyThoughtSignature: "not-valid-base64!!!"},
			},
			expected: nil,
		},
		{
			name: "valid signature decodes correctly",
			extra: map[string]providers.ProviderData{
				providerName: {
					extraKeyThoughtSignature: base64.StdEncoding.EncodeToString([]byte("test-sig")),
				},
			},
			expected: []byte("test-sig"),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result := thoughtSignatureFromExtra(tc.extra)
			require.Equal(t, tc.expected, result)
		})
	}
}

func TestThoughtSignatureRoundTrip(t *testing.T) {
	t.Parallel()

	// Simulate an API response with a ThoughtSignature on a function call.
	originalSig := []byte("opaque-signature-from-gemini-api-xyz123")
	originalID := "call_gemini_123"

	resp := &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{{
			Content: &genai.Content{
				Parts: []*genai.Part{{
					FunctionCall: &genai.FunctionCall{
						ID:   originalID,
						Name: "search",
						Args: map[string]any{"query": "test"},
					},
					ThoughtSignature: originalSig,
				}},
			},
			FinishReason: genai.FinishReasonStop,
		}},
	}

	// Capture via non-streaming path.
	result, err := convertResponse(resp, "gemini-2.5-pro")
	require.NoError(t, err)
	require.Len(t, result.Choices[0].Message.ToolCalls, 1)

	capturedTC := result.Choices[0].Message.ToolCalls[0]
	require.Equal(t, originalID, capturedTC.ID)
	require.NotNil(t, capturedTC.Extra)

	// Build a message with the captured tool call (as a caller would).
	assistantMsg := providers.Message{
		Role:      providers.RoleAssistant,
		Content:   "",
		ToolCalls: []providers.ToolCall{capturedTC},
	}

	// Replay via convertAssistantMessage.
	content, err := convertAssistantMessage(assistantMsg)
	require.NoError(t, err)
	require.NotNil(t, content)
	require.Len(t, content.Parts, 1)

	// Verify the signature round-tripped identically.
	require.Equal(t, originalSig, content.Parts[0].ThoughtSignature)
	require.Equal(t, originalID, content.Parts[0].FunctionCall.ID)
}

func TestThoughtSignatureWireFormat(t *testing.T) {
	t.Parallel()

	t.Run("request hook sends the documented literal bypass", func(t *testing.T) {
		t.Parallel()

		provider := &Provider{}
		contents, cfg, err := provider.convertParams(providers.CompletionParams{
			Model: gemini3FlashPreviewModel,
			Messages: []providers.Message{{Role: providers.RoleAssistant, ToolCalls: []providers.ToolCall{{
				ID: "call_1", Type: "function",
				Function: providers.FunctionCall{Name: "search", Arguments: `{"q":"test"}`},
			}}}},
		})
		require.NoError(t, err)
		require.True(t, hasThoughtSignatureBypass(contents))
		require.NotNil(t, cfg.HTTPOptions)
		require.NotNil(t, cfg.HTTPOptions.ExtrasRequestProvider)

		encoded := base64.StdEncoding.EncodeToString([]byte(thoughtSignatureBypass))
		body := map[string]any{"contents": []map[string]any{{
			"parts": []map[string]any{{"thoughtSignature": encoded}},
		}}}
		result := cfg.HTTPOptions.ExtrasRequestProvider(body)
		raw, err := json.Marshal(result)
		require.NoError(t, err)
		require.Contains(t, string(raw), `"thoughtSignature":"`+thoughtSignatureBypass+`"`)
		require.NotContains(t, string(raw), encoded)
	})

	t.Run("real signature is base64-encoded by json.Marshal", func(t *testing.T) {
		t.Parallel()

		realSig := []byte("opaque-gemini-signature-abc123")
		storedB64 := base64.StdEncoding.EncodeToString(realSig)

		msg := providers.Message{
			Role: providers.RoleAssistant,
			ToolCalls: []providers.ToolCall{{
				ID:   "call_1",
				Type: "function",
				Function: providers.FunctionCall{
					Name:      "search",
					Arguments: `{"q":"test"}`,
				},
				Extra: map[string]providers.ProviderData{
					providerName: {extraKeyThoughtSignature: storedB64},
				},
			}},
		}

		content, err := convertAssistantMessage(msg)
		require.NoError(t, err)
		require.Len(t, content.Parts, 1)
		require.Equal(t, realSig, content.Parts[0].ThoughtSignature)

		raw, err := json.Marshal(content.Parts[0])
		require.NoError(t, err)
		require.Contains(t, string(raw), storedB64)
	})
}

func TestThoughtSignatureBypassWireFormat(t *testing.T) {
	t.Parallel()

	var requestBody []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if !assert.NoError(t, err) {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		requestBody = body
		w.Header().Set("Content-Type", "application/json")
		_, err = w.Write([]byte(`{
			"candidates":[{"content":{"parts":[{"text":"ok"}],"role":"model"},"finishReason":"STOP"}],
			"modelVersion":"gemini-3-flash-preview",
			"responseId":"response-1"
		}`))
		assert.NoError(t, err)
	}))
	defer server.Close()

	provider, err := New(config.WithAPIKey("test-key"), config.WithBaseURL(server.URL))
	require.NoError(t, err)
	_, err = provider.Completion(t.Context(), providers.CompletionParams{
		Model: gemini3FlashPreviewModel,
		Messages: []providers.Message{{Role: providers.RoleAssistant, ToolCalls: []providers.ToolCall{{
			ID: "call_1", Type: "function",
			Function: providers.FunctionCall{Name: "search", Arguments: `{"q":"test"}`},
		}}}},
	})
	require.NoError(t, err)
	require.Contains(t, string(requestBody), `"thoughtSignature":"`+thoughtSignatureBypass+`"`)
	require.NotContains(t, string(requestBody), base64.StdEncoding.EncodeToString([]byte(thoughtSignatureBypass)))
}

func TestStreamStateProcessResponse(t *testing.T) {
	t.Parallel()

	t.Run("processes text content", func(t *testing.T) {
		t.Parallel()

		state, err := newStreamState("test-model")
		require.NoError(t, err)
		resp := &genai.GenerateContentResponse{
			Candidates: []*genai.Candidate{{
				Content: &genai.Content{
					Parts: []*genai.Part{{Text: "Hello "}},
				},
			}},
		}

		chunks, err := state.processResponse(resp)
		require.NoError(t, err)
		require.Len(t, chunks, 1)
		require.Equal(t, "Hello ", chunks[0].Choices[0].Delta.Content)
		require.Equal(t, "Hello ", state.content.String())
	})

	t.Run("processes thinking content", func(t *testing.T) {
		t.Parallel()

		state, err := newStreamState("test-model")
		require.NoError(t, err)
		resp := &genai.GenerateContentResponse{
			Candidates: []*genai.Candidate{{
				Content: &genai.Content{
					Parts: []*genai.Part{{Text: "Let me think...", Thought: true}},
				},
			}},
		}

		chunks, err := state.processResponse(resp)
		require.NoError(t, err)
		require.Len(t, chunks, 1)
		require.NotNil(t, chunks[0].Choices[0].Delta.Reasoning)
		require.Equal(t, "Let me think...", chunks[0].Choices[0].Delta.Reasoning.Content)
	})

	t.Run("processes function call", func(t *testing.T) {
		t.Parallel()

		state, err := newStreamState("test-model")
		require.NoError(t, err)
		resp := &genai.GenerateContentResponse{
			Candidates: []*genai.Candidate{{
				Content: &genai.Content{
					Parts: []*genai.Part{{
						FunctionCall: &genai.FunctionCall{
							ID:   "call_gemini_123",
							Name: "get_weather",
							Args: map[string]any{"location": "Paris"},
						},
					}},
				},
			}},
		}

		chunks, err := state.processResponse(resp)
		require.NoError(t, err)
		require.Len(t, chunks, 1)
		require.Len(t, chunks[0].Choices[0].Delta.ToolCalls, 1)
		require.Equal(t, "call_gemini_123", chunks[0].Choices[0].Delta.ToolCalls[0].ID)
		require.Equal(t, "get_weather", chunks[0].Choices[0].Delta.ToolCalls[0].Function.Name)
		require.Contains(t, chunks[0].Choices[0].Delta.ToolCalls[0].Function.Arguments, "Paris")
	})

	t.Run("tracks usage metadata", func(t *testing.T) {
		t.Parallel()

		state, err := newStreamState("test-model")
		require.NoError(t, err)
		resp := &genai.GenerateContentResponse{
			UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
				PromptTokenCount:     10,
				CandidatesTokenCount: 5,
			},
			Candidates: []*genai.Candidate{{
				Content: &genai.Content{
					Parts: []*genai.Part{{Text: "Hi"}},
				},
			}},
		}

		_, err = state.processResponse(resp)
		require.NoError(t, err)
		require.NotNil(t, state.usage)
		require.Equal(t, 10, state.usage.PromptTokens)
		require.Equal(t, 5, state.usage.CompletionTokens)
		require.Equal(t, 15, state.usage.TotalTokens)
	})

	t.Run("captures thought signature on function call", func(t *testing.T) {
		t.Parallel()

		state, err := newStreamState("test-model")
		require.NoError(t, err)
		resp := &genai.GenerateContentResponse{
			Candidates: []*genai.Candidate{{
				Content: &genai.Content{
					Parts: []*genai.Part{{
						FunctionCall: &genai.FunctionCall{
							Name: "get_weather",
							Args: map[string]any{"location": "Paris"},
						},
						ThoughtSignature: []byte("test-signature-bytes"),
					}},
				},
			}},
		}

		chunks, err := state.processResponse(resp)
		require.NoError(t, err)
		require.Len(t, chunks, 1)

		tc := chunks[0].Choices[0].Delta.ToolCalls[0]
		require.NotNil(t, tc.Extra)
		geminiData, ok := tc.Extra[providerName]
		require.True(t, ok, "expected google provider data in Extra")

		sig, ok := geminiData[extraKeyThoughtSignature].(string)
		require.True(t, ok, "expected thought_signature to be a string")

		// Value should be base64-encoded.
		decoded, err := base64.StdEncoding.DecodeString(sig)
		require.NoError(t, err)
		require.Equal(t, []byte("test-signature-bytes"), decoded)
	})

	t.Run("no thought signature leaves Extra nil", func(t *testing.T) {
		t.Parallel()

		state, err := newStreamState("test-model")
		require.NoError(t, err)
		resp := &genai.GenerateContentResponse{
			Candidates: []*genai.Candidate{{
				Content: &genai.Content{
					Parts: []*genai.Part{{
						FunctionCall: &genai.FunctionCall{
							Name: "get_weather",
							Args: map[string]any{"location": "Paris"},
						},
					}},
				},
			}},
		}

		chunks, err := state.processResponse(resp)
		require.NoError(t, err)
		require.Len(t, chunks, 1)

		tc := chunks[0].Choices[0].Delta.ToolCalls[0]
		require.Nil(t, tc.Extra)
	})

	t.Run("does not emit metadata-only chunks without candidates", func(t *testing.T) {
		t.Parallel()

		state, err := newStreamState("test-model")
		require.NoError(t, err)
		resp := &genai.GenerateContentResponse{}

		chunks, err := state.processResponse(resp)
		require.NoError(t, err)
		require.Empty(t, chunks)
		final := state.finalChunk()
		require.Len(t, final.Choices[0].Delta.Extra[providerName][extraKeyResponseEvents], 1)
	})
}

func TestStreamStateFinalChunk(t *testing.T) {
	t.Parallel()

	t.Run("defaults to stop finish reason", func(t *testing.T) {
		t.Parallel()

		state, err := newStreamState("test-model")
		require.NoError(t, err)
		state.finishReason = genai.FinishReasonStop

		chunk := state.finalChunk()
		require.Equal(t, providers.FinishReasonStop, chunk.Choices[0].FinishReason)
	})

	t.Run("uses tool_calls when tool calls present", func(t *testing.T) {
		t.Parallel()

		state, err := newStreamState("test-model")
		require.NoError(t, err)
		state.finishReason = genai.FinishReasonStop
		state.toolCalls = []providers.ToolCall{
			{ID: "call_1", Type: "function", Function: providers.FunctionCall{Name: "get_weather"}},
		}

		chunk := state.finalChunk()
		require.Equal(t, providers.FinishReasonToolCalls, chunk.Choices[0].FinishReason)
	})

	t.Run("uses max_tokens finish reason", func(t *testing.T) {
		t.Parallel()

		state, err := newStreamState("test-model")
		require.NoError(t, err)
		state.finishReason = genai.FinishReasonMaxTokens

		chunk := state.finalChunk()
		require.Equal(t, providers.FinishReasonLength, chunk.Choices[0].FinishReason)
	})

	t.Run("includes usage", func(t *testing.T) {
		t.Parallel()

		state, err := newStreamState("test-model")
		require.NoError(t, err)
		state.usage = &providers.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15}

		chunk := state.finalChunk()
		require.NotNil(t, chunk.Usage)
		require.Equal(t, 15, chunk.Usage.TotalTokens)
	})
}

func TestConvertParams(t *testing.T) {
	t.Parallel()

	provider, err := New(config.WithAPIKey("test-key"))
	require.NoError(t, err)

	t.Run("json_object sets mime type", func(t *testing.T) {
		t.Parallel()

		_, cfg, err := provider.convertParams(providers.CompletionParams{
			Model:    "gemini-2.0-flash",
			Messages: []providers.Message{{Role: providers.RoleUser, Content: "hi"}},
			ResponseFormat: &providers.ResponseFormat{
				Type: responseFormatJSON,
			},
		})
		require.NoError(t, err)

		require.Equal(t, responseMIMETypeJSON, cfg.ResponseMIMEType)
		require.Nil(t, cfg.ResponseJsonSchema)
	})

	t.Run("json_schema sets mime type and schema", func(t *testing.T) {
		t.Parallel()

		schema := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"name":       map[string]any{"type": "string"},
				"population": map[string]any{"type": "integer"},
			},
			"required": []string{"name", "population"},
		}

		_, cfg, err := provider.convertParams(providers.CompletionParams{
			Model:    "gemini-2.0-flash",
			Messages: []providers.Message{{Role: providers.RoleUser, Content: "hi"}},
			ResponseFormat: &providers.ResponseFormat{
				Type: responseFormatJSONSchema,
				JSONSchema: &providers.JSONSchema{
					Name:   "city_info",
					Schema: schema,
				},
			},
		})
		require.NoError(t, err)

		require.Equal(t, responseMIMETypeJSON, cfg.ResponseMIMEType)
		require.Equal(t, schema, cfg.ResponseJsonSchema)
	})

	t.Run("nil response format leaves config unchanged", func(t *testing.T) {
		t.Parallel()

		_, cfg, err := provider.convertParams(providers.CompletionParams{
			Model:    "gemini-2.0-flash",
			Messages: []providers.Message{{Role: providers.RoleUser, Content: "hi"}},
		})
		require.NoError(t, err)

		require.Empty(t, cfg.ResponseMIMEType)
		require.Nil(t, cfg.ResponseJsonSchema)
	})

	t.Run("unknown reasoning effort returns error", func(t *testing.T) {
		t.Parallel()

		_, _, err := provider.convertParams(providers.CompletionParams{
			Model:           "gemini-2.5-flash",
			Messages:        []providers.Message{{Role: providers.RoleUser, Content: "hi"}},
			ReasoningEffort: "custom",
		})
		require.Error(t, err)
		require.ErrorIs(t, err, errors.ErrUnsupportedParam)
	})
}

func TestConvertResponse(t *testing.T) {
	t.Parallel()

	t.Run("converts text response", func(t *testing.T) {
		t.Parallel()

		resp := &genai.GenerateContentResponse{
			Candidates: []*genai.Candidate{{
				Content: &genai.Content{
					Parts: []*genai.Part{{Text: "Hello World"}},
				},
				FinishReason: genai.FinishReasonStop,
			}},
			UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
				PromptTokenCount:     10,
				CandidatesTokenCount: 5,
			},
		}

		result, err := convertResponse(resp, "gemini-1.5-flash")
		require.NoError(t, err)
		require.Equal(t, objectChatCompletion, result.Object)
		require.Equal(t, "gemini-1.5-flash", result.Model)
		require.Len(t, result.Choices, 1)
		require.Equal(t, "Hello World", result.Choices[0].Message.ContentString())
		require.Equal(t, providers.RoleAssistant, result.Choices[0].Message.Role)
		require.Equal(t, providers.FinishReasonStop, result.Choices[0].FinishReason)
		require.NotNil(t, result.Usage)
		require.Equal(t, 10, result.Usage.PromptTokens)
		require.Equal(t, 5, result.Usage.CompletionTokens)
		require.Equal(t, 15, result.Usage.TotalTokens)
	})

	t.Run("converts function call response", func(t *testing.T) {
		t.Parallel()

		resp := &genai.GenerateContentResponse{
			Candidates: []*genai.Candidate{{
				Content: &genai.Content{
					Parts: []*genai.Part{{
						FunctionCall: &genai.FunctionCall{
							Name: "get_weather",
							Args: map[string]any{"location": "Paris"},
						},
					}},
				},
				FinishReason: genai.FinishReasonStop,
			}},
		}

		result, err := convertResponse(resp, "gemini-1.5-flash")
		require.NoError(t, err)
		require.Len(t, result.Choices[0].Message.ToolCalls, 1)
		require.Equal(t, "get_weather", result.Choices[0].Message.ToolCalls[0].Function.Name)
		require.Equal(t, providers.FinishReasonToolCalls, result.Choices[0].FinishReason)
	})

	t.Run("captures thought signature on function call", func(t *testing.T) {
		t.Parallel()

		resp := &genai.GenerateContentResponse{
			Candidates: []*genai.Candidate{{
				Content: &genai.Content{
					Parts: []*genai.Part{{
						FunctionCall: &genai.FunctionCall{
							Name: "get_weather",
							Args: map[string]any{"location": "Paris"},
						},
						ThoughtSignature: []byte("non-stream-sig"),
					}},
				},
				FinishReason: genai.FinishReasonStop,
			}},
		}

		result, err := convertResponse(resp, "gemini-2.5-pro")
		require.NoError(t, err)
		require.Len(t, result.Choices[0].Message.ToolCalls, 1)

		tc := result.Choices[0].Message.ToolCalls[0]
		require.NotNil(t, tc.Extra)
		geminiData := tc.Extra[providerName]
		sig, ok := geminiData[extraKeyThoughtSignature].(string)
		require.True(t, ok)

		decoded, err := base64.StdEncoding.DecodeString(sig)
		require.NoError(t, err)
		require.Equal(t, []byte("non-stream-sig"), decoded)
	})

	t.Run("converts thinking response", func(t *testing.T) {
		t.Parallel()

		resp := &genai.GenerateContentResponse{
			Candidates: []*genai.Candidate{{
				Content: &genai.Content{
					Parts: []*genai.Part{
						{Text: "Let me think...", Thought: true},
						{Text: "Hello!"},
					},
				},
				FinishReason: genai.FinishReasonStop,
			}},
		}

		result, err := convertResponse(resp, "gemini-2.0-flash")
		require.NoError(t, err)
		require.Equal(t, "Hello!", result.Choices[0].Message.ContentString())
		require.NotNil(t, result.Choices[0].Message.Reasoning)
		require.Equal(t, "Let me think...", result.Choices[0].Message.Reasoning.Content)
	})
}

func TestApplyResponseFormat(t *testing.T) {
	t.Parallel()

	t.Run("json_object sets mime type", func(t *testing.T) {
		t.Parallel()

		cfg := &genai.GenerateContentConfig{}
		err := applyResponseFormat(cfg, &providers.ResponseFormat{Type: "json_object"})
		require.NoError(t, err)
		require.Equal(t, "application/json", cfg.ResponseMIMEType)
	})

	t.Run("unsupported text returns error", func(t *testing.T) {
		t.Parallel()

		cfg := &genai.GenerateContentConfig{}
		err := applyResponseFormat(cfg, &providers.ResponseFormat{Type: "text"})
		require.Error(t, err)
		require.Empty(t, cfg.ResponseMIMEType)
	})

	t.Run("json_schema sets mime type and schema", func(t *testing.T) {
		t.Parallel()

		schema := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"name": map[string]any{"type": "string"},
			},
		}
		cfg := &genai.GenerateContentConfig{}
		err := applyResponseFormat(cfg, &providers.ResponseFormat{
			Type: "json_schema",
			JSONSchema: &providers.JSONSchema{
				Name:   "test_schema",
				Schema: schema,
			},
		})
		require.NoError(t, err)
		require.Equal(t, "application/json", cfg.ResponseMIMEType)
		require.Equal(t, schema, cfg.ResponseJsonSchema)
	})

	t.Run("json_schema with nil JSONSchema does not set mime type", func(t *testing.T) {
		t.Parallel()

		cfg := &genai.GenerateContentConfig{}
		err := applyResponseFormat(cfg, &providers.ResponseFormat{Type: "json_schema"})
		require.Error(t, err)
		require.Empty(t, cfg.ResponseMIMEType)
		require.Nil(t, cfg.ResponseJsonSchema)
	})

	t.Run("json_schema with nil Schema does not set mime type", func(t *testing.T) {
		t.Parallel()

		cfg := &genai.GenerateContentConfig{}
		err := applyResponseFormat(cfg, &providers.ResponseFormat{
			Type:       "json_schema",
			JSONSchema: &providers.JSONSchema{Name: "test"},
		})
		require.Error(t, err)
		require.Empty(t, cfg.ResponseMIMEType)
		require.Nil(t, cfg.ResponseJsonSchema)
	})
}

func TestTextThoughtSignatureRoundTrip(t *testing.T) {
	t.Parallel()

	response := &genai.GenerateContentResponse{Candidates: []*genai.Candidate{{
		Content: &genai.Content{Parts: []*genai.Part{{Text: "answer", ThoughtSignature: []byte("text-signature")}}},
	}}}
	completion, err := convertResponse(response, "gemini-2.5-pro")
	require.NoError(t, err)

	encoded, err := json.Marshal(completion.Choices[0].Message)
	require.NoError(t, err)
	var restored providers.Message
	require.NoError(t, json.Unmarshal(encoded, &restored))
	converted, err := convertAssistantMessage(restored)
	require.NoError(t, err)
	require.Equal(t, []byte("text-signature"), converted.Parts[0].ThoughtSignature)
}

func TestEmptySignedPartsRoundTrip(t *testing.T) {
	t.Parallel()

	response := &genai.GenerateContentResponse{Candidates: []*genai.Candidate{{
		Content: &genai.Content{Parts: []*genai.Part{
			{Text: "", ThoughtSignature: []byte("text-signature")},
			{Text: "", Thought: true, ThoughtSignature: []byte("thought-signature")},
		}},
	}}}
	completion, err := convertResponse(response, "gemini-2.5-pro")
	require.NoError(t, err)
	require.NotNil(t, completion.Choices[0].Message.Reasoning)
	require.Equal(t, "", completion.Choices[0].Message.Reasoning.Content)

	encoded, err := json.Marshal(completion.Choices[0].Message)
	require.NoError(t, err)
	var restored providers.Message
	require.NoError(t, json.Unmarshal(encoded, &restored))
	replayed, err := convertAssistantMessage(restored)
	require.NoError(t, err)
	require.Len(t, replayed.Parts, 2)
	require.Equal(t, []byte("text-signature"), replayed.Parts[0].ThoughtSignature)
	require.False(t, replayed.Parts[0].Thought)
	require.Equal(t, []byte("thought-signature"), replayed.Parts[1].ThoughtSignature)
	require.True(t, replayed.Parts[1].Thought)
}

func TestResponsePreservesExactGeminiMetadataBody(t *testing.T) {
	t.Parallel()

	const body = `{"candidates":[{"content":{"parts":[{"text":"answer"}]},` +
		`"citationMetadata":{"citations":[]},"tokenCount":0,` +
		`"safetyRatings":[{"blocked":false,"probabilityScore":0}]}],` +
		`"usageMetadata":{"promptTokenCount":0,"totalTokenCount":0}}`
	response := &genai.GenerateContentResponse{
		SDKHTTPResponse: &genai.HTTPResponse{Body: body},
		Candidates: []*genai.Candidate{{
			Content: &genai.Content{Parts: []*genai.Part{{Text: "answer"}}},
		}},
	}

	completion, err := convertResponse(response, "gemini-2.5-pro")
	require.NoError(t, err)
	raw, ok := completion.Choices[0].Message.Extra[providerName][extraKeyResponse].(json.RawMessage)
	require.True(t, ok)
	require.JSONEq(t, body, string(raw))

	encoded, err := json.Marshal(completion.Choices[0].Message)
	require.NoError(t, err)
	var restored providers.Message
	require.NoError(t, json.Unmarshal(encoded, &restored))
	metadata, ok := restored.Extra[providerName][extraKeyResponse].(map[string]any)
	require.True(t, ok)
	candidates, ok := metadata["candidates"].([]any)
	require.True(t, ok)
	candidate, ok := candidates[0].(map[string]any)
	require.True(t, ok)
	require.Equal(t, float64(0), candidate["tokenCount"])
	ratings, ok := candidate["safetyRatings"].([]any)
	require.True(t, ok)
	rating, ok := ratings[0].(map[string]any)
	require.True(t, ok)
	blocked, ok := rating["blocked"].(bool)
	require.True(t, ok)
	require.False(t, blocked)
}

func TestResponsePreservesExplicitZeroTotalTokenCount(t *testing.T) {
	t.Parallel()

	response := &genai.GenerateContentResponse{
		SDKHTTPResponse: &genai.HTTPResponse{
			Body: `{"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":3,"totalTokenCount":0}}`,
		},
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     2,
			CandidatesTokenCount: 3,
			TotalTokenCount:      0,
		},
	}

	completion, err := convertResponse(response, "gemini-2.5-pro")
	require.NoError(t, err)
	require.Zero(t, completion.Usage.TotalTokens)
}

func TestCodeExecutionPartsRoundTrip(t *testing.T) {
	t.Parallel()

	response := &genai.GenerateContentResponse{Candidates: []*genai.Candidate{{
		Content: &genai.Content{Parts: []*genai.Part{
			{Text: "I will calculate it."},
			{ExecutableCode: &genai.ExecutableCode{Code: "print(6 * 7)", Language: genai.LanguagePython, ID: "code-1"}},
			{CodeExecutionResult: &genai.CodeExecutionResult{Outcome: genai.OutcomeOK, Output: "42\n", ID: "code-1"}},
			{Text: "The result is 42."},
		}},
	}}}
	completion, err := convertResponse(response, "gemini-2.5-pro")
	require.NoError(t, err)
	require.Equal(t, "I will calculate it.The result is 42.", completion.Choices[0].Message.Content)

	encoded, err := json.Marshal(completion.Choices[0].Message)
	require.NoError(t, err)
	var restored providers.Message
	require.NoError(t, json.Unmarshal(encoded, &restored))
	replayed, err := convertAssistantMessage(restored)
	require.NoError(t, err)
	require.Len(t, replayed.Parts, 4)
	require.Equal(t, "print(6 * 7)", replayed.Parts[1].ExecutableCode.Code)
	require.Equal(t, genai.LanguagePython, replayed.Parts[1].ExecutableCode.Language)
	require.Equal(t, "42\n", replayed.Parts[2].CodeExecutionResult.Output)
	require.Equal(t, genai.OutcomeOK, replayed.Parts[2].CodeExecutionResult.Outcome)
}

func TestResponsePartsReplayRejectsConflictingNormalizedContent(t *testing.T) {
	t.Parallel()

	response := &genai.GenerateContentResponse{Candidates: []*genai.Candidate{{
		Content: &genai.Content{Parts: []*genai.Part{
			{ExecutableCode: &genai.ExecutableCode{Language: genai.LanguagePython, Code: "print('ok')"}},
			{Text: "original"},
		}},
	}}}
	completion, err := convertResponse(response, "gemini-2.5-pro")
	require.NoError(t, err)
	message := completion.Choices[0].Message
	message.Content = "edited"

	_, err = convertAssistantMessage(message)
	require.ErrorIs(t, err, errors.ErrInvalidRequest)
}

func TestStreamCodeExecutionPartsAppearInDeltasAndFinalReplayMetadata(t *testing.T) {
	t.Parallel()

	state, err := newStreamState("gemini-2.5-pro")
	require.NoError(t, err)
	textChunks, err := state.processResponse(&genai.GenerateContentResponse{Candidates: []*genai.Candidate{{
		Content: &genai.Content{Parts: []*genai.Part{{Text: "Calculating."}}},
	}}})
	require.NoError(t, err)
	require.Len(t, textChunks, 1)
	response := &genai.GenerateContentResponse{Candidates: []*genai.Candidate{{
		Content: &genai.Content{Parts: []*genai.Part{
			{ExecutableCode: &genai.ExecutableCode{Code: "print(42)", Language: genai.LanguagePython}},
			{CodeExecutionResult: &genai.CodeExecutionResult{Outcome: genai.OutcomeOK, Output: "42\n"}},
		}},
	}}}

	chunks, err := state.processResponse(response)
	require.NoError(t, err)
	require.Len(t, chunks, 2)
	for _, chunk := range chunks {
		require.NotNil(t, chunk.Choices[0].Delta.Extra[providerName][extraKeyResponseParts])
	}

	final := state.finalChunk()
	replayed, ok, err := responsePartsFromExtra(final.Choices[0].Delta.Extra)
	require.NoError(t, err)
	require.True(t, ok)
	require.Len(t, replayed, 3)
	require.Equal(t, "Calculating.", replayed[0].Text)
	require.Equal(t, "print(42)", replayed[1].ExecutableCode.Code)
	require.Equal(t, "42\n", replayed[2].CodeExecutionResult.Output)
}

func TestStreamTextThoughtSignatureIsImmutableAndRoundTrips(t *testing.T) {
	t.Parallel()

	state, err := newStreamState("gemini-2.5-pro")
	require.NoError(t, err)
	signature := []byte("stream-signature")
	response := &genai.GenerateContentResponse{Candidates: []*genai.Candidate{{
		Content: &genai.Content{Parts: []*genai.Part{{Text: "answer", ThoughtSignature: signature}}},
	}}}
	chunks, err := state.processResponse(response)
	require.NoError(t, err)
	require.Len(t, chunks, 1)
	signature[0] = 'X'

	encoded, err := json.Marshal(chunks[0].Choices[0].Delta)
	require.NoError(t, err)
	var restored providers.ChunkDelta
	require.NoError(t, json.Unmarshal(encoded, &restored))
	value := restored.Extra[providerName][extraKeyThoughtSignature]
	encodedSignature, ok := value.(string)
	require.True(t, ok)
	decoded, err := base64.StdEncoding.DecodeString(encodedSignature)
	require.NoError(t, err)
	require.Equal(t, []byte("stream-signature"), decoded)
}

func TestStreamPreservesResponseEventsAndEmptySignedParts(t *testing.T) {
	t.Parallel()

	state, err := newStreamState("gemini-2.5-pro")
	require.NoError(t, err)
	const body = `{"candidates":[{"content":{"parts":[{"text":"","thoughtSignature":"c2ln"}]},` +
		`"tokenCount":0}]}`
	response := &genai.GenerateContentResponse{
		SDKHTTPResponse: &genai.HTTPResponse{Body: body},
		Candidates: []*genai.Candidate{{
			Content: &genai.Content{Parts: []*genai.Part{{Text: "", ThoughtSignature: []byte("sig")}}},
		}},
	}

	chunks, err := state.processResponse(response)
	require.NoError(t, err)
	require.Len(t, chunks, 1)

	final := state.finalChunk()
	events, ok := final.Choices[0].Delta.Extra[providerName][extraKeyResponseEvents].([]json.RawMessage)
	require.True(t, ok)
	require.Len(t, events, 1)
	require.JSONEq(t, body, string(events[0]))
	replayed, ok, err := responsePartsFromExtra(final.Choices[0].Delta.Extra)
	require.NoError(t, err)
	require.True(t, ok)
	require.Equal(t, []byte("sig"), replayed[0].ThoughtSignature)
}

func TestSendStreamChunk(t *testing.T) {
	t.Parallel()

	chunks := make(chan providers.ChatCompletionChunk, 1)
	errs := make(chan error, 1)
	want := providers.ChatCompletionChunk{ID: "chunk-1"}

	require.True(t, sendStreamChunk(t.Context(), chunks, errs, want))
	require.Equal(t, want, <-chunks)
	require.Empty(t, errs)
}

func TestSendStreamChunkReportsCancellation(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	chunks := make(chan providers.ChatCompletionChunk)
	errs := make(chan error, 1)

	require.False(t, sendStreamChunk(ctx, chunks, errs, providers.ChatCompletionChunk{}))
	require.ErrorIs(t, <-errs, context.Canceled)
	require.Empty(t, chunks)
}

// Integration tests - only run if API key is available.

func TestIntegrationCompletion(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey(providerName) {
		t.Skip("GEMINI_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:    testutil.TestModel(providerName),
		Messages: testutil.SimpleMessages(),
	}

	resp, err := provider.Completion(ctx, params)
	require.NoError(t, err)

	require.NotEmpty(t, resp.ID)
	require.Equal(t, objectChatCompletion, resp.Object)
	require.Len(t, resp.Choices, 1)
	require.NotEmpty(t, resp.Choices[0].Message.Content)
	require.Equal(t, providers.RoleAssistant, resp.Choices[0].Message.Role)
	require.NotNil(t, resp.Usage)
	require.Greater(t, resp.Usage.TotalTokens, 0)
}

func TestIntegrationCompletionWithSystemMessage(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey(providerName) {
		t.Skip("GEMINI_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:    testutil.TestModel(providerName),
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

	if testutil.SkipIfNoAPIKey(providerName) {
		t.Skip("GEMINI_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:    testutil.TestModel(providerName),
		Messages: testutil.SimpleMessages(),
		Stream:   true,
	}

	chunks, errs := provider.CompletionStream(ctx, params)

	var content strings.Builder
	chunkCount := 0

	for chunk := range chunks {
		chunkCount++
		require.Equal(t, objectChatCompletionChunk, chunk.Object)
		if len(chunk.Choices) > 0 {
			content.WriteString(chunk.Choices[0].Delta.Content)
		}
	}

	err = <-errs
	require.NoError(t, err)

	require.Greater(t, chunkCount, 0)
	require.NotEmpty(t, content.String())
}

func TestIntegrationCompletionConversation(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey(providerName) {
		t.Skip("GEMINI_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:    testutil.TestModel(providerName),
		Messages: testutil.ConversationMessages(),
	}

	resp, err := provider.Completion(ctx, params)
	require.NoError(t, err)

	require.NotEmpty(t, resp.ID)
	require.Len(t, resp.Choices, 1)

	contentStr, ok := resp.Choices[0].Message.Content.(string)
	require.True(t, ok, "expected string content")
	require.Contains(t, strings.ToLower(contentStr), "alice")
}

func TestIntegrationEmbedding(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey(providerName) {
		t.Skip("GEMINI_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.EmbeddingParams{
		Model: testutil.EmbeddingModel(providerName),
		Input: "Hello world",
	}

	resp, err := provider.Embedding(ctx, params)
	require.NoError(t, err)

	require.Equal(t, objectList, resp.Object)
	require.NotEmpty(t, resp.Data)
	require.NotEmpty(t, resp.Data[0].Embedding)
	require.Equal(t, objectEmbedding, resp.Data[0].Object)
}

func TestIntegrationListModels(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey(providerName) {
		t.Skip("GEMINI_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	resp, err := provider.ListModels(ctx)
	require.NoError(t, err)

	require.Equal(t, objectList, resp.Object)
	require.NotEmpty(t, resp.Data)

	// Verify model structure.
	for _, model := range resp.Data {
		require.NotEmpty(t, model.ID)
		require.Equal(t, objectModel, model.Object)
		require.Equal(t, "google", model.OwnedBy)
	}
}
