package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	oaisdk "github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

func TestConvertParamsPreservesExplicitEmptyLogitBias(t *testing.T) {
	t.Parallel()

	params := providers.CompletionParams{Model: "gpt-5.6", LogitBias: map[string]int{}}
	wire, err := json.Marshal(convertParams(params, providerName))
	require.NoError(t, err)

	var body map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(wire, &body))
	require.JSONEq(t, `{}`, string(body["logit_bias"]))

	params.LogitBias = nil
	wire, err = json.Marshal(convertParams(params, providerName))
	require.NoError(t, err)
	body = nil
	require.NoError(t, json.Unmarshal(wire, &body))
	require.NotContains(t, body, "logit_bias")
}

func TestConvertAssistantMessageOmitsAbsentToolCalls(t *testing.T) {
	t.Parallel()

	message, err := convertAssistantMessage(providers.Message{
		Role:    providers.RoleAssistant,
		Content: "plain response",
	}, providerName)
	require.NoError(t, err)

	wire, err := json.Marshal(message)
	require.NoError(t, err)
	var body map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(wire, &body))
	require.NotContains(t, body, "tool_calls")
}

func TestConvertAssistantMessagePreservesTextPartsBesideToolCalls(t *testing.T) {
	t.Parallel()

	message, err := convertAssistantMessage(providers.Message{
		Role: providers.RoleAssistant,
		Content: []providers.ContentPart{
			{Type: contentTypeText, Text: "before"},
			{Type: contentTypeText, Text: "after"},
		},
		ToolCalls: []providers.ToolCall{{
			ID:   "call-1",
			Type: "function",
			Function: providers.FunctionCall{
				Name:      "lookup",
				Arguments: "{}",
			},
		}},
	}, providerName)
	require.NoError(t, err)

	wire, err := json.Marshal(message)
	require.NoError(t, err)
	var body struct {
		Content []struct {
			Text string `json:"text"`
			Type string `json:"type"`
		} `json:"content"`
		ToolCalls []json.RawMessage `json:"tool_calls"`
	}
	require.NoError(t, json.Unmarshal(wire, &body))
	require.Equal(t, "before", body.Content[0].Text)
	require.Equal(t, "text", body.Content[0].Type)
	require.Equal(t, "after", body.Content[1].Text)
	require.Len(t, body.ToolCalls, 1)
}

func TestConvertAssistantMessageRejectsUnsupportedContentParts(t *testing.T) {
	t.Parallel()

	_, err := convertAssistantMessage(providers.Message{
		Role: providers.RoleAssistant,
		Content: []providers.ContentPart{{
			Type:     contentTypeImageURL,
			ImageURL: &providers.ImageURL{URL: "https://example.test/image.png"},
		}},
	}, providerName)
	require.ErrorIs(t, err, errors.ErrInvalidRequest)
}

func TestConvertResponsePreservesUnnormalizedMessageFields(t *testing.T) {
	t.Parallel()

	const rawMessage = `{"role":"assistant","content":"","refusal":"","annotations":[],` +
		`"audio":null,"function_call":{"name":"legacy","arguments":"{}"},` +
		`"tool_calls":[{"id":"call-1","type":"custom","custom":{"name":"shell","input":""}}]}`
	var message oaisdk.ChatCompletionMessage
	require.NoError(t, json.Unmarshal([]byte(rawMessage), &message))

	converted := convertResponseMessage(message, providerName)
	encoded, err := json.Marshal(converted)
	require.NoError(t, err)
	var restored providers.Message
	require.NoError(t, json.Unmarshal(encoded, &restored))

	metadata, ok := restored.Extra[providerName][extraKeyResponseMessage].(map[string]any)
	require.True(t, ok)
	require.Equal(t, "", metadata["refusal"])
	require.Equal(t, []any{}, metadata["annotations"])
	require.Contains(t, metadata, "audio")

	replayed, err := convertAssistantMessage(restored, providerName)
	require.NoError(t, err)
	wire, err := json.Marshal(replayed)
	require.NoError(t, err)
	var body map[string]any
	require.NoError(t, json.Unmarshal(wire, &body))
	require.Equal(t, "", body["content"])
	require.Equal(t, "", body["refusal"])
	toolCalls, ok := body["tool_calls"].([]any)
	require.True(t, ok)
	customCall, ok := toolCalls[0].(map[string]any)
	require.True(t, ok)
	require.Equal(t, "custom", customCall["type"])
}

func TestConvertResponseUsesConfiguredMetadataNamespace(t *testing.T) {
	t.Parallel()

	const name = "azureopenai"
	var response oaisdk.ChatCompletionMessage
	require.NoError(t, json.Unmarshal([]byte(`{"role":"assistant","content":"","refusal":""}`), &response))

	message := convertResponseMessage(response, name)
	require.Contains(t, message.Extra, name)
	require.NotContains(t, message.Extra, providerName)

	_, err := convertAssistantMessage(message, name)
	require.NoError(t, err)
}

func TestConvertAssistantMessageRejectsInvalidResponseMetadata(t *testing.T) {
	t.Parallel()

	for name, value := range map[string]any{
		"invalid JSON shape": "invalid",
		"unencodable value":  make(chan int),
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			_, err := convertAssistantMessage(providers.Message{
				Role: providers.RoleAssistant,
				Extra: map[string]providers.ProviderData{
					providerName: {extraKeyResponseMessage: value},
				},
			}, providerName)
			require.ErrorIs(t, err, errors.ErrInvalidRequest)
		})
	}
}

func TestConvertAssistantMessageRejectsEditsThatConflictWithResponseMetadata(t *testing.T) {
	t.Parallel()

	var response oaisdk.ChatCompletionMessage
	require.NoError(t, json.Unmarshal([]byte(`{
		"role":"assistant","content":"original","refusal":"", "tool_calls":[]
	}`), &response))
	message := convertResponseMessage(response, providerName)
	message.Content = "edited"

	_, err := convertAssistantMessage(message, providerName)
	require.ErrorIs(t, err, errors.ErrInvalidRequest)
}

func TestConvertChunkPreservesUnnormalizedDeltaFields(t *testing.T) {
	t.Parallel()

	var chunk oaisdk.ChatCompletionChunk
	require.NoError(t, json.Unmarshal([]byte(`{
		"choices":[{"index":0,"delta":{"content":"","refusal":"","function_call":{"name":"","arguments":""}}}]
	}`), &chunk))

	converted := convertChunk(&chunk, providerName)
	metadata, ok := converted.Choices[0].Delta.Extra[providerName][extraKeyResponseDelta].(json.RawMessage)
	require.True(t, ok)
	var delta map[string]any
	require.NoError(t, json.Unmarshal(metadata, &delta))
	require.Equal(t, "", delta["content"])
	require.Equal(t, "", delta["refusal"])
	require.Contains(t, delta, "function_call")
}

func TestConvertResponsePreservesAllZeroUsage(t *testing.T) {
	t.Parallel()

	var response oaisdk.ChatCompletion
	require.NoError(
		t,
		json.Unmarshal([]byte(`{"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}`), &response),
	)

	result := convertResponse(&response, providerName)
	require.Equal(t, &providers.Usage{}, result.Usage)
}

func TestConvertChunkPreservesAllZeroUsage(t *testing.T) {
	t.Parallel()

	var chunk oaisdk.ChatCompletionChunk
	require.NoError(
		t,
		json.Unmarshal([]byte(`{"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}`), &chunk),
	)

	result := convertChunk(&chunk, providerName)
	require.Equal(t, &providers.Usage{}, result.Usage)
}

func TestNewCompatible(t *testing.T) {
	// Note: Not using t.Parallel() here because child test uses t.Setenv.

	t.Run("creates provider with valid config", func(t *testing.T) {
		t.Parallel()

		baseCfg := CompatibleConfig{
			Name:           "test-provider",
			DefaultBaseURL: "http://localhost:8080/v1",
			DefaultAPIKey:  "test-key",
			RequireAPIKey:  false,
			Capabilities: providers.Capabilities{
				Completion: true,
			},
		}

		provider, err := NewCompatible(baseCfg)
		require.NoError(t, err)
		require.NotNil(t, provider)
		require.Equal(t, "test-provider", provider.Name())
	})

	t.Run("returns error when name is missing", func(t *testing.T) {
		t.Parallel()

		baseCfg := CompatibleConfig{
			DefaultBaseURL: "http://localhost:8080/v1",
		}

		provider, err := NewCompatible(baseCfg)
		require.Error(t, err)
		require.Nil(t, provider)
		require.Contains(t, err.Error(), "provider name is required")
	})

	t.Run("returns error when API key required but missing", func(t *testing.T) {
		t.Parallel()

		baseCfg := CompatibleConfig{
			Name:          "test-provider",
			APIKeyEnvVar:  "TEST_API_KEY",
			RequireAPIKey: true,
		}

		provider, err := NewCompatible(baseCfg)
		require.Error(t, err)
		require.Nil(t, provider)

		var missingKeyErr *errors.MissingAPIKeyError
		require.ErrorAs(t, err, &missingKeyErr)
	})

	t.Run("uses default API key when not required", func(t *testing.T) {
		t.Parallel()

		baseCfg := CompatibleConfig{
			Name:          "test-provider",
			DefaultAPIKey: "default-key",
			RequireAPIKey: false,
		}

		provider, err := NewCompatible(baseCfg)
		require.NoError(t, err)
		require.NotNil(t, provider)
	})

	t.Run("uses config base URL over default", func(t *testing.T) {
		t.Parallel()

		baseCfg := CompatibleConfig{
			Name:           "test-provider",
			DefaultBaseURL: "http://default:8080/v1",
			DefaultAPIKey:  "test-key",
		}

		provider, err := NewCompatible(baseCfg, config.WithBaseURL("http://custom:9090/v1"))
		require.NoError(t, err)
		require.NotNil(t, provider)
	})

	t.Run("uses environment variable for base URL", func(t *testing.T) {
		t.Setenv("TEST_BASE_URL", "http://env:8080/v1")

		baseCfg := CompatibleConfig{
			Name:           "test-provider",
			BaseURLEnvVar:  "TEST_BASE_URL",
			DefaultBaseURL: "http://default:8080/v1",
			DefaultAPIKey:  "test-key",
		}

		provider, err := NewCompatible(baseCfg)
		require.NoError(t, err)
		require.NotNil(t, provider)
	})
}

func TestNewCompatibleRequireBaseURL(t *testing.T) {
	// Note: Not using t.Parallel() because subtests use t.Setenv.

	const (
		envVar       = "TEST_COMPATIBLE_REQUIRE_BASEURL"
		providerName = "test-provider"
	)

	tests := []struct {
		name           string
		baseURLEnvVar  string
		defaultBaseURL string
		envValue       string
		withBaseURL    string
		requireBaseURL bool
		wantErr        string // empty means no error expected.
	}{
		{
			name:           "errors when required and no env var configured",
			requireBaseURL: true,
			wantErr:        providerName + " base URL is required (set via WithBaseURL option)",
		},
		{
			name:           "errors when required and env var name set but unset",
			baseURLEnvVar:  envVar,
			requireBaseURL: true,
			wantErr: providerName + ` base URL is required (set via WithBaseURL option or "` +
				envVar + `" env var)`,
		},
		{
			name:           "succeeds when required and WithBaseURL is provided",
			requireBaseURL: true,
			withBaseURL:    "http://custom:9090/v1",
		},
		{
			name:           "succeeds when required and env var resolves",
			baseURLEnvVar:  envVar,
			envValue:       "http://env:8080/v1",
			requireBaseURL: true,
		},
		{
			name:           "succeeds when required and DefaultBaseURL is set",
			defaultBaseURL: "http://default:8080/v1",
			requireBaseURL: true,
		},
		{
			name:           "does not error when not required and no URL resolves",
			requireBaseURL: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv(envVar, tc.envValue)

			baseCfg := CompatibleConfig{
				Name:           providerName,
				BaseURLEnvVar:  tc.baseURLEnvVar,
				DefaultAPIKey:  "test-key",
				DefaultBaseURL: tc.defaultBaseURL,
				RequireBaseURL: tc.requireBaseURL,
			}

			var opts []config.Option
			if tc.withBaseURL != "" {
				opts = append(opts, config.WithBaseURL(tc.withBaseURL))
			}

			provider, err := NewCompatible(baseCfg, opts...)

			if tc.wantErr != "" {
				require.EqualError(t, err, tc.wantErr)
				require.Nil(t, provider)
				return
			}

			require.NoError(t, err)
			require.NotNil(t, provider)
		})
	}
}

func TestCompatibleProviderCapabilities(t *testing.T) {
	t.Parallel()

	expectedCaps := providers.Capabilities{
		Completion:          true,
		CompletionStreaming: true,
		Embedding:           true,
	}

	baseCfg := CompatibleConfig{
		Name:         "test-provider",
		Capabilities: expectedCaps,
	}

	provider, err := NewCompatible(baseCfg)
	require.NoError(t, err)

	caps := provider.Capabilities()
	require.Equal(t, expectedCaps, caps)
}

func TestValidateCompletionParams(t *testing.T) {
	t.Parallel()

	t.Run("returns error when model is empty", func(t *testing.T) {
		t.Parallel()

		params := providers.CompletionParams{
			Messages: []providers.Message{{Role: providers.RoleUser, Content: "Hello"}},
		}

		err := validateCompletionParams(params, providerName)
		require.Error(t, err)
		require.Contains(t, err.Error(), "model is required")
	})

	t.Run("returns error when messages is empty", func(t *testing.T) {
		t.Parallel()

		params := providers.CompletionParams{
			Model:    "gpt-4",
			Messages: []providers.Message{},
		}

		err := validateCompletionParams(params, providerName)
		require.Error(t, err)
		require.Contains(t, err.Error(), "at least one message is required")
	})

	t.Run("returns error for unknown message role", func(t *testing.T) {
		t.Parallel()

		params := providers.CompletionParams{
			Model: "gpt-4",
			Messages: []providers.Message{
				{Role: "unknown_role", Content: "Hello"},
			},
		}

		err := validateCompletionParams(params, providerName)
		require.Error(t, err)
		require.Contains(t, err.Error(), "unknown message role")
	})

	t.Run("accepts valid params", func(t *testing.T) {
		t.Parallel()

		params := providers.CompletionParams{
			Model: "gpt-4",
			Messages: []providers.Message{
				{Role: providers.RoleUser, Content: "Hello"},
			},
		}

		err := validateCompletionParams(params, providerName)
		require.NoError(t, err)
	})

	t.Run("leaves numeric ranges to the compatible API", func(t *testing.T) {
		t.Parallel()

		temperature := 3.0
		params := providers.CompletionParams{
			Model:       "provider-specific-model",
			Messages:    []providers.Message{{Role: providers.RoleUser, Content: "Hello"}},
			Temperature: &temperature,
			Stop:        []string{"one", "two", "three", "four", "five"},
		}

		err := validateCompletionParams(params, providerName)
		require.NoError(t, err)
	})
}

func TestConvertResponseFormat(t *testing.T) {
	t.Parallel()

	t.Run("handles nil format", func(t *testing.T) {
		t.Parallel()

		result := convertResponseFormat(nil)
		require.NotNil(t, result)
	})

	t.Run("converts json_object format", func(t *testing.T) {
		t.Parallel()

		format := &providers.ResponseFormat{Type: responseFormatJSONObject}
		result := convertResponseFormat(format)
		require.NotNil(t, result.OfJSONObject)
	})

	t.Run("converts json_schema format", func(t *testing.T) {
		t.Parallel()

		strict := true
		format := &providers.ResponseFormat{
			Type: responseFormatJSONSchema,
			JSONSchema: &providers.JSONSchema{
				Name:        "test_schema",
				Description: "Test schema",
				Schema:      map[string]any{"type": "object"},
				Strict:      &strict,
			},
		}
		result := convertResponseFormat(format)
		require.NotNil(t, result.OfJSONSchema)
	})

	t.Run("defaults to text format for unknown type", func(t *testing.T) {
		t.Parallel()

		format := &providers.ResponseFormat{Type: "unknown"}
		result := convertResponseFormat(format)
		require.NotNil(t, result.OfText)
	})
}

func TestConvertEmbeddingParams(t *testing.T) {
	t.Parallel()

	t.Run("converts string input", func(t *testing.T) {
		t.Parallel()

		params := providers.EmbeddingParams{
			Model: "text-embedding-3-small",
			Input: "Hello, world!",
		}

		result := convertEmbeddingParams(params)
		require.NotNil(t, result.Input.OfString)
	})

	t.Run("converts string array input", func(t *testing.T) {
		t.Parallel()

		params := providers.EmbeddingParams{
			Model: "text-embedding-3-small",
			Input: []string{"Hello", "World"},
		}

		result := convertEmbeddingParams(params)
		require.NotNil(t, result.Input.OfArrayOfStrings)
	})

	t.Run("handles unknown input type", func(t *testing.T) {
		t.Parallel()

		params := providers.EmbeddingParams{
			Model: "text-embedding-3-small",
			Input: 12345, // Unsupported type.
		}

		result := convertEmbeddingParams(params)
		// Should convert to string representation.
		require.NotNil(t, result.Input.OfString)
	})

	t.Run("includes optional parameters", func(t *testing.T) {
		t.Parallel()

		dims := 256
		params := providers.EmbeddingParams{
			Model:          "text-embedding-3-small",
			Input:          "Hello",
			EncodingFormat: "float",
			Dimensions:     &dims,
			User:           "test-user",
		}

		result := convertEmbeddingParams(params)
		require.Equal(t, int64(256), result.Dimensions.Value)
		require.Equal(t, "test-user", result.User.Value)
	})
}

func TestStreamingContextCancellation(t *testing.T) {
	t.Parallel()

	t.Run("respects context cancellation", func(t *testing.T) {
		t.Parallel()

		baseCfg := CompatibleConfig{
			Name:           "test-provider",
			DefaultBaseURL: "http://localhost:9999/v1", // Non-existent server.
			DefaultAPIKey:  "test-key",
		}

		provider, err := NewCompatible(baseCfg)
		require.NoError(t, err)

		ctx, cancel := context.WithCancel(context.Background())
		cancel() // Cancel immediately.

		params := providers.CompletionParams{
			Model:    "test-model",
			Messages: []providers.Message{{Role: providers.RoleUser, Content: "Hello"}},
		}

		chunks, errs := provider.CompletionStream(ctx, params)

		// Drain channels.
		for range chunks {
		}
		<-errs

		// Test passes if it doesn't hang.
	})

	// Regression for #85: when the caller cancels the context, the consumer
	// reading from `errs` should receive `context.Canceled` (not a closed
	// channel with no value) so it can distinguish "stream finished cleanly"
	// from "I cancelled the request".
	t.Run("surfaces ctx.Err on cancellation", func(t *testing.T) {
		t.Parallel()

		// Slow upstream: holds the connection open until the test cancels.
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
			<-r.Context().Done()
		}))
		t.Cleanup(srv.Close)

		provider, err := NewCompatible(CompatibleConfig{
			Name:           "test-provider",
			DefaultBaseURL: srv.URL + "/v1",
			DefaultAPIKey:  "test-key",
		})
		require.NoError(t, err)

		ctx, cancel := context.WithCancel(context.Background())
		params := providers.CompletionParams{
			Model:    "test-model",
			Messages: []providers.Message{{Role: providers.RoleUser, Content: "Hello"}},
		}

		chunks, errs := provider.CompletionStream(ctx, params)

		go func() {
			time.Sleep(50 * time.Millisecond)
			cancel()
		}()

		// Drain chunks until the channel closes.
		for range chunks {
		}

		select {
		case got, ok := <-errs:
			require.True(t, ok, "errs should yield a value before close")
			require.ErrorIs(t, got, context.Canceled)
		case <-time.After(2 * time.Second):
			t.Fatal("expected an error on errs after cancellation, got nothing")
		}
	})
}
