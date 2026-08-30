package openai

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/providers"
)

func TestCapabilitiesIncludesResponses(t *testing.T) {
	t.Parallel()

	provider, err := New(config.WithAPIKey("test-key"))
	require.NoError(t, err)
	require.True(t, provider.Capabilities().Responses)

	_, ok := any(provider).(providers.ResponsesProvider)
	require.True(t, ok)
}

func TestCompatibleProviderDoesNotImplementResponses(t *testing.T) {
	t.Parallel()

	provider, err := NewCompatible(CompatibleConfig{
		Name:          "test-provider",
		DefaultAPIKey: "test-key",
		RequireAPIKey: false,
	})
	require.NoError(t, err)

	_, ok := any(provider).(providers.ResponsesProvider)
	require.False(t, ok)
}

func TestResponsesEmptyOutputText(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "resp_test",
			"object": "response",
			"created_at": 1700000000,
			"model": "gpt-4o-mini",
			"output": [],
			"status": "completed"
		}`))
	}))
	t.Cleanup(srv.Close)

	provider, err := New(
		config.WithAPIKey("test-key"),
		config.WithBaseURL(srv.URL),
	)
	require.NoError(t, err)

	result, err := provider.Responses(t.Context(), providers.ResponsesParams{
		Model: "gpt-4o-mini",
		Input: []providers.ResponsesInputItem{
			{Role: providers.RoleUser, Content: "hi"},
		},
	})
	require.NoError(t, err)
	require.Equal(t, "resp_test", result.ID)
	require.Equal(t, "completed", result.Status)
	require.Empty(t, result.Output)
	require.Empty(t, result.OutputItems)
}

func TestResponsesOutputItems(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "resp_tools",
			"object": "response",
			"created_at": 1700000000,
			"model": "gpt-4o-mini",
			"status": "completed",
			"output": [
				{
					"type": "reasoning",
					"id": "rs_1",
					"status": "completed",
					"summary": [{"type": "summary_text", "text": "thinking..."}]
				},
				{
					"type": "function_call",
					"id": "fc_1",
					"status": "completed",
					"call_id": "call_abc",
					"name": "get_weather",
					"arguments": "{\"city\": \"SF\"}"
				}
			]
		}`))
	}))
	t.Cleanup(srv.Close)

	provider, err := New(
		config.WithAPIKey("test-key"),
		config.WithBaseURL(srv.URL),
	)
	require.NoError(t, err)

	result, err := provider.Responses(t.Context(), providers.ResponsesParams{
		Model: "gpt-4o-mini",
		Input: []providers.ResponsesInputItem{
			{Role: providers.RoleUser, Content: "weather?"},
		},
	})
	require.NoError(t, err)
	require.Equal(t, "completed", result.Status)
	require.Empty(t, result.Output)
	require.Len(t, result.OutputItems, 2)

	require.Equal(t, "reasoning", result.OutputItems[0].Type)
	require.Equal(t, "rs_1", result.OutputItems[0].ID)
	require.Equal(t, "thinking...", result.OutputItems[0].Summary)

	require.Equal(t, "function_call", result.OutputItems[1].Type)
	require.Equal(t, "fc_1", result.OutputItems[1].ID)
	require.Equal(t, "get_weather", result.OutputItems[1].Name)
	require.Equal(t, "call_abc", result.OutputItems[1].CallID)
	require.JSONEq(t, `{"city": "SF"}`, result.OutputItems[1].Arguments)
}

func TestResponsesPreservesUsageAndRawEnvelope(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"resp_usage",
			"object":"response",
			"created_at":1700000000,
			"model":"o3-mini",
			"status":"incomplete",
			"output":[],
			"usage":{
				"input_tokens":11,
				"input_tokens_details":{"cached_tokens":3},
				"output_tokens":7,
				"output_tokens_details":{"reasoning_tokens":5},
				"total_tokens":18
			},
			"metadata":{"request_kind":"test"},
			"error":null,
			"incomplete_details":{"reason":"max_output_tokens"},
			"future_field":{"value":true}
		}`))
	}))
	t.Cleanup(srv.Close)

	provider, err := New(config.WithAPIKey("test-key"), config.WithBaseURL(srv.URL))
	require.NoError(t, err)

	result, err := provider.Responses(t.Context(), providers.ResponsesParams{
		Model: "o3-mini",
		Input: []providers.ResponsesInputItem{{Role: providers.RoleUser, Content: "think"}},
	})
	require.NoError(t, err)
	require.Equal(t, &providers.Usage{
		PromptTokens:     11,
		CompletionTokens: 7,
		TotalTokens:      18,
		ReasoningTokens:  5,
	}, result.Usage)

	// The Responses contract includes metadata, errors, incomplete details, and
	// evolving fields outside the portable result: https://developers.openai.com/api/reference/resources/responses
	var envelope map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(result.ProviderRaw, &envelope))
	require.JSONEq(t, `{"request_kind":"test"}`, string(envelope["metadata"]))
	require.JSONEq(t, `{"reason":"max_output_tokens"}`, string(envelope["incomplete_details"]))
	require.JSONEq(t, `{"value":true}`, string(envelope["future_field"]))
	require.Equal(t, "null", string(envelope["error"]))
}

func TestResponsesPreservesPresentAllZeroUsage(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"resp_zero_usage",
			"object":"response",
			"created_at":1700000000,
			"model":"gpt-4o-mini",
			"status":"completed",
			"output":[],
			"usage":{
				"input_tokens":0,
				"input_tokens_details":{"cached_tokens":0},
				"output_tokens":0,
				"output_tokens_details":{"reasoning_tokens":0},
				"total_tokens":0
			}
		}`))
	}))
	t.Cleanup(srv.Close)

	provider, err := New(config.WithAPIKey("test-key"), config.WithBaseURL(srv.URL))
	require.NoError(t, err)

	result, err := provider.Responses(t.Context(), providers.ResponsesParams{
		Model: "gpt-4o-mini",
		Input: []providers.ResponsesInputItem{{Role: providers.RoleUser, Content: "hi"}},
	})
	require.NoError(t, err)
	require.NotNil(t, result.Usage)
	require.Equal(t, providers.Usage{}, *result.Usage)
}
