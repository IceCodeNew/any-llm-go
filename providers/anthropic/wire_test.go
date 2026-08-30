package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/providers"
)

const wireMessage = `{"id":"msg_1","type":"message","role":"assistant",` +
	`"content":[{"type":"text","text":"ok"}],"model":"claude-test",` +
	`"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1}}`

func newWireProvider(t *testing.T, handler http.HandlerFunc) *Provider {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	provider, err := New(config.WithAPIKey("test-key"), config.WithBaseURL(server.URL))
	require.NoError(t, err)
	return provider
}

func nativeParams() anthropic.MessageNewParams {
	return anthropic.MessageNewParams{
		Model: "claude-test", MaxTokens: 16,
		Messages: []anthropic.MessageParam{anthropic.NewUserMessage(anthropic.NewTextBlock("hello"))},
	}
}

func TestNativeMessagesAndBetaMessagesWire(t *testing.T) {
	t.Parallel()

	requests := 0
	provider := newWireProvider(t, func(w http.ResponseWriter, r *http.Request) {
		requests++
		if r.URL.Path != "/v1/messages" {
			t.Errorf("request path = %q, want /v1/messages", r.URL.Path)
		}
		if key := r.Header.Get("X-Api-Key"); key != "test-key" {
			t.Errorf("X-Api-Key = %q, want test-key", key)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
		}
		if body["model"] != "claude-test" {
			t.Errorf("model = %v, want claude-test", body["model"])
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, wireMessage)
	})

	message, err := provider.Messages(t.Context(), nativeParams())
	require.NoError(t, err)
	require.Equal(t, "msg_1", message.ID)
	betaParams := anthropic.BetaMessageNewParams{
		Model: "claude-test", MaxTokens: 16,
		Messages: []anthropic.BetaMessageParam{anthropic.NewBetaUserMessage(anthropic.NewBetaTextBlock("hello"))},
	}
	beta, err := provider.BetaMessages(t.Context(), betaParams)
	require.NoError(t, err)
	require.Equal(t, "msg_1", beta.ID)
	require.Equal(t, 2, requests)
}

func TestNativeStreamingErrorAndCancellationCloseChannels(t *testing.T) {
	t.Parallel()

	t.Run("error", func(t *testing.T) {
		t.Parallel()

		provider := newWireProvider(t, func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = fmt.Fprint(
				w,
				"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":\"broken\"}}\n\n",
			)
		})
		events, errs := provider.MessagesStreaming(t.Context(), nativeParams())
		eventCount := 0
		for range events {
			eventCount++
		}
		require.Zero(t, eventCount)
		require.Error(t, <-errs)
		_, open := <-errs
		require.False(t, open)
	})

	t.Run("cancel", func(t *testing.T) {
		t.Parallel()

		started := make(chan struct{})
		provider := newWireProvider(t, func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			if flusher, ok := w.(http.Flusher); ok {
				flusher.Flush()
			}
			close(started)
			<-r.Context().Done()
		})
		ctx, cancel := context.WithCancel(t.Context())
		events, errs := provider.MessagesStreaming(ctx, nativeParams())
		<-started
		cancel()
		eventCount := 0
		for range events {
			eventCount++
		}
		require.Zero(t, eventCount)
		require.ErrorIs(t, <-errs, context.Canceled)
	})
}

func TestBatchCreateValidationAndFullRequestWire(t *testing.T) {
	t.Parallel()

	provider := newWireProvider(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages/batches" {
			t.Errorf("request path = %q, want /v1/messages/batches", r.URL.Path)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		requests := wireSlice(t, body["requests"])
		params := wireMap(t, wireMap(t, requests[0])["params"])
		assertToolAndErrorWire(t, params)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, batchJSON("batch_1", "in_progress", false))
	})

	body := map[string]any{
		"max_tokens": 16,
		"messages": []any{map[string]any{"role": "user", "content": []any{
			map[string]any{
				"type":          "tool_result",
				"tool_use_id":   "call_1",
				"content":       "bad",
				"is_error":      true,
				"cache_control": map[string]any{"type": "ephemeral"},
			},
		}}},
		"tools": []any{map[string]any{"name": "lookup", "description": "Lookup", "input_schema": map[string]any{
			"type":       "object",
			"properties": map[string]any{"city": map[string]any{"type": "string"}},
			"required":   []string{"city"},
		}, "cache_control": map[string]any{"type": "ephemeral", "ttl": "1h"}}},
	}
	batch, err := provider.CreateBatch(
		t.Context(),
		providers.CreateBatchParams{
			Model:    "claude-test",
			Requests: []providers.BatchRequestItem{{CustomID: "one", Body: body}},
		},
	)
	require.NoError(t, err)
	require.Equal(t, providers.BatchStatusInProgress, batch.Status)

	_, err = provider.CreateBatch(t.Context(), providers.CreateBatchParams{})
	require.ErrorContains(t, err, "required")
	for _, params := range []providers.CreateBatchParams{
		{CompletionWindow: "1h", Requests: []providers.BatchRequestItem{{CustomID: "one"}}},
		{Endpoint: "/v1/responses", Requests: []providers.BatchRequestItem{{CustomID: "one"}}},
		{Metadata: map[string]string{"team": "search"}, Requests: []providers.BatchRequestItem{{CustomID: "one"}}},
	} {
		_, err = provider.CreateBatch(t.Context(), params)
		require.Error(t, err)
	}
	_, err = provider.CreateBatch(
		t.Context(),
		providers.CreateBatchParams{Requests: []providers.BatchRequestItem{{CustomID: "same"}, {CustomID: "same"}}},
	)
	require.ErrorContains(t, err, "duplicate")
}

func TestCompletionWirePreservesToolSchemaCacheAndErrorMetadata(t *testing.T) {
	t.Parallel()

	provider := newWireProvider(t, func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		assertToolAndErrorWire(t, body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, wireMessage)
	})

	_, err := provider.Completion(t.Context(), providers.CompletionParams{
		Model: "claude-test",
		Messages: []providers.Message{
			{
				Role: providers.RoleSystem,
				Content: []providers.ContentPart{
					{Type: "text", Text: "cached", CacheControl: &providers.CacheControl{Type: "ephemeral"}},
				},
			},
			{
				Role:       providers.RoleTool,
				ToolCallID: "call_1",
				Content:    "bad",
				Extra:      map[string]providers.ProviderData{providerName: {"is_error": true}},
			},
		},
		Tools: []providers.Tool{{
			Type: "function", CacheControl: &providers.CacheControl{Type: "ephemeral", TTL: "1h"},
			Function: providers.Function{Name: "lookup", Parameters: map[string]any{
				"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}},
				"required": []string{"city"}, "additionalProperties": false,
			}},
		}},
	})
	require.NoError(t, err)
}

func TestCompletionWireForwardsAllSamplingParameters(t *testing.T) {
	t.Parallel()

	provider := newWireProvider(t, func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if !assert.NoError(t, json.NewDecoder(r.Body).Decode(&body)) {
			http.Error(w, "invalid request body", http.StatusBadRequest)

			return
		}
		assert.InDelta(t, 0.2, body["temperature"], 1e-9)
		assert.InDelta(t, 0.8, body["top_p"], 1e-9)
		assert.InDelta(t, 40, body["top_k"], 1e-9)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, wireMessage)
	})
	temperature, topP, topK := 0.2, 0.8, 40
	_, err := provider.Completion(t.Context(), providers.CompletionParams{
		Model: "claude-test", Messages: []providers.Message{{Role: providers.RoleUser, Content: "hello"}},
		Temperature: &temperature, TopP: &topP, TopK: &topK,
	})
	require.NoError(t, err)
}

func assertToolAndErrorWire(t *testing.T, body map[string]any) {
	t.Helper()

	tool := wireMap(t, wireSlice(t, body["tools"])[0])
	require.Equal(t, map[string]any{"type": "ephemeral", "ttl": "1h"}, tool["cache_control"])
	schema := wireMap(t, tool["input_schema"])
	require.Equal(t, []any{"city"}, schema["required"])
	require.Equal(t, map[string]any{"city": map[string]any{"type": "string"}}, schema["properties"])

	messages := wireSlice(t, body["messages"])
	content := wireSlice(t, wireMap(t, messages[len(messages)-1])["content"])
	toolResult := wireMap(t, content[0])
	require.Equal(t, true, toolResult["is_error"])
}

func wireMap(t *testing.T, value any) map[string]any {
	t.Helper()
	result, ok := value.(map[string]any)
	require.True(t, ok)
	return result
}

func wireSlice(t *testing.T, value any) []any {
	t.Helper()
	result, ok := value.([]any)
	require.True(t, ok)
	return result
}

func batchJSON(id, status string, ended bool) string {
	endedAt := "null"
	if ended {
		endedAt = `"2025-01-01T00:01:00Z"`
	}
	return fmt.Sprintf(
		`{"id":%q,"type":"message_batch","processing_status":%q,`+
			`"request_counts":{"processing":0,"succeeded":1,"errored":0,"canceled":0,"expired":0},`+
			`"ended_at":%s,"created_at":"2025-01-01T00:00:00Z","expires_at":"2025-01-02T00:00:00Z",`+
			`"cancel_initiated_at":null,"results_url":"https://example.test/results"}`,
		id,
		status,
		endedAt,
	)
}

func TestBatchListHonorsTotalLimitAcrossPages(t *testing.T) {
	t.Parallel()

	calls := 0
	provider := newWireProvider(t, func(w http.ResponseWriter, r *http.Request) {
		calls++
		if r.URL.Path != "/v1/messages/batches" {
			t.Errorf("request path = %q, want /v1/messages/batches", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		if calls == 1 {
			_, _ = fmt.Fprintf(
				w,
				`{"data":[%s,%s],"has_more":true,"first_id":"b1","last_id":"b2"}`,
				batchJSON("b1", "ended", true),
				batchJSON("b2", "ended", true),
			)
			return
		}
		_, _ = fmt.Fprintf(
			w,
			`{"data":[%s],"has_more":false,"first_id":"b3","last_id":"b3"}`,
			batchJSON("b3", "ended", true),
		)
	})
	limit := 2
	batches, err := provider.ListBatches(t.Context(), providerName, providers.ListBatchesOptions{Limit: &limit})
	require.NoError(t, err)
	require.Len(t, batches, 2)
	require.Equal(t, 1, calls)
	require.Equal(t, providers.BatchStatusCompleted, batches[0].Status)
}

func TestBatchResultsWireConversions(t *testing.T) {
	t.Parallel()

	provider := newWireProvider(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if strings.HasSuffix(r.URL.Path, "/results") {
			_, _ = fmt.Fprintf(
				w,
				"{\"custom_id\":\"ok\",\"result\":{\"type\":\"succeeded\",\"message\":%s}}\n",
				wireMessage,
			)
			_, _ = fmt.Fprintln(
				w,
				`{"custom_id":"bad","result":{"type":"errored",`+
					`"error":{"type":"error","error":`+
					`{"type":"invalid_request_error","message":"bad input"}}}}`,
			)
			_, _ = fmt.Fprintln(w, `{"custom_id":"cancel","result":{"type":"canceled"}}`)
			_, _ = fmt.Fprintln(w, `{"custom_id":"expire","result":{"type":"expired"}}`)
			return
		}
		_, _ = fmt.Fprint(w, batchJSON("batch_1", "ended", true))
	})
	result, err := provider.RetrieveBatchResults(t.Context(), "batch_1", providerName)
	require.NoError(t, err)
	require.Len(t, result.Results, 4)
	require.Equal(t, "ok", result.Results[0].CustomID)
	require.Equal(t, "invalid_request_error", result.Results[1].Error.Code)
	require.Equal(t, "canceled", result.Results[2].Error.Code)
	require.Equal(t, "expired", result.Results[3].Error.Code)
}
