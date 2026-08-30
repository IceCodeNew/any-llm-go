package gemini

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"google.golang.org/genai"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/providers"
)

func validBatchParams() providers.CreateBatchParams {
	return providers.CreateBatchParams{
		Model: "gemini-2.5-flash",
		Requests: []providers.BatchRequestItem{{
			CustomID: "request-1",
			Body: map[string]any{
				"messages": []any{map[string]any{"role": "user", "content": "hello"}},
			},
		}},
	}
}

func TestConvertCreateBatchParams(t *testing.T) {
	t.Parallel()

	provider, err := New(config.WithAPIKey("test-key"))
	require.NoError(t, err)

	t.Run("converts valid request and display name", func(t *testing.T) {
		t.Parallel()

		params := validBatchParams()
		params.Metadata = map[string]string{"display_name": "nightly"}
		source, cfg, err := provider.convertCreateBatchParams(params)
		require.NoError(t, err)
		require.Equal(t, "nightly", cfg.DisplayName)
		require.Len(t, source.InlinedRequests, 1)
		require.Equal(t, "request-1", source.InlinedRequests[0].Metadata["custom_id"])
	})

	t.Run("validates required arguments", func(t *testing.T) {
		t.Parallel()

		_, _, err := provider.convertCreateBatchParams(providers.CreateBatchParams{})
		require.Error(t, err)

		params := validBatchParams()
		params.CompletionWindow = "1h"
		_, _, err = provider.convertCreateBatchParams(params)
		require.Error(t, err)

		params = validBatchParams()
		params.Endpoint = "/v1/responses"
		_, _, err = provider.convertCreateBatchParams(params)
		require.Error(t, err)
	})

	t.Run("rejects duplicate custom IDs", func(t *testing.T) {
		t.Parallel()

		params := validBatchParams()
		params.Requests = append(params.Requests, params.Requests[0])
		_, _, err := provider.convertCreateBatchParams(params)
		require.Error(t, err)
	})

	t.Run("rejects request model mismatch", func(t *testing.T) {
		t.Parallel()

		params := validBatchParams()
		params.Requests[0].Body["model"] = "gemini-2.0-flash"
		_, _, err := provider.convertCreateBatchParams(params)
		require.Error(t, err)
	})

	t.Run("rejects unsupported metadata", func(t *testing.T) {
		t.Parallel()

		params := validBatchParams()
		params.Metadata = map[string]string{"display_name": "nightly", "team": "search"}
		_, _, err := provider.convertCreateBatchParams(params)
		require.Error(t, err)
	})
}

func TestConvertCreateBatchParamsInstallsThoughtSignatureBypassHook(t *testing.T) {
	t.Parallel()

	provider := &Provider{}
	params := validBatchParams()
	params.Requests[0].Body["messages"] = []any{map[string]any{
		"role": "assistant",
		"tool_calls": []any{map[string]any{
			"id": "call-1", "type": "function",
			"function": map[string]any{"name": "search", "arguments": `{"q":"test"}`},
		}},
	}}
	_, cfg, err := provider.convertCreateBatchParams(params)
	require.NoError(t, err)
	require.NotNil(t, cfg.HTTPOptions)
	require.NotNil(t, cfg.HTTPOptions.ExtrasRequestProvider)
}

func TestBatchArgumentValidation(t *testing.T) {
	t.Parallel()

	provider := &Provider{}
	require.Error(t, validateBatchProvider("openai"))
	_, err := provider.RetrieveBatch(t.Context(), "", providerName)
	require.Error(t, err)
	_, err = provider.RetrieveBatch(t.Context(), "batch-1", "openai")
	require.Error(t, err)
	limit := 0
	_, err = provider.ListBatches(t.Context(), providerName, providers.ListBatchesOptions{Limit: &limit})
	require.Error(t, err)
}

func TestConvertBatchStatus(t *testing.T) {
	t.Parallel()

	tests := map[genai.JobState]providers.BatchStatus{
		genai.JobStateSucceeded:          providers.BatchStatusCompleted,
		genai.JobStatePartiallySucceeded: providers.BatchStatusCompleted,
		genai.JobStateFailed:             providers.BatchStatusFailed,
		genai.JobStateCancelling:         providers.BatchStatusCancelling,
		genai.JobStateCancelled:          providers.BatchStatusCancelled,
		genai.JobStateExpired:            providers.BatchStatusExpired,
		genai.JobStateQueued:             providers.BatchStatusValidating,
		genai.JobStatePending:            providers.BatchStatusValidating,
		genai.JobStateRunning:            providers.BatchStatusInProgress,
	}
	for state, want := range tests {
		require.Equal(t, want, convertBatchStatus(state))
	}
}

func TestConvertBatch(t *testing.T) {
	t.Parallel()

	created := time.Unix(100, 0)
	started := time.Unix(110, 0)
	ended := time.Unix(120, 0)
	batch := convertBatch(&genai.BatchJob{
		Name: "batches/1", DisplayName: "nightly", State: genai.JobStateSucceeded,
		CreateTime: created, StartTime: started, EndTime: ended,
		Src:             &genai.BatchJobSource{FileName: "files/input"},
		Dest:            &genai.BatchJobDestination{FileName: "files/output"},
		CompletionStats: &genai.CompletionStats{SuccessfulCount: 2, FailedCount: 1, IncompleteCount: 1},
	})
	require.Equal(t, "batches/1", batch.ID)
	require.Equal(t, providers.BatchStatusCompleted, batch.Status)
	require.Equal(t, int64(100), batch.CreatedAt)
	require.Equal(t, int64(110), *batch.InProgressAt)
	require.Equal(t, int64(120), *batch.CompletedAt)
	require.Equal(t, &providers.BatchRequestCounts{Completed: 2, Failed: 1, Total: 4}, batch.RequestCounts)
	require.Equal(t, "files/input", batch.InputFileID)
	require.Equal(t, "files/output", batch.OutputFileID)
	require.Equal(t, map[string]string{"display_name": "nightly"}, batch.Metadata)

	params := validBatchParams()
	params.Metadata = batch.Metadata
	_, cfg, err := (&Provider{}).convertCreateBatchParams(params)
	require.NoError(t, err)
	require.Equal(t, "nightly", cfg.DisplayName)
	require.Zero(t, convertBatch(&genai.BatchJob{}).CreatedAt)
}

func TestConvertBatchResults(t *testing.T) {
	t.Parallel()

	code := int32(429)
	job := &genai.BatchJob{
		Model: "gemini-2.5-flash", State: genai.JobStatePartiallySucceeded,
		Dest: &genai.BatchJobDestination{InlinedResponses: []*genai.InlinedResponse{
			{Metadata: map[string]string{"custom_id": "success"}, Response: &genai.GenerateContentResponse{
				Candidates: []*genai.Candidate{
					{Content: genai.NewContentFromText("hello", roleModel), FinishReason: genai.FinishReasonStop},
				},
			}},
			{Metadata: map[string]string{"custom_id": "error"}, Error: &genai.JobError{Code: &code, Message: "quota"}},
			{Metadata: map[string]string{"custom_id": "absent"}},
		}},
	}
	require.True(t, batchResultsAvailable(job))
	result, err := convertBatchResults(job)
	require.NoError(t, err)
	require.Len(t, result.Results, 3)
	require.Equal(t, "hello", result.Results[0].Result.Choices[0].Message.ContentString())
	require.Equal(t, "429", result.Results[1].Error.Code)
	require.Equal(t, "quota", result.Results[1].Error.Message)
	require.Equal(t, "batch result has no response payload", result.Results[2].Error.Message)
}

func TestBatchResultsAvailable(t *testing.T) {
	t.Parallel()

	for _, state := range []genai.JobState{genai.JobStateSucceeded, genai.JobStatePartiallySucceeded} {
		require.True(t, batchResultsAvailable(&genai.BatchJob{
			State: state,
			Dest:  &genai.BatchJobDestination{},
		}))
	}
	require.False(t, batchResultsAvailable(&genai.BatchJob{State: genai.JobStatePartiallySucceeded}))
	require.False(t, batchResultsAvailable(&genai.BatchJob{
		State: genai.JobStateRunning,
		Dest:  &genai.BatchJobDestination{},
	}))
}
