package openai

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	openai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/providers"
)

func TestModerationMultimodalWireAndRaw(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/moderations", r.URL.Path)
		var body map[string]any
		if !assert.NoError(t, json.NewDecoder(r.Body).Decode(&body)) {
			return
		}
		assert.Len(t, body["input"], 2)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(
			w,
			`{"id":"mod_1","model":"omni-moderation-latest","results":[{`+
				`"flagged":true,"categories":{"violence":true},"category_scores":{"violence":0.9},`+
				`"category_applied_input_types":{"violence":["image"]}}]}`,
		)
	}))
	t.Cleanup(server.Close)
	p, err := New(config.WithAPIKey("test"), config.WithBaseURL(server.URL))
	require.NoError(t, err)
	result, err := p.Moderation(
		t.Context(),
		providers.ModerationParams{
			IncludeRaw: true,
			Input: []providers.ContentPart{
				{Type: "text", Text: "hello"},
				{
					Type:     "image_url",
					ImageURL: &providers.ImageURL{URL: "https://example.test/a.png"},
				},
			},
		},
	)
	require.NoError(t, err)
	require.True(t, result.Results[0].Flagged)
	require.True(t, result.Results[0].Categories["violence"])
	require.JSONEq(
		t,
		`{"flagged":true,"categories":{"violence":true},"category_scores":{"violence":0.9},`+
			`"category_applied_input_types":{"violence":["image"]}}`,
		string(result.Results[0].ProviderRaw),
	)
}

func TestResponsesLifecycleWire(t *testing.T) {
	t.Parallel()

	requests := make(chan string, 7)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests <- r.Method + " " + r.URL.RequestURI()
		assert.Equal(t, "Bearer test", r.Header.Get("Authorization"))
		w.Header().Set("Content-Type", "application/json")
		switch {
		case strings.HasSuffix(r.URL.Path, "/input_items"):
			_, _ = io.WriteString(
				w,
				`{"object":"list","data":[],"first_id":"","last_id":"","has_more":false}`,
			)
		case r.Method == http.MethodDelete:
			w.WriteHeader(http.StatusNoContent)
		case strings.Contains(r.Header.Get("Accept"), "text/event-stream"):
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, "data: [DONE]\n\n")
		default:
			_, _ = io.WriteString(
				w,
				`{"id":"resp_1","object":"response","status":"completed","output":[]}`,
			)
		}
	}))
	t.Cleanup(server.Close)
	p, err := New(config.WithAPIKey("test"), config.WithBaseURL(server.URL))
	require.NoError(t, err)

	_, err = p.CreateResponse(t.Context(), responses.ResponseNewParams{Model: "gpt-4o"})
	require.NoError(t, err)
	_, err = p.RetrieveResponse(t.Context(), "resp_1", responses.ResponseGetParams{})
	require.NoError(t, err)
	require.NoError(t, p.DeleteResponse(t.Context(), "resp_1"))
	_, err = p.CancelResponse(t.Context(), "resp_1")
	require.NoError(t, err)
	_, err = p.ListResponseInputItems(t.Context(), "resp_1", responses.InputItemListParams{})
	require.NoError(t, err)
	events, errs := p.StreamResponse(t.Context(), responses.ResponseNewParams{Model: "gpt-4o"})
	require.Empty(t, collectOpenAIResponseStream(t, events, errs))
	events, errs = p.ReplayResponse(t.Context(), "resp_1", responses.ResponseGetParams{})
	require.Empty(t, collectOpenAIResponseStream(t, events, errs))

	want := []string{
		"POST /responses",
		"GET /responses/resp_1",
		"DELETE /responses/resp_1",
		"POST /responses/resp_1/cancel",
		"GET /responses/resp_1/input_items",
		"POST /responses",
		"GET /responses/resp_1",
	}
	requireRequests(t, requests, want)
}

func requireRequests(t *testing.T, requests <-chan string, expectedRequests []string) {
	t.Helper()
	for _, expected := range expectedRequests {
		require.Equal(t, expected, <-requests)
	}
}

func collectOpenAIResponseStream(
	t *testing.T,
	events <-chan responses.ResponseStreamEventUnion,
	errs <-chan error,
) []responses.ResponseStreamEventUnion {
	t.Helper()
	var result []responses.ResponseStreamEventUnion
	for event := range events {
		result = append(result, event)
	}
	for err := range errs {
		require.NoError(t, err)
	}
	return result
}

func TestImageAudioAndFileOperationsWire(t *testing.T) {
	t.Parallel()

	requests := make(chan string, 8)
	handler := imageAudioFileHandler(t)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests <- r.Method + " " + r.URL.Path
		handler.ServeHTTP(w, r)
	}))
	t.Cleanup(server.Close)
	p, err := New(config.WithAPIKey("test"), config.WithBaseURL(server.URL))
	require.NoError(t, err)

	testImageAndAudioOperations(t, p)
	testFileOperations(t, p)

	want := []string{
		"POST /images/generations",
		"POST /audio/transcriptions",
		"POST /audio/speech",
		"POST /files",
		"GET /files/file_1",
		"GET /files",
		"GET /files/file_1/content",
		"DELETE /files/file_1",
	}
	requireRequests(t, requests, want)
}

func testImageAndAudioOperations(t *testing.T, provider *Provider) {
	t.Helper()

	_, err := provider.GenerateImage(t.Context(), openai.ImageGenerateParams{Model: "gpt-image-1", Prompt: "fox"})
	require.NoError(t, err)
	_, err = provider.Transcribe(
		t.Context(),
		openai.AudioTranscriptionNewParams{
			File:  strings.NewReader("audio"),
			Model: "gpt-4o-transcribe",
		},
	)
	require.NoError(t, err)
	speech, err := provider.Speech(
		t.Context(),
		openai.AudioSpeechNewParams{
			Input: "hello",
			Model: "gpt-4o-mini-tts",
			Voice: openai.AudioSpeechNewParamsVoiceUnion{
				OfString: openai.String("alloy"),
			},
		},
	)
	require.NoError(t, err)
	require.NoError(t, speech.Body.Close())
}

func testFileOperations(t *testing.T, provider *Provider) {
	t.Helper()

	_, err := provider.UploadFile(
		t.Context(),
		openai.FileNewParams{File: strings.NewReader("{}"), Purpose: openai.FilePurposeBatch},
	)
	require.NoError(t, err)
	_, err = provider.RetrieveFile(t.Context(), "file_1")
	require.NoError(t, err)
	_, err = provider.ListFiles(t.Context(), openai.FileListParams{})
	require.NoError(t, err)
	content, err := provider.FileContent(t.Context(), "file_1")
	require.NoError(t, err)
	require.NoError(t, content.Body.Close())
	deleted, err := provider.DeleteFile(t.Context(), "file_1")
	require.NoError(t, err)
	require.True(t, deleted.Deleted)
}

func imageAudioFileHandler(t *testing.T) http.Handler {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("POST /images/generations", func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		assert.NoError(t, json.NewDecoder(r.Body).Decode(&body))
		assert.Equal(t, "gpt-image-1", body["model"])
		assert.Equal(t, "fox", body["prompt"])
		writeJSON(w, `{"created":1,"data":[]}`)
	})
	mux.HandleFunc("POST /audio/transcriptions", func(w http.ResponseWriter, r *http.Request) {
		r.Body = http.MaxBytesReader(w, r.Body, 2<<20)
		assert.NoError(t, r.ParseMultipartForm(1<<20))
		assert.Equal(t, "gpt-4o-transcribe", r.FormValue("model"))
		file, _, err := r.FormFile("file")
		if assert.NoError(t, err) {
			assert.NoError(t, file.Close())
		}
		writeJSON(w, `{"text":"hello"}`)
	})
	mux.HandleFunc("POST /audio/speech", func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		assert.NoError(t, json.NewDecoder(r.Body).Decode(&body))
		assert.Equal(t, "hello", body["input"])
		w.Header().Set("Content-Type", "audio/mpeg")
		_, _ = io.WriteString(w, "audio")
	})
	mux.HandleFunc("POST /files", func(w http.ResponseWriter, r *http.Request) {
		r.Body = http.MaxBytesReader(w, r.Body, 2<<20)
		assert.NoError(t, r.ParseMultipartForm(1<<20))
		assert.Equal(t, "batch", r.FormValue("purpose"))
		file, _, err := r.FormFile("file")
		if assert.NoError(t, err) {
			assert.NoError(t, file.Close())
		}
		writeJSON(w, fileJSON())
	})
	mux.HandleFunc("GET /files/file_1", func(w http.ResponseWriter, _ *http.Request) { writeJSON(w, fileJSON()) })
	mux.HandleFunc("GET /files", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, `{"object":"list","data":[],"first_id":"","last_id":"","has_more":false}`)
	})
	mux.HandleFunc("GET /files/file_1/content", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, "content")
	})
	mux.HandleFunc("DELETE /files/file_1", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, `{"id":"file_1","object":"file","deleted":true}`)
	})
	return mux
}

func writeJSON(w http.ResponseWriter, body string) {
	w.Header().Set("Content-Type", "application/json")
	_, _ = io.WriteString(w, body)
}

func fileJSON() string {
	return `{"id":"file_1","bytes":2,"created_at":1,"filename":"batch.jsonl",` +
		`"object":"file","purpose":"batch","status":"processed"}`
}

func TestBatchLifecycleWire(t *testing.T) {
	t.Parallel()

	var uploaded string
	server := httptest.NewServer(batchLifecycleHandler(t, &uploaded))
	t.Cleanup(server.Close)
	p, err := New(config.WithAPIKey("test"), config.WithBaseURL(server.URL))
	require.NoError(t, err)
	batch, err := p.CreateBatch(
		t.Context(),
		providers.CreateBatchParams{
			Model: "gpt-4o-mini",
			Requests: []providers.BatchRequestItem{
				{
					CustomID: "req_1",
					Body: map[string]any{
						"messages": []any{map[string]any{"role": "user", "content": "hello"}},
					},
				},
			},
		},
	)
	require.NoError(t, err)
	require.Equal(t, "batch_1", batch.ID)
	require.Contains(t, uploaded, `"custom_id":"req_1"`)
	require.Contains(t, uploaded, `"model":"gpt-4o-mini"`)
	cancelled, err := p.CancelBatch(t.Context(), "batch_1", "openai")
	require.NoError(t, err)
	require.Equal(t, providers.BatchStatusCancelling, cancelled.Status)
	result, err := p.RetrieveBatchResults(t.Context(), "batch_1", "openai")
	require.NoError(t, err)
	require.Equal(t, "req_1", result.Results[0].CustomID)
	require.Equal(t, "chat_1", result.Results[0].Result.ID)
	require.Equal(t, "http_error", result.Results[1].Error.Code)
	require.Equal(t, "malformed_response", result.Results[2].Error.Code)
	require.Equal(t, "bad_request", result.Results[3].Error.Code)
}

func batchLifecycleHandler(t *testing.T, uploaded *string) http.Handler {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("POST /files", func(w http.ResponseWriter, r *http.Request) {
		r.Body = http.MaxBytesReader(w, r.Body, 2<<20)
		assert.NoError(t, r.ParseMultipartForm(1<<20))
		file, _, err := r.FormFile("file")
		if !assert.NoError(t, err) {
			return
		}
		defer func() { assert.NoError(t, file.Close()) }()
		data, err := io.ReadAll(file)
		assert.NoError(t, err)
		*uploaded = string(data)
		assert.Equal(t, "batch", r.FormValue("purpose"))
		writeJSON(
			w,
			`{"id":"file_in","bytes":1,"created_at":1,"filename":"batch.jsonl",`+
				`"object":"file","purpose":"batch","status":"processed"}`,
		)
	})
	mux.HandleFunc("POST /batches", func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		assert.NoError(t, json.NewDecoder(r.Body).Decode(&body))
		assert.Equal(t, "file_in", body["input_file_id"])
		writeJSON(w, batchJSON("in_progress", ""))
	})
	mux.HandleFunc("GET /batches/batch_1", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, batchJSON("completed", "file_out"))
	})
	mux.HandleFunc("POST /batches/batch_1/cancel", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, batchJSON("cancelling", ""))
	})
	mux.HandleFunc("GET /files/file_out/content", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, strings.Join([]string{
			`{"custom_id":"req_1","response":{"status_code":200,"body":{"id":"chat_1",` +
				`"object":"chat.completion","created":1,"model":"gpt-4o-mini","choices":[],` +
				`"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}}}`,
			`{"custom_id":"req_2","response":{"status_code":500,"body":{}}}`,
			`{"custom_id":"req_3","response":{"status_code":200,"body":{}}}`,
			`{"custom_id":"req_4","error":{"code":"bad_request","message":"bad input"}}`,
		}, "\n")+"\n")
	})
	return mux
}

func TestCreateBatchSupportsOfficialEndpoints(t *testing.T) {
	t.Parallel()

	endpoints := []string{
		"/v1/responses",
		"/v1/chat/completions",
		"/v1/embeddings",
		"/v1/completions",
		"/v1/moderations",
		"/v1/images/generations",
		"/v1/images/edits",
		"/v1/videos",
	}
	for _, endpoint := range endpoints {
		t.Run(endpoint, func(t *testing.T) {
			t.Parallel()

			testCreateBatchEndpoint(t, endpoint)
		})
	}
}

func testCreateBatchEndpoint(t *testing.T, endpoint string) {
	t.Helper()
	var uploaded string
	server := httptest.NewServer(batchEndpointHandler(t, endpoint, &uploaded))
	t.Cleanup(server.Close)
	p, err := New(config.WithAPIKey("test"), config.WithBaseURL(server.URL))
	require.NoError(t, err)
	body := map[string]any{"input": "hello"}
	if endpoint == defaultBatchEndpoint {
		body = map[string]any{"messages": []any{map[string]any{"role": "user", "content": "hello"}}}
	}
	batch, err := p.CreateBatch(t.Context(), providers.CreateBatchParams{
		Endpoint: endpoint,
		Model:    "test-model",
		Requests: []providers.BatchRequestItem{{CustomID: "one", Body: body}},
	})
	require.NoError(t, err)
	require.Equal(t, endpoint, batch.Endpoint)
	require.Contains(t, uploaded, `"url":"`+endpoint+`"`)
}

func batchEndpointHandler(t *testing.T, endpoint string, uploaded *string) http.Handler {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("POST /files", func(w http.ResponseWriter, r *http.Request) {
		r.Body = http.MaxBytesReader(w, r.Body, 2<<20)
		assert.NoError(t, r.ParseMultipartForm(1<<20))
		file, _, err := r.FormFile("file")
		if !assert.NoError(t, err) {
			return
		}
		defer func() { assert.NoError(t, file.Close()) }()
		data, err := io.ReadAll(file)
		assert.NoError(t, err)
		*uploaded = string(data)
		writeJSON(w, `{"id":"file_in","bytes":1,"created_at":1,"filename":"batch.jsonl",`+
			`"object":"file","purpose":"batch","status":"processed"}`)
	})
	mux.HandleFunc("POST /batches", func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		assert.NoError(t, json.NewDecoder(r.Body).Decode(&body))
		assert.Equal(t, endpoint, body["endpoint"])
		writeJSON(w, strings.Replace(batchJSON("in_progress", ""), defaultBatchEndpoint, endpoint, 1))
	})
	return mux
}

func TestCreateBatchDeletesUploadedFileWhenBatchCreationFails(t *testing.T) {
	t.Parallel()

	deleted := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/files":
			_, _ = io.WriteString(w, fileJSON())
		case r.Method == http.MethodPost && r.URL.Path == "/batches":
			w.WriteHeader(http.StatusBadRequest)
			_, _ = io.WriteString(w, `{"error":{"message":"batch creation failed","type":"invalid_request_error"}}`)
		case r.Method == http.MethodDelete && r.URL.Path == "/files/file_1":
			deleted = true
			_, _ = io.WriteString(w, `{"id":"file_1","object":"file","deleted":true}`)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)
	p, err := New(config.WithAPIKey("test"), config.WithBaseURL(server.URL))
	require.NoError(t, err)

	_, err = p.CreateBatch(t.Context(), validCreateBatchParams())
	require.ErrorContains(t, err, "batch creation failed")
	require.True(t, deleted)
}

func TestCreateBatchJoinsFileCleanupFailureWithBatchCreationFailure(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/files":
			_, _ = io.WriteString(w, fileJSON())
		case r.Method == http.MethodPost && r.URL.Path == "/batches":
			w.WriteHeader(http.StatusBadRequest)
			_, _ = io.WriteString(w, `{"error":{"message":"batch creation failed","type":"invalid_request_error"}}`)
		case r.Method == http.MethodDelete && r.URL.Path == "/files/file_1":
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = io.WriteString(w, `{"error":{"message":"file cleanup failed","type":"server_error"}}`)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)
	p, err := New(config.WithAPIKey("test"), config.WithBaseURL(server.URL))
	require.NoError(t, err)

	_, err = p.CreateBatch(t.Context(), validCreateBatchParams())
	require.ErrorContains(t, err, "batch creation failed")
	require.ErrorContains(t, err, "file cleanup failed")
}

func TestParseBatchOutputNonSuccessPreservesProviderErrorAndRawBody(t *testing.T) {
	t.Parallel()

	body := `{"error":{"code":"rate_limit","message":"slow down","type":"rate_limit_error"},"request_id":"req_1"}`
	item, err := parseBatchOutputItem([]byte(
		`{"custom_id":"one","response":{"status_code":429,"body":`+body+`}}`,
	), defaultBatchEndpoint)
	require.NoError(t, err)
	require.Equal(t, "rate_limit", item.Error.Code)
	require.Equal(t, "slow down", item.Error.Message)
	require.JSONEq(t, body, string(item.Raw))
}

func TestParseBatchOutputNonSuccessFallsBackToHTTPStatusAndPreservesRawBody(t *testing.T) {
	t.Parallel()

	body := `{"unexpected":"diagnostic"}`
	item, err := parseBatchOutputItem([]byte(
		`{"custom_id":"one","response":{"status_code":503,"body":`+body+`}}`,
	), defaultBatchEndpoint)
	require.NoError(t, err)
	require.Equal(t, "http_error", item.Error.Code)
	require.Equal(t, "batch request returned HTTP 503", item.Error.Message)
	require.JSONEq(t, body, string(item.Raw))
}

func validCreateBatchParams() providers.CreateBatchParams {
	return providers.CreateBatchParams{
		Model: "gpt-4o-mini",
		Requests: []providers.BatchRequestItem{{
			CustomID: "one",
			Body: map[string]any{
				"messages": []any{map[string]any{"role": "user", "content": "hello"}},
			},
		}},
	}
}

func TestRetrieveBatchResultsPreservesNonChatBody(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/batches/batch_1":
			_, _ = io.WriteString(
				w,
				strings.Replace(
					batchJSON("completed", "file_out"),
					defaultBatchEndpoint,
					"/v1/responses",
					1,
				),
			)
		case "/files/file_out/content":
			_, _ = io.WriteString(
				w,
				`{"custom_id":"one","response":{"status_code":200,"body":{"id":"resp_1","output":[]}}}`+"\n",
			)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)
	p, err := New(config.WithAPIKey("test"), config.WithBaseURL(server.URL))
	require.NoError(t, err)
	result, err := p.RetrieveBatchResults(t.Context(), "batch_1", providerName)
	require.NoError(t, err)
	require.JSONEq(t, `{"id":"resp_1","output":[]}`, string(result.Results[0].Raw))
	require.Nil(t, result.Results[0].Result)
}

func TestListBatchesHonorsTotalLimit(t *testing.T) {
	t.Parallel()

	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		assert.Equal(t, "2", r.URL.Query().Get("limit"))
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(
			w,
			`{"object":"list","data":[`+batchJSON(
				"completed",
				"file_1",
			)+`,`+strings.ReplaceAll(
				batchJSON("completed", "file_2"),
				"batch_1",
				"batch_2",
			)+`],"has_more":true,"last_id":"batch_2"}`,
		)
	}))
	t.Cleanup(server.Close)
	p, err := New(config.WithAPIKey("test"), config.WithBaseURL(server.URL))
	require.NoError(t, err)

	limit := 2
	batches, err := p.ListBatches(
		t.Context(),
		"openai",
		providers.ListBatchesOptions{Limit: &limit},
	)
	require.NoError(t, err)
	require.Len(t, batches, 2)
	require.Equal(t, 1, requests)
}

func TestResourceValidationDoesNotSend(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(
		http.HandlerFunc(
			func(http.ResponseWriter, *http.Request) { t.Fatal("unexpected request") },
		),
	)
	t.Cleanup(server.Close)
	p, err := New(config.WithAPIKey("test"), config.WithBaseURL(server.URL))
	require.NoError(t, err)
	_, err = p.RetrieveResponse(t.Context(), "", responses.ResponseGetParams{})
	require.Error(t, err)
	require.Error(t, p.DeleteResponse(t.Context(), ""))
	_, err = p.CancelResponse(t.Context(), "")
	require.Error(t, err)
	_, err = p.RetrieveFile(t.Context(), "")
	require.Error(t, err)
	_, err = p.CreateBatch(t.Context(), providers.CreateBatchParams{})
	require.Error(t, err)
}

func TestTypedCreateValidationDoesNotSend(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(
		http.HandlerFunc(
			func(http.ResponseWriter, *http.Request) { t.Fatal("unexpected request") },
		),
	)
	t.Cleanup(server.Close)
	p, err := New(config.WithAPIKey("test"), config.WithBaseURL(server.URL))
	require.NoError(t, err)

	_, err = p.GenerateImage(t.Context(), openai.ImageGenerateParams{})
	require.Error(t, err)
	_, err = p.Transcribe(t.Context(), openai.AudioTranscriptionNewParams{})
	require.Error(t, err)
	speech, err := p.Speech(t.Context(), openai.AudioSpeechNewParams{})
	if speech != nil {
		require.NoError(t, speech.Body.Close())
	}
	require.Error(t, err)
	_, err = p.CreateOpenAIBatch(t.Context(), openai.BatchNewParams{})
	require.Error(t, err)
	_, err = p.UploadFile(t.Context(), openai.FileNewParams{})
	require.Error(t, err)
}

func TestCreateBatchRejectsInvalidRequestsBeforeUpload(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(
		http.HandlerFunc(
			func(http.ResponseWriter, *http.Request) { t.Fatal("unexpected request") },
		),
	)
	t.Cleanup(server.Close)
	p, err := New(config.WithAPIKey("test"), config.WithBaseURL(server.URL))
	require.NoError(t, err)
	validBody := map[string]any{
		"messages": []any{map[string]any{"role": "user", "content": "hello"}},
	}

	tests := []providers.CreateBatchParams{
		{
			Endpoint: "/v1/unknown",
			Model:    "gpt-4o",
			Requests: []providers.BatchRequestItem{{CustomID: "one", Body: validBody}},
		},
		{
			Model: "gpt-4o",
			Requests: []providers.BatchRequestItem{
				{CustomID: "same", Body: validBody},
				{CustomID: "same", Body: validBody},
			},
		},
		{
			Model: "gpt-4o",
			Requests: []providers.BatchRequestItem{
				{
					CustomID: "one",
					Body:     map[string]any{"model": "gpt-4.1", "messages": validBody["messages"]},
				},
			},
		},
	}
	for _, params := range tests {
		_, err := p.CreateBatch(t.Context(), params)
		require.Error(t, err)
	}
}

func TestNormalizeBatchRequestAcceptsTypedMessages(t *testing.T) {
	t.Parallel()

	body := map[string]any{
		"messages": []providers.Message{{Role: providers.RoleUser, Content: "hello"}},
	}

	normalized, err := normalizeBatchRequest("gpt-4o", body)
	require.NoError(t, err)
	require.NotContains(t, body, "model")
	require.Equal(t, "gpt-4o", normalized["model"])

	messages, ok := normalized["messages"].([]any)
	require.True(t, ok)
	require.Len(t, messages, 1)
	require.Equal(t, map[string]any{"role": providers.RoleUser, "content": "hello"}, messages[0])
}

func TestBuildBatchJSONLPreservesServiceValidatedFields(t *testing.T) {
	t.Parallel()

	const functionToolType = "function"

	data, err := buildBatchJSONL(
		"gpt-4o",
		defaultBatchEndpoint,
		[]providers.BatchRequestItem{{
			CustomID: "one",
			Body: map[string]any{
				"messages": []map[string]any{{
					"role":       providers.RoleAssistant,
					"content":    nil,
					"tool_calls": []map[string]any{{"id": "call_1", "type": functionToolType}},
				}},
			},
		}},
	)
	require.NoError(t, err)
	require.Contains(t, string(data), `"content":null`)
	require.Contains(t, string(data), `"tool_calls":[{"id":"call_1","type":"function"}]`)
}

func batchJSON(status, output string) string {
	return strings.NewReplacer("STATUS", status, "OUTPUT", output).
		Replace(`{"id":"batch_1","completion_window":"24h","created_at":1,` +
			`"endpoint":"/v1/chat/completions","input_file_id":"file_in","object":"batch",` +
			`"status":"STATUS","output_file_id":"OUTPUT",` +
			`"request_counts":{"completed":1,"failed":0,"total":1}}`)
}
