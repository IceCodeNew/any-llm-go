package azureopenai

import (
	"context"
	"encoding/json"
	stderrors "errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/policy"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/internal/testutil"
	"github.com/mozilla-ai/any-llm-go/providers"
)

type testTokenCredential struct{}

func (testTokenCredential) GetToken(context.Context, policy.TokenRequestOptions) (azcore.AccessToken, error) {
	return azcore.AccessToken{Token: "test-token", ExpiresOn: time.Now().Add(time.Hour)}, nil
}

type closeErrorResponseStream struct {
	closeErr error
}

var errTestCloseResponseStream = stderrors.New("close failed")

func (*closeErrorResponseStream) Next() bool {
	return false
}

func (*closeErrorResponseStream) Current() responses.ResponseStreamEventUnion {
	var event responses.ResponseStreamEventUnion

	return event
}

func (*closeErrorResponseStream) Err() error {
	return nil
}

func (stream *closeErrorResponseStream) Close() error {
	return stream.closeErr
}

type scopeTokenCredential struct {
	scopes chan []string
}

func (c scopeTokenCredential) GetToken(
	_ context.Context,
	options policy.TokenRequestOptions,
) (azcore.AccessToken, error) {
	c.scopes <- options.Scopes
	return azcore.AccessToken{Token: "test-token", ExpiresOn: time.Now().Add(time.Hour)}, nil
}

type azureWireTest struct {
	name            string
	path            string
	body            string
	previewAPI      bool
	reasoningEffort providers.ReasoningEffort
	call            func(context.Context, *Provider) error
}

func TestNew(t *testing.T) {
	t.Parallel()

	provider, err := New(
		config.WithAPIKey("test-key"),
		config.WithBaseURL("https://example.openai.azure.com"),
	)
	require.NoError(t, err)
	require.NotNil(t, provider)
	require.Equal(t, providerName, provider.Name())
}

func TestV1BaseURL(t *testing.T) {
	t.Parallel()

	require.Equal(t, "https://example.openai.azure.com/openai/v1/", v1BaseURL("https://example.openai.azure.com/"))
	require.Equal(
		t,
		"https://example.services.ai.azure.com/openai/v1/",
		v1BaseURL("https://example.services.ai.azure.com/openai/v1/"),
	)
}

func TestNewFromEnvironment(t *testing.T) {
	t.Setenv(envAPIKey, "env-key")
	t.Setenv(envBaseURL, "https://example.openai.azure.com")

	provider, err := New()
	require.NoError(t, err)
	require.NotNil(t, provider)
}

func TestNewRequiresAPIKey(t *testing.T) {
	t.Setenv(envAPIKey, "")

	provider, err := New(config.WithBaseURL("https://example.openai.azure.com"))
	require.Nil(t, provider)
	require.Error(t, err)

	var missingKeyErr *errors.MissingAPIKeyError
	require.ErrorAs(t, err, &missingKeyErr)
	require.Equal(t, providerName, missingKeyErr.Provider)
	require.Equal(t, envAPIKey, missingKeyErr.EnvVar)
}

func TestNewRequiresEndpoint(t *testing.T) {
	t.Setenv(envAPIKey, "")
	t.Setenv(envBaseURL, "")

	provider, err := New(config.WithAPIKey("test-key"))
	require.Nil(t, provider)
	require.Error(t, err)
	require.Contains(t, err.Error(), "endpoint is required")
}

func TestNewRejectsConflictingAuthentication(t *testing.T) {
	t.Setenv(envAPIKey, "env-key")

	provider, err := New(
		WithTokenCredential(testTokenCredential{}),
		config.WithBaseURL("https://example.openai.azure.com"),
	)
	require.Nil(t, provider)
	require.ErrorContains(t, err, "both configured")
}

func TestNewRejectsInvalidTokenCredentialType(t *testing.T) {
	t.Setenv(envAPIKey, "")

	provider, err := New(
		config.WithExtra(extraTokenCredential, "not a credential"),
		config.WithBaseURL("https://example.openai.azure.com"),
	)
	require.Nil(t, provider)
	require.ErrorContains(t, err, "must implement azcore.TokenCredential")
}

func TestCapabilities(t *testing.T) {
	t.Parallel()

	provider, err := New(
		config.WithAPIKey("test-key"),
		config.WithBaseURL("https://example.openai.azure.com"),
	)
	require.NoError(t, err)

	caps := provider.Capabilities()
	require.True(t, caps.Completion)
	require.True(t, caps.CompletionReasoning)
	require.True(t, caps.CompletionStreaming)
	require.True(t, caps.CompletionTools)
	require.True(t, caps.Embedding)
	require.True(t, caps.AudioSpeech)
	require.True(t, caps.AudioTranscription)
	require.True(t, caps.ImageGeneration)
	require.True(t, caps.ListModels)
	require.True(t, caps.Responses)
	require.NotImplements(t, (*providers.ModerationProvider)(nil), provider)
}

func TestImageGenerationWireFormat(t *testing.T) {
	t.Parallel()

	tests := []azureWireTest{
		{
			name:       "image generation",
			path:       "/openai/v1/images/generations",
			body:       `{"created":1,"data":[]}`,
			previewAPI: true,
			call: func(ctx context.Context, p *Provider) error {
				_, err := p.GenerateImage(ctx, openai.ImageGenerateParams{Model: "deploy/image", Prompt: "fox"})
				return err
			},
		},
		{
			name:       "normal deployment name",
			path:       "/openai/v1/images/generations",
			body:       `{"created":1,"data":[]}`,
			previewAPI: true,
			call: func(ctx context.Context, p *Provider) error {
				_, err := p.GenerateImage(ctx, openai.ImageGenerateParams{Model: "image-deployment", Prompt: "fox"})
				return err
			},
		},
	}
	runAzureWireTests(t, tests, "test-key")
}

func TestAudioWireFormat(t *testing.T) {
	t.Parallel()

	tests := []azureWireTest{
		{
			name:       "transcription",
			path:       "/openai/v1/audio/transcriptions",
			body:       `{"text":"hello"}`,
			previewAPI: true,
			call: func(ctx context.Context, p *Provider) error {
				_, err := p.Transcribe(
					ctx,
					openai.AudioTranscriptionNewParams{Model: "deploy/audio", File: strings.NewReader("audio")},
				)
				return err
			},
		},
		{
			name:       "speech",
			path:       "/openai/v1/audio/speech",
			body:       "audio",
			previewAPI: true,
			call: func(ctx context.Context, p *Provider) error {
				resp, err := p.Speech(
					ctx,
					openai.AudioSpeechNewParams{
						Model: "deploy/speech",
						Input: "hello",
						Voice: openai.AudioSpeechNewParamsVoiceUnion{
							OfAudioSpeechNewsVoiceString2: openai.String("alloy"),
						},
					},
				)
				if resp != nil {
					if closeErr := resp.Body.Close(); err == nil {
						err = closeErr
					}
				}
				return err
			},
		},
	}
	runAzureWireTests(t, tests, "test-key")
}

func runAzureWireTest(t *testing.T, tc azureWireTest, apiKey string) {
	t.Helper()
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, tc.path, r.URL.EscapedPath())
		if tc.previewAPI {
			assert.Equal(t, previewAPIVersion, r.URL.Query().Get("api-version"))
		} else {
			assert.Empty(t, r.URL.RawQuery)
		}
		assert.Empty(t, r.Header.Get("Authorization"))
		assert.Equal(t, apiKey, r.Header.Get("Api-Key"))

		if tc.reasoningEffort != "" {
			var requestBody map[string]any
			if !assert.NoError(t, json.NewDecoder(r.Body).Decode(&requestBody)) {
				return
			}

			assert.Equal(t, string(tc.reasoningEffort), requestBody["reasoning_effort"])
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, tc.body)
	}))
	t.Cleanup(srv.Close)

	provider, err := New(
		config.WithAPIKey(apiKey),
		config.WithBaseURL(srv.URL),
	)
	require.NoError(t, err)
	require.NoError(t, tc.call(t.Context(), provider))
}

func TestTokenCredentialAuthentication(t *testing.T) {
	t.Parallel()
	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "Bearer test-token", r.Header.Get("Authorization"))
		assert.Empty(t, r.Header.Get("Api-Key"))
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"object":"list","data":[]}`)
	}))
	t.Cleanup(srv.Close)

	provider, err := New(
		WithTokenCredential(testTokenCredential{}),
		config.WithBaseURL(srv.URL),
		config.WithHTTPClient(srv.Client()),
	)
	require.NoError(t, err)
	_, err = provider.ListModels(t.Context())
	require.NoError(t, err)
}

func TestADTokenEnvironmentAuthentication(t *testing.T) {
	t.Setenv(envAPIKey, "")
	t.Setenv(envADToken, "environment-token")
	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "Bearer environment-token", r.Header.Get("Authorization"))
		assert.Empty(t, r.Header.Get("Api-Key"))
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"object":"list","data":[]}`)
	}))
	t.Cleanup(srv.Close)

	provider, err := New(config.WithBaseURL(srv.URL), config.WithHTTPClient(srv.Client()))
	require.NoError(t, err)
	_, err = provider.ListModels(t.Context())
	require.NoError(t, err)
}

func TestCompatibleOperationsUseAzureOptions(t *testing.T) {
	t.Parallel()
	tests := []azureWireTest{
		{
			name:            "chat",
			path:            "/openai/v1/chat/completions",
			body:            `{"id":"chat","object":"chat.completion","created":1,"model":"chat/deployment","choices":[]}`,
			reasoningEffort: providers.ReasoningEffortNone,
			call: func(ctx context.Context, p *Provider) error {
				_, err := p.Completion(
					ctx,
					providers.CompletionParams{
						Model:           "chat/deployment",
						Messages:        []providers.Message{{Role: providers.RoleUser, Content: "hello"}},
						ReasoningEffort: providers.ReasoningEffortNone,
					},
				)
				if err != nil {
					return fmt.Errorf("complete chat: %w", err)
				}

				return nil
			},
		},
		{
			name: "embedding",
			path: "/openai/v1/embeddings",
			body: `{"object":"list","data":[],"model":"embed/deployment","usage":{"prompt_tokens":0,"total_tokens":0}}`,
			call: func(ctx context.Context, p *Provider) error {
				_, err := p.Embedding(ctx, providers.EmbeddingParams{Model: "embed/deployment", Input: "hello"})
				if err != nil {
					return fmt.Errorf("create embedding: %w", err)
				}

				return nil
			},
		},
		{
			name: "models",
			path: "/openai/v1/models",
			body: `{"object":"list","data":[]}`,
			call: func(ctx context.Context, p *Provider) error {
				_, err := p.ListModels(ctx)
				return err
			},
		},
	}
	runAzureWireTests(t, tests, "wire-key")
}

func runAzureWireTests(t *testing.T, tests []azureWireTest, apiKey string) {
	t.Helper()
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			runAzureWireTest(t, tc, apiKey)
		})
	}
}

func responsesOperationsHandler(t *testing.T, requests chan<- string) http.Handler {
	t.Helper()

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests <- r.Method + " " + r.URL.RequestURI()
		assert.Empty(t, r.Header.Get("Authorization"))
		assert.Equal(t, "response-key", r.Header.Get("Api-Key"))
		assert.Empty(t, r.URL.RawQuery)
		if r.Method == http.MethodPost && r.URL.Path == "/openai/v1/responses" {
			var body map[string]any
			if !assert.NoError(t, json.NewDecoder(r.Body).Decode(&body)) {
				http.Error(w, "invalid request body", http.StatusBadRequest)

				return
			}
			assert.Equal(t, "deployment", body["model"])
		}
		writeResponseOperation(w, r)
	})
}

func writeResponseOperation(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	switch {
	case strings.HasSuffix(r.URL.Path, "/input_items"):
		_, _ = io.WriteString(w, `{"object":"list","data":[],"first_id":"","last_id":"","has_more":false}`)
	case r.Method == http.MethodDelete:
		w.WriteHeader(http.StatusNoContent)
	case strings.Contains(r.Header.Get("Accept"), "text/event-stream") ||
		r.Method == http.MethodPost && r.URL.Path == "/openai/v1/responses":
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	default:
		_, _ = io.WriteString(w, `{"id":"resp/id","object":"response","status":"completed","output":[]}`)
	}
}

func TestResponsesOperationsWireFormat(t *testing.T) {
	t.Parallel()
	requests := make(chan string, 7)
	srv := httptest.NewServer(responsesOperationsHandler(t, requests))
	t.Cleanup(srv.Close)
	provider, err := New(
		config.WithAPIKey("response-key"),
		config.WithBaseURL(srv.URL),
	)
	require.NoError(t, err)

	_, err = provider.RetrieveResponse(t.Context(), "resp_1", responses.ResponseGetParams{})
	require.NoError(t, err)
	require.NoError(t, provider.DeleteResponse(t.Context(), "resp_1"))
	_, err = provider.CancelResponse(t.Context(), "resp_1")
	require.NoError(t, err)
	_, err = provider.ListResponseInputItems(t.Context(), "resp_1", responses.InputItemListParams{})
	require.NoError(t, err)
	events, errs := provider.StreamResponse(t.Context(), responses.ResponseNewParams{Model: "deployment"})
	require.Empty(t, collectResponseStream(t, events, errs))
	events, errs = provider.ReplayResponse(t.Context(), "resp_1", responses.ResponseGetParams{})
	require.Empty(t, collectResponseStream(t, events, errs))

	want := []string{
		"GET /openai/v1/responses/resp_1",
		"DELETE /openai/v1/responses/resp_1",
		"POST /openai/v1/responses/resp_1/cancel",
		"GET /openai/v1/responses/resp_1/input_items",
		"POST /openai/v1/responses",
		"GET /openai/v1/responses/resp_1",
	}
	for _, expected := range want {
		require.Equal(t, expected, <-requests)
	}
}

func TestResponsesTokenCredentialAuthentication(t *testing.T) {
	t.Parallel()

	scopes := make(chan []string, 1)
	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/openai/v1/responses", r.URL.Path)
		assert.Equal(t, "Bearer test-token", r.Header.Get("Authorization"))
		assert.Empty(t, r.Header.Get("Api-Key"))
		assert.Empty(t, r.URL.RawQuery)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"resp_1","object":"response","status":"completed","output":[]}`)
	}))
	t.Cleanup(srv.Close)
	provider, err := New(
		WithTokenCredential(scopeTokenCredential{scopes: scopes}),
		config.WithBaseURL(srv.URL),
		config.WithHTTPClient(srv.Client()),
	)
	require.NoError(t, err)
	_, err = provider.CreateResponse(t.Context(), responses.ResponseNewParams{Model: "deployment"})
	require.NoError(t, err)
	require.Equal(t, []string{v1TokenScope}, <-scopes)
}

func TestResponsesRejectMissingIDs(t *testing.T) {
	t.Parallel()

	provider, err := New(config.WithAPIKey("key"), config.WithBaseURL("https://example.openai.azure.com"))
	require.NoError(t, err)
	_, err = provider.RetrieveResponse(t.Context(), "", responses.ResponseGetParams{})
	require.ErrorContains(t, err, "response ID is required")
	require.ErrorContains(t, provider.DeleteResponse(t.Context(), ""), "response ID is required")
	_, err = provider.CancelResponse(t.Context(), "")
	require.ErrorContains(t, err, "response ID is required")
	_, err = provider.ListResponseInputItems(t.Context(), "", responses.InputItemListParams{})
	require.ErrorContains(t, err, "response ID is required")
	events, errs := provider.ReplayResponse(t.Context(), "", responses.ResponseGetParams{})
	require.Empty(t, collectEvents(events))
	require.ErrorContains(t, <-errs, "response ID is required")
}

func TestRequiredFieldValidationDoesNotSend(t *testing.T) {
	t.Parallel()

	requestSent := make(chan struct{}, 1)
	server := httptest.NewServer(
		http.HandlerFunc(func(http.ResponseWriter, *http.Request) { requestSent <- struct{}{} }),
	)
	t.Cleanup(server.Close)
	provider, err := New(config.WithAPIKey("key"), config.WithBaseURL(server.URL))
	require.NoError(t, err)
	callSpeech := func(params openai.AudioSpeechNewParams) error {
		resp, callErr := provider.Speech(t.Context(), params)
		if resp == nil {
			return callErr
		}

		return stderrors.Join(callErr, resp.Body.Close())
	}

	invalidCalls := []func() error{
		func() error {
			_, err := provider.CreateResponse(t.Context(), responses.ResponseNewParams{})
			return err
		},
		func() error { _, err := provider.GenerateImage(t.Context(), openai.ImageGenerateParams{}); return err },
		func() error {
			_, err := provider.Transcribe(t.Context(), openai.AudioTranscriptionNewParams{})
			return err
		},
		func() error {
			_, err := provider.Transcribe(
				t.Context(),
				openai.AudioTranscriptionNewParams{File: strings.NewReader("audio")},
			)
			return err
		},
		func() error { return callSpeech(openai.AudioSpeechNewParams{}) },
		func() error { return callSpeech(openai.AudioSpeechNewParams{Input: "hello"}) },
		func() error { return callSpeech(openai.AudioSpeechNewParams{Input: "hello", Model: "speech"}) },
	}
	for _, call := range invalidCalls {
		err := call()
		var invalidErr *errors.InvalidRequestError
		require.ErrorAs(t, err, &invalidErr)
	}

	events, streamErrs := provider.StreamResponse(t.Context(), responses.ResponseNewParams{})
	require.Empty(t, collectResponseStream(t, events, streamErrs))
	select {
	case <-requestSent:
		t.Fatal("unexpected request")
	default:
	}
}

func TestOperationCancellationAndErrors(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "images") {
			http.Error(w, `{"error":{"message":"bad image"}}`, http.StatusBadRequest)
			return
		}
		<-r.Context().Done()
	}))
	t.Cleanup(srv.Close)
	provider, err := New(config.WithAPIKey("test-key"), config.WithBaseURL(srv.URL))
	require.NoError(t, err)

	_, err = provider.GenerateImage(t.Context(), openai.ImageGenerateParams{Model: "image", Prompt: "bad"})
	require.Error(t, err)

	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	_, err = provider.CreateResponse(ctx, responses.ResponseNewParams{Model: "response"})
	require.Error(t, err)

	events, errs := provider.StreamResponse(ctx, responses.ResponseNewParams{Model: "response"})
	require.Empty(t, collectEvents(events))
	require.ErrorIs(t, <-errs, context.Canceled)
}

func TestResponseStreamConvertsSDKErrors(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = io.WriteString(w, `{"error":{"message":"slow down"}}`)
	}))
	t.Cleanup(server.Close)
	provider, err := New(config.WithAPIKey("key"), config.WithBaseURL(server.URL))
	require.NoError(t, err)

	events, errs := provider.StreamResponse(t.Context(), responses.ResponseNewParams{Model: "response"})
	require.Empty(t, collectEvents(events))
	streamErr := <-errs
	var rateLimitErr *errors.RateLimitError
	require.ErrorAs(t, streamErr, &rateLimitErr)
}

func TestResponseStreamConvertsCloseErrors(t *testing.T) {
	t.Parallel()

	provider, err := New(config.WithAPIKey("key"), config.WithBaseURL("https://example.test"))
	require.NoError(t, err)

	events, errs := provider.streamResponse(
		t.Context(),
		func() error { return nil },
		func() responseStream {
			return &closeErrorResponseStream{closeErr: errTestCloseResponseStream}
		},
	)
	require.Empty(t, collectEvents(events))

	streamErr := <-errs
	require.ErrorContains(t, streamErr, "close response stream")
	require.ErrorIs(t, streamErr, errors.ErrProvider)
}

func collectResponseStream(
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
		require.Error(t, err)
	}
	return result
}

func collectEvents(events <-chan responses.ResponseStreamEventUnion) []responses.ResponseStreamEventUnion {
	var result []responses.ResponseStreamEventUnion
	for event := range events {
		result = append(result, event)
	}
	return result
}

func TestIntegrationCompletion(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey(providerName) || os.Getenv(envBaseURL) == "" {
		t.Skip("AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT not set")
	}

	provider, err := New()
	require.NoError(t, err)

	resp, err := provider.Completion(t.Context(), providers.CompletionParams{
		Model:    testutil.TestModel(providerName),
		Messages: testutil.SimpleMessages(),
	})
	require.NoError(t, err)
	require.NotEmpty(t, resp.ID)
	require.Len(t, resp.Choices, 1)
	require.NotEmpty(t, resp.Choices[0].Message.Content)
	require.Equal(t, providers.RoleAssistant, resp.Choices[0].Message.Role)
}

func TestListModelsParsesAzureResponseShape(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/openai/v1/models", r.URL.Path)
		assert.Empty(t, r.URL.RawQuery)

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"object": "list",
			"data": [
				{
					"id": "gpt-4o",
					"object": "model",
					"created_at": 1715367600
				},
				{
					"id": "gpt-4o-mini",
					"object": "model",
					"created_at": 1717372800
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

	resp, err := provider.ListModels(t.Context())
	require.NoError(t, err)
	require.Equal(t, "list", resp.Object)
	require.Len(t, resp.Data, 2)

	require.Equal(t, "gpt-4o", resp.Data[0].ID)
	require.Equal(t, "model", resp.Data[0].Object)
	require.Equal(t, int64(1715367600), resp.Data[0].Created)
	require.Empty(t, resp.Data[0].OwnedBy)

	require.Equal(t, "gpt-4o-mini", resp.Data[1].ID)
	require.Equal(t, int64(1717372800), resp.Data[1].Created)
}

func TestListModelsRejectsInvalidCreatedAt(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"object": "list",
			"data": [{
				"id": "gpt-4o",
				"object": "model",
				"created_at": "yesterday"
			}]
		}`))
	}))
	t.Cleanup(srv.Close)

	provider, err := New(
		config.WithAPIKey("test-key"),
		config.WithBaseURL(srv.URL),
	)
	require.NoError(t, err)

	_, err = provider.ListModels(t.Context())
	require.ErrorContains(t, err, `decode created_at for model "gpt-4o"`)
}
