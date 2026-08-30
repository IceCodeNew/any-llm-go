package platform

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
	"github.com/mozilla-ai/any-llm-go/sdk"
)

const validPlatformKey = "ANY.v1.12345678.abcdef01-YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY="

type roundTripFunc func(*http.Request) (*http.Response, error)

func (fn roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return fn(request)
}

type streamingProviderStub struct {
	completionStream func(
		context.Context,
		providers.CompletionParams,
	) (<-chan providers.ChatCompletionChunk, <-chan error)
}

func (*streamingProviderStub) Name() string {
	return "streaming-stub"
}

func (*streamingProviderStub) Completion(
	context.Context,
	providers.CompletionParams,
) (*providers.ChatCompletion, error) {
	return new(providers.ChatCompletion), nil
}

func (s *streamingProviderStub) CompletionStream(
	ctx context.Context,
	params providers.CompletionParams,
) (<-chan providers.ChatCompletionChunk, <-chan error) {
	return s.completionStream(ctx, params)
}

func TestNew(t *testing.T) {
	t.Run("returns error when API key is missing", func(t *testing.T) {
		t.Setenv("ANY_LLM_KEY", "")

		provider, err := New()

		require.Nil(t, provider)
		require.Error(t, err)
		require.ErrorIs(t, err, errors.ErrMissingAPIKey)
	})

	t.Run("creates provider with API key from env", func(t *testing.T) {
		t.Setenv("ANY_LLM_KEY", "ANY.v1.test.fingerprint-dGVzdHByaXZhdGVrZXkxMjM0NTY3ODkwMTI=")

		provider, err := New()

		require.NoError(t, err)
		require.NotNil(t, provider)
		require.Equal(t, "platform", provider.Name())
	})

	t.Run("creates provider with explicit API key", func(t *testing.T) {
		t.Setenv("ANY_LLM_KEY", "")

		provider, err := New(config.WithAPIKey("ANY.v1.test.fingerprint-dGVzdHByaXZhdGVrZXkxMjM0NTY3ODkwMTI="))

		require.NoError(t, err)
		require.NotNil(t, provider)
	})
}

func TestProvider_Name(t *testing.T) {
	t.Setenv("ANY_LLM_KEY", "ANY.v1.test.fingerprint-dGVzdHByaXZhdGVrZXkxMjM0NTY3ODkwMTI=")

	provider, err := New()
	require.NoError(t, err)

	require.Equal(t, "platform", provider.Name())
}

func TestProvider_Capabilities(t *testing.T) {
	t.Setenv("ANY_LLM_KEY", "ANY.v1.test.fingerprint-dGVzdHByaXZhdGVrZXkxMjM0NTY3ODkwMTI=")

	provider, err := New()
	require.NoError(t, err)

	caps := provider.Capabilities()

	require.True(t, caps.Completion)
	require.True(t, caps.CompletionReasoning)
	require.True(t, caps.CompletionStreaming)
	require.True(t, caps.CompletionTools)
	require.True(t, caps.Embedding)
}

func TestParseModelString(t *testing.T) {
	t.Parallel()

	tests := []struct {
		input        string
		wantProvider string
		wantModel    string
	}{
		{"openai:gpt-4o-mini", "openai", "gpt-4o-mini"},
		{"anthropic:claude-3-5-haiku-latest", "anthropic", "claude-3-5-haiku-latest"},
		{"gpt-4o-mini", "", "gpt-4o-mini"},
		{"provider:model:with:colons", "provider", "model:with:colons"},
	}

	for _, tc := range tests {
		t.Run(tc.input, func(t *testing.T) {
			t.Parallel()

			provider, model := parseModelString(tc.input)
			require.Equal(t, tc.wantProvider, provider)
			require.Equal(t, tc.wantModel, model)
		})
	}
}

func TestUserAgent(t *testing.T) {
	t.Parallel()

	ua := userAgent()

	// Verify RFC 9110 product/version format: "any-llm/X.Y.Z go/X.Y.Z".
	parts := strings.SplitN(ua, " ", 2)
	require.Len(t, parts, 2, "expected sdk/version and language/version pair")

	require.True(t, strings.HasPrefix(parts[0], sdk.Name+"/"), "first token should start with library name")
	require.NotContains(t, parts[0], "/v", "version should not have v prefix")

	require.True(t, strings.HasPrefix(parts[1], "go/"), "second token should be the go runtime product")
	require.Contains(t, parts[1], strings.TrimPrefix(runtime.Version(), "go"), "should contain the go runtime version")
}

func TestCompletionDoesNotMutateParams(t *testing.T) {
	t.Parallel()

	// Test that even when the provider fails (platform auth will fail), original params are unchanged.
	provider, err := New(config.WithAPIKey("ANY.v1.test.fingerprint-dGVzdHByaXZhdGVrZXkxMjM0NTY3ODkwMTI="))
	require.NoError(t, err)

	originalModel := "openai:gpt-4"
	originalStream := false
	originalStreamOptions := &providers.StreamOptions{IncludeUsage: false}

	params := providers.CompletionParams{
		Model: originalModel,
		Messages: []providers.Message{
			{Role: providers.RoleUser, Content: "Hello"},
		},
		Stream:        originalStream,
		StreamOptions: originalStreamOptions,
	}

	// Call will fail (platform auth will fail), but params should remain unchanged.
	_, _ = provider.Completion(context.Background(), params)

	// Verify params were not mutated.
	require.Equal(t, originalModel, params.Model)
	require.Equal(t, originalStream, params.Stream)
	require.Equal(t, false, params.StreamOptions.IncludeUsage)
}

func TestCompletionStreamDoesNotMutateParams(t *testing.T) {
	t.Parallel()

	provider, err := New(config.WithAPIKey("ANY.v1.test.fingerprint-dGVzdHByaXZhdGVrZXkxMjM0NTY3ODkwMTI="))
	require.NoError(t, err)

	originalModel := "openai:gpt-4"
	originalStream := false
	originalStreamOptions := &providers.StreamOptions{IncludeUsage: false}

	params := providers.CompletionParams{
		Model: originalModel,
		Messages: []providers.Message{
			{Role: providers.RoleUser, Content: "Hello"},
		},
		Stream:        originalStream,
		StreamOptions: originalStreamOptions,
	}

	// Call will fail (platform auth will fail), but params should remain unchanged.
	chunks, errs := provider.CompletionStream(context.Background(), params)

	// Drain channels.
	for range chunks {
	}
	<-errs

	// Verify params were not mutated.
	require.Equal(t, originalModel, params.Model)
	require.Equal(t, originalStream, params.Stream)
	require.Equal(t, false, params.StreamOptions.IncludeUsage)
}

func TestCompletionStreamDrainsUpstreamAfterCallerCancellation(t *testing.T) {
	t.Parallel()

	firstChunkSent := make(chan struct{})
	producerDone := make(chan struct{})
	underlying := new(streamingProviderStub)
	underlying.completionStream = func(
		context.Context,
		providers.CompletionParams,
	) (<-chan providers.ChatCompletionChunk, <-chan error) {
		chunks := make(chan providers.ChatCompletionChunk)
		errs := make(chan error)

		go func() {
			defer close(chunks)
			defer close(errs)
			defer close(producerDone)

			var chunk providers.ChatCompletionChunk
			chunks <- chunk

			close(firstChunkSent)

			chunks <- chunk
		}()

		return chunks, errs
	}

	ready := make(chan struct{})
	close(ready)

	state := new(providerState)
	state.provider = underlying
	state.ready = ready
	provider, err := New(config.WithAPIKey(validPlatformKey))
	require.NoError(t, err)

	provider.providers["openai"] = state

	ctx, cancel := context.WithCancel(t.Context())

	var params providers.CompletionParams

	params.Model = "openai:model"
	chunks, errs := provider.CompletionStream(ctx, params)

	<-firstChunkSent
	cancel()

	require.ErrorIs(t, <-errs, context.Canceled)

	for chunk := range chunks {
		t.Fatalf("received unexpected chunk after cancellation: %#v", chunk)
	}

	select {
	case <-producerDone:
	case <-time.After(time.Second):
		t.Fatal("upstream producer remained blocked after caller cancellation")
	}
}

func TestInitializeProviderAllowsDifferentProvidersConcurrently(t *testing.T) {
	started := make(chan struct{}, 2)
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		started <- struct{}{}
		<-release
		response.WriteHeader(http.StatusUnauthorized)
	}))
	defer server.Close()

	t.Setenv(envPlatformURL, server.URL)
	provider, err := New(config.WithAPIKey(validPlatformKey))
	require.NoError(t, err)

	results := make(chan error, 2)
	for _, name := range []string{"openai", "anthropic"} {
		go func() {
			_, initErr := provider.initializeProvider(t.Context(), name)
			results <- initErr
		}()
	}

	for range 2 {
		select {
		case <-started:
		case <-time.After(time.Second):
			t.Fatal("different providers did not initialize concurrently")
		}
	}
	close(release)
	for range 2 {
		require.Error(t, <-results)
	}
}

func TestInitializeProviderWaiterCanBeCanceled(t *testing.T) {
	started := make(chan struct{}, 1)
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		started <- struct{}{}
		<-release
		response.WriteHeader(http.StatusUnauthorized)
	}))
	defer server.Close()

	t.Setenv(envPlatformURL, server.URL)
	provider, err := New(config.WithAPIKey(validPlatformKey))
	require.NoError(t, err)

	firstResult := make(chan error, 1)
	go func() {
		_, initErr := provider.initializeProvider(t.Context(), "openai")
		firstResult <- initErr
	}()
	<-started

	waiterCtx, cancelWaiter := context.WithCancel(t.Context())
	cancelWaiter()
	_, err = provider.initializeProvider(waiterCtx, "openai")
	require.ErrorIs(t, err, context.Canceled)

	close(release)
	require.Error(t, <-firstResult)
}

func TestInitializeProviderSurvivesInitiatorCancellation(t *testing.T) {
	t.Parallel()

	provider, err := New(config.WithAPIKey(validPlatformKey))
	require.NoError(t, err)

	requestStarted := make(chan *http.Request, 1)
	release := make(chan struct{})
	provider.httpClient.Timeout = time.Minute
	provider.httpClient.Transport = roundTripFunc(func(request *http.Request) (*http.Response, error) {
		requestStarted <- request

		<-release

		response := new(http.Response)
		response.StatusCode = http.StatusUnauthorized
		response.Header = make(http.Header)
		response.Body = http.NoBody
		response.Request = request

		return response, nil
	})

	initiatorCtx, cancelInitiator := context.WithCancel(t.Context())
	initiatorResult := make(chan error, 1)

	go func() {
		_, initErr := provider.initializeProvider(initiatorCtx, "openai")
		initiatorResult <- initErr
	}()

	request := <-requestStarted

	provider.mu.Lock()
	state := provider.providers["openai"]
	provider.mu.Unlock()
	require.NotNil(t, state)

	cancelInitiator()
	require.ErrorIs(t, <-initiatorResult, context.Canceled)
	require.NoError(t, request.Context().Err())
	close(release)
	<-state.ready
	require.Error(t, state.err)
}

func TestInitializeProviderRetriesAfterFailure(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		response.WriteHeader(http.StatusUnauthorized)
	}))
	defer server.Close()

	t.Setenv(envPlatformURL, server.URL)
	provider, err := New(config.WithAPIKey(validPlatformKey))
	require.NoError(t, err)

	_, err = provider.initializeProvider(t.Context(), "openai")
	require.Error(t, err)
	_, err = provider.initializeProvider(t.Context(), "openai")
	require.Error(t, err)
	require.Equal(t, int32(2), requests.Load())
}

// Integration tests - require actual platform connection and ANY_LLM_KEY

func TestIntegrationOpenAICompletion(t *testing.T) {
	t.Parallel()

	if os.Getenv("ANY_LLM_KEY") == "" {
		t.Skip("ANY_LLM_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()

	// Request a completion through the platform, which will:
	// 1. Authenticate with the platform using ANY_LLM_KEY
	// 2. Get the decrypted OpenAI API key from the platform
	// 3. Create an OpenAI provider and delegate the request
	// 4. Report usage metrics back to the platform
	response, err := provider.Completion(ctx, providers.CompletionParams{
		Model: "openai:gpt-4o-mini",
		Messages: []providers.Message{
			{Role: providers.RoleUser, Content: "Say 'hello' and nothing else."},
		},
	})
	require.NoError(t, err)

	require.NotNil(t, response)
	require.NotEmpty(t, response.Choices)
	require.NotEmpty(t, response.Choices[0].Message.Content)

	content, ok := response.Choices[0].Message.Content.(string)
	require.True(t, ok, "Content should be a string")
	require.True(t, strings.Contains(strings.ToLower(content), "hello"))

	// Verify usage was tracked
	require.NotNil(t, response.Usage)
	require.Greater(t, response.Usage.TotalTokens, 0)

	t.Logf("Response: %s", content)
	t.Logf("Tokens used: %d", response.Usage.TotalTokens)

	// Wait a bit for the usage event goroutine to complete
	time.Sleep(2 * time.Second)
}

func TestIntegrationOpenAIStreaming(t *testing.T) {
	t.Parallel()

	if os.Getenv("ANY_LLM_KEY") == "" {
		t.Skip("ANY_LLM_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()

	chunks, errs := provider.CompletionStream(ctx, providers.CompletionParams{
		Model: "openai:gpt-4o-mini",
		Messages: []providers.Message{
			{Role: providers.RoleUser, Content: "Count from 1 to 5, one number per line."},
		},
		Stream: true,
	})

	var content strings.Builder
	chunkCount := 0

	for chunk := range chunks {
		chunkCount++
		if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
			content.WriteString(chunk.Choices[0].Delta.Content)
		}
	}

	err = <-errs
	require.NoError(t, err)

	require.Greater(t, chunkCount, 0, "Should have received chunks")
	require.NotEmpty(t, content.String(), "Should have received content")

	t.Logf("Received %d chunks", chunkCount)
	t.Logf("Content: %s", content.String())

	// Wait a bit for the usage event goroutine to complete
	time.Sleep(2 * time.Second)
}
