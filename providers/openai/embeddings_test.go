package openai

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

func TestEmbeddingWireInputs(t *testing.T) {
	t.Parallel()

	dimensions := 256
	for _, testCase := range []struct {
		name   string
		params providers.EmbeddingParams
		want   string
	}{
		{name: "string", params: providers.EmbeddingParams{Model: "text-embedding-3-small", Input: "Hello"}, want: `{"input":"Hello","model":"text-embedding-3-small"}`},
		{name: "string array", params: providers.EmbeddingParams{Model: "text-embedding-3-small", Input: []string{"Hello", "World"}}, want: `{"input":["Hello","World"],"model":"text-embedding-3-small"}`},
		{name: "native token array", params: providers.EmbeddingParams{Model: "text-embedding-3-small", Input: []int{1, 2}}, want: `{"input":[1,2],"model":"text-embedding-3-small"}`},
		{name: "SDK token array", params: providers.EmbeddingParams{Model: "text-embedding-3-small", Input: []int64{1, 2}}, want: `{"input":[1,2],"model":"text-embedding-3-small"}`},
		{name: "native token arrays", params: providers.EmbeddingParams{Model: "text-embedding-3-small", Input: [][]int{{1, 2}, {3}}}, want: `{"input":[[1,2],[3]],"model":"text-embedding-3-small"}`},
		{name: "SDK token arrays", params: providers.EmbeddingParams{Model: "text-embedding-3-small", Input: [][]int64{{1, 2}, {3}}}, want: `{"input":[[1,2],[3]],"model":"text-embedding-3-small"}`},
		{
			name: "optional parameters",
			params: providers.EmbeddingParams{
				Model:          "text-embedding-3-small",
				Input:          "Hello",
				EncodingFormat: "float",
				Dimensions:     &dimensions,
				User:           "test-user",
			},
			want: `{"dimensions":256,"encoding_format":"float","input":"Hello","model":"text-embedding-3-small","user":"test-user"}`,
		},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			var requestBody []byte
			var requestPath string
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
				requestPath = request.URL.Path
				requestBody, _ = io.ReadAll(request.Body)
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `{
					"object":"list",
					"data":[{"object":"embedding","embedding":[0.25,-0.5],"index":7,"future":"ignored"}],
					"model":"text-embedding-3-small",
					"usage":{"prompt_tokens":2,"total_tokens":2},
					"future":"ignored"
				}`)
			}))
			t.Cleanup(server.Close)

			provider, err := NewCompatible(CompatibleConfig{
				Name:           providerName,
				DefaultAPIKey:  "test-key",
				DefaultBaseURL: server.URL,
			})
			require.NoError(t, err)

			response, err := provider.Embedding(t.Context(), testCase.params)
			require.NoError(t, err)
			require.Equal(t, "/embeddings", requestPath)
			require.JSONEq(t, testCase.want, string(requestBody))
			require.Equal(t, "list", response.Object)
			require.Equal(t, "text-embedding-3-small", response.Model)
			require.Equal(t, []providers.EmbeddingData{{
				Object:    "embedding",
				Embedding: []float64{0.25, -0.5},
				Index:     7,
			}}, response.Data)
			require.Equal(t, &providers.EmbeddingUsage{PromptTokens: 2, TotalTokens: 2}, response.Usage)
		})
	}
}

func TestEmbeddingRejectsUnsupportedInputBeforeRequest(t *testing.T) {
	t.Parallel()

	var requested atomic.Bool
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		requested.Store(true)
	}))
	t.Cleanup(server.Close)

	provider, err := NewCompatible(CompatibleConfig{
		Name:           providerName,
		DefaultAPIKey:  "test-key",
		DefaultBaseURL: server.URL,
	})
	require.NoError(t, err)

	_, err = provider.Embedding(t.Context(), providers.EmbeddingParams{
		Model: "text-embedding-3-small",
		Input: map[string]string{"unsupported": "object"},
	})
	require.ErrorIs(t, err, errors.ErrInvalidRequest)
	require.False(t, requested.Load())
}

func TestEmbeddingMapsAPIError(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = io.WriteString(w, `{"error":{"message":"invalid key","type":"invalid_request_error"}}`)
	}))
	t.Cleanup(server.Close)

	provider, err := NewCompatible(CompatibleConfig{
		Name:           providerName,
		DefaultAPIKey:  "test-key",
		DefaultBaseURL: server.URL,
	})
	require.NoError(t, err)

	_, err = provider.Embedding(t.Context(), providers.EmbeddingParams{
		Model: "text-embedding-3-small",
		Input: "Hello",
	})
	require.ErrorIs(t, err, errors.ErrAuthentication)
}

func TestEmbeddingHonorsCancellation(t *testing.T) {
	t.Parallel()

	var requested atomic.Bool
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		requested.Store(true)
	}))
	t.Cleanup(server.Close)

	provider, err := NewCompatible(CompatibleConfig{
		Name:           providerName,
		DefaultAPIKey:  "test-key",
		DefaultBaseURL: server.URL,
	})
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	_, err = provider.Embedding(ctx, providers.EmbeddingParams{
		Model: "text-embedding-3-small",
		Input: "Hello",
	})
	require.ErrorIs(t, err, context.Canceled)
	require.False(t, requested.Load())
}

func TestEmbeddingHonorsTimeout(t *testing.T) {
	t.Parallel()

	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		<-release
	}))
	t.Cleanup(server.Close)
	t.Cleanup(func() { close(release) })

	provider, err := NewCompatible(CompatibleConfig{
		Name:           providerName,
		DefaultAPIKey:  "test-key",
		DefaultBaseURL: server.URL,
	})
	require.NoError(t, err)

	ctx, cancel := context.WithTimeout(t.Context(), 100*time.Millisecond)
	t.Cleanup(cancel)
	_, err = provider.Embedding(ctx, providers.EmbeddingParams{
		Model: "text-embedding-3-small",
		Input: "Hello",
	})
	require.ErrorIs(t, err, context.DeadlineExceeded)
}
