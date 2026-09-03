package deepseek

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/internal/testutil"
	"github.com/mozilla-ai/any-llm-go/providers"
)

func TestCompletionPreservesReasoningResponse(t *testing.T) {
	t.Parallel()

	serverURL, _ := deepSeekCompletionServer(t, `{
		"id":"chatcmpl-test","object":"chat.completion","created":1700000000,
		"model":"deepseek-v4-pro","choices":[{"index":0,"message":{
			"role":"assistant","content":"answer","reasoning_content":"","future_field":true
		},"finish_reason":"insufficient_system_resource"}]
	}`)
	provider, err := New(
		config.WithAPIKey("test-key"),
		config.WithBaseURL(serverURL),
	)
	require.NoError(t, err)

	completion, err := provider.Completion(t.Context(), providers.CompletionParams{
		Model:    "deepseek-v4-pro",
		Messages: testutil.SimpleMessages(),
	})
	require.NoError(t, err)
	require.Equal(t, "answer", completion.Choices[0].Message.Content)
	require.Equal(t, "insufficient_system_resource", completion.Choices[0].FinishReason)
	require.NotNil(t, completion.Choices[0].Message.Reasoning)
	require.Empty(t, completion.Choices[0].Message.Reasoning.Content)
}

func TestCompletionPreservesNullableContent(t *testing.T) {
	t.Parallel()

	serverURL, _ := deepSeekCompletionServer(t, `{
		"id":"chatcmpl-test","object":"chat.completion","created":1700000000,
		"model":"deepseek-v4-pro","choices":[{"index":0,"message":{
			"role":"assistant","content":null,"reasoning_content":"reasoning"
		},"finish_reason":"stop"}]
	}`)
	provider, err := New(
		config.WithAPIKey("test-key"),
		config.WithBaseURL(serverURL),
	)
	require.NoError(t, err)

	completion, err := provider.Completion(t.Context(), providers.CompletionParams{
		Model:    "deepseek-v4-pro",
		Messages: testutil.SimpleMessages(),
	})
	require.NoError(t, err)
	require.Nil(t, completion.Choices[0].Message.Content)
}

func TestCompletionRejectsMalformedContent(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		message string
		want    string
	}{
		{name: "missing answer", message: `"reasoning_content":"reasoning"`, want: "missing required field"},
		{name: "invalid answer", message: `"content":{}`, want: "cannot unmarshal object"},
		{
			name:    "invalid reasoning",
			message: `"content":"answer","reasoning_content":{}`,
			want:    "cannot unmarshal object",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			serverURL, _ := deepSeekCompletionServer(t, fmt.Sprintf(`{
				"id":"chatcmpl-test","object":"chat.completion","created":1700000000,
				"model":"deepseek-v4-pro","choices":[{"index":0,"message":{
					"role":"assistant",%s
				},"finish_reason":"stop"}]
			}`, test.message))
			provider, err := New(
				config.WithAPIKey("test-key"),
				config.WithBaseURL(serverURL),
			)
			require.NoError(t, err)

			_, err = provider.Completion(t.Context(), providers.CompletionParams{
				Model:    "deepseek-v4-pro",
				Messages: testutil.SimpleMessages(),
			})
			require.ErrorContains(t, err, test.want)
		})
	}
}
