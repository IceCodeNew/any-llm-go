package anthropic

import (
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/providers"
)

func TestConvertContentPDFAndCacheControl(t *testing.T) {
	t.Parallel()

	message := providers.Message{Role: providers.RoleUser, Content: []providers.ContentPart{
		{Type: "text", Text: "Read this", CacheControl: &providers.CacheControl{Type: "ephemeral", TTL: "1h"}},
		{Type: "file", File: &providers.File{FileData: "data:application/pdf;base64,JVBERg=="}},
		{Type: "file", File: &providers.File{FileData: "https://example.com/report.pdf"}},
	}}
	blocks, err := convertContent(message)
	require.NoError(t, err)
	require.Len(t, blocks, 3)
	require.Equal(t, anthropic.CacheControlEphemeralTTLTTL1h, blocks[0].OfText.CacheControl.TTL)
	require.Equal(t, "JVBERg==", blocks[1].OfDocument.Source.OfBase64.Data)
	require.Equal(t, "https://example.com/report.pdf", blocks[2].OfDocument.Source.OfURL.URL)
}

func TestConvertContentRejectsInvalidPDFAndCache(t *testing.T) {
	t.Parallel()

	tests := []providers.ContentPart{
		{Type: "file", File: &providers.File{FileData: "data:text/plain;base64,QQ=="}},
		{Type: "file", File: &providers.File{FileData: "http://example.com/a.pdf"}},
		{Type: "file"},
		{Type: "text", Text: "x", CacheControl: &providers.CacheControl{Type: "forever"}},
		{Type: "text", Text: "x", CacheControl: &providers.CacheControl{Type: "ephemeral", TTL: "2h"}},
	}
	for _, part := range tests {
		_, err := convertContent(providers.Message{Content: []providers.ContentPart{part}})
		require.Error(t, err)
	}
}

func TestThinkingBlocksJSONRoundTripPreservesOrderAndOpaqueData(t *testing.T) {
	t.Parallel()

	response, err := convertResponse(&anthropic.Message{Content: []anthropic.ContentBlockUnion{
		{Type: blockTypeThinking, Thinking: "first", Signature: "sig-1"},
		{Type: blockTypeRedactedThinking, Data: "opaque-secret"},
		{Type: blockTypeThinking, Thinking: "second", Signature: "sig-2"},
	}})
	require.NoError(t, err)
	message := response.Choices[0].Message
	require.Equal(t, "firstsecond", message.Reasoning.Content)

	encoded, err := json.Marshal(message)
	require.NoError(t, err)
	require.NotContains(t, string(encoded), `"index"`)
	var decoded providers.Message
	require.NoError(t, json.Unmarshal(encoded, &decoded))
	replayed, err := convertAssistantMessage(decoded)
	require.NoError(t, err)
	require.Len(t, replayed.Content, 3)
	require.Equal(t, "sig-1", replayed.Content[0].OfThinking.Signature)
	require.Equal(t, "opaque-secret", replayed.Content[1].OfRedactedThinking.Data)
	require.Equal(t, "sig-2", replayed.Content[2].OfThinking.Signature)
}

func TestConvertMessagesMergesParallelToolResults(t *testing.T) {
	t.Parallel()

	messages, _, err := convertMessages([]providers.Message{
		{Role: providers.RoleTool, ToolCallID: "one", Content: "1"},
		{Role: providers.RoleTool, ToolCallID: "two", Content: []providers.ContentPart{{Type: "text", Text: "2"}}},
	})
	require.NoError(t, err)
	require.Len(t, messages, 1)
	require.Len(t, messages[0].Content, 2)
	require.Equal(t, "one", messages[0].Content[0].OfToolResult.ToolUseID)
	require.Equal(t, "two", messages[0].Content[1].OfToolResult.ToolUseID)
}

func TestConvertMessagesDoesNotMergeToolResultsAcrossInterveningMessages(t *testing.T) {
	t.Parallel()

	for _, intervening := range []providers.Message{
		{Role: providers.RoleSystem, Content: "new instructions"},
		{Role: providers.RoleAssistant, Content: "intervening response"},
		{Role: providers.RoleUser, Content: "intervening request"},
	} {
		messages, _, err := convertMessages([]providers.Message{
			{Role: providers.RoleTool, ToolCallID: "one", Content: "1"},
			intervening,
			{Role: providers.RoleTool, ToolCallID: "two", Content: "2"},
		})
		require.NoError(t, err)
		expected := 3
		if intervening.Role == providers.RoleSystem {
			expected = 2
		}
		require.Len(t, messages, expected)
		require.Len(t, messages[len(messages)-1].Content, 1)
		require.Equal(t, "two", messages[len(messages)-1].Content[0].OfToolResult.ToolUseID)
	}
}

func TestConvertAssistantMessageThinkingMetadataFallbackAndErrors(t *testing.T) {
	t.Parallel()

	reasoning := &providers.Reasoning{Content: "thought", Signature: "signed"}
	for _, extra := range []map[string]providers.ProviderData{
		nil,
		{providerName: {"other": true}},
	} {
		message, err := convertAssistantMessage(
			providers.Message{Role: providers.RoleAssistant, Reasoning: reasoning, Extra: extra},
		)
		require.NoError(t, err)
		require.Len(t, message.Content, 1)
		require.Equal(t, "signed", message.Content[0].OfThinking.Signature)
	}

	for _, blocks := range []any{"invalid", []any{"invalid"}, []any{map[string]any{"type": blockTypeThinking}}} {
		_, err := convertAssistantMessage(providers.Message{
			Role:  providers.RoleAssistant,
			Extra: map[string]providers.ProviderData{providerName: {"thinking_blocks": blocks}},
		})
		require.ErrorContains(t, err, "thinking")
	}
}

func TestStreamThinkingMetadataIsSnapshot(t *testing.T) {
	t.Parallel()

	state := newStreamState()
	state.handleContentBlockStart(anthropic.ContentBlockStartEvent{
		ContentBlock: anthropic.ContentBlockStartEventContentBlockUnion{Type: blockTypeThinking},
	})
	first := state.handleThinkingDelta("first")
	second := state.handleThinkingDelta(" second")
	firstBlocks, ok := first.Choices[0].Delta.Extra[providerName]["thinking_blocks"].([]providers.ProviderData)
	require.True(t, ok)
	secondBlocks, ok := second.Choices[0].Delta.Extra[providerName]["thinking_blocks"].([]providers.ProviderData)
	require.True(t, ok)
	require.Equal(t, "first", firstBlocks[0]["thinking"])
	require.Equal(t, "first second", secondBlocks[0]["thinking"])
}

func TestConvertBatchMapsTerminalStatesAndCounts(t *testing.T) {
	t.Parallel()

	ended := time.Unix(20, 0)
	batch := convertBatch(&anthropic.MessageBatch{
		ID: "batch-1", CreatedAt: time.Unix(10, 0), EndedAt: ended,
		ProcessingStatus: anthropic.MessageBatchProcessingStatusEnded,
		RequestCounts:    anthropic.MessageBatchRequestCounts{Succeeded: 2, Errored: 1},
	})
	require.Equal(t, providers.BatchStatusCompleted, batch.Status)
	require.Equal(t, &[]int64{20}[0], batch.CompletedAt)
	require.Equal(t, &providers.BatchRequestCounts{Completed: 2, Failed: 1, Total: 3}, batch.RequestCounts)
}

func TestConvertResponsePreservesRefusalAndReasoningUsage(t *testing.T) {
	t.Parallel()

	var message anthropic.Message
	require.NoError(t, json.Unmarshal([]byte(`{
		"id":"msg_1","type":"message","role":"assistant","model":"claude-test",
		"content":[],"stop_reason":"refusal","stop_sequence":null,
		"stop_details":{"type":"refusal","category":"general_harms","explanation":"declined"},
		"usage":{"input_tokens":2,"output_tokens":7,"cache_creation_input_tokens":3,
			"cache_read_input_tokens":5,"output_tokens_details":{"thinking_tokens":4}}
	}`), &message))
	response, err := convertResponse(&message)
	require.NoError(t, err)
	require.Equal(t, providers.FinishReasonContentFilter, response.Choices[0].FinishReason)
	require.Equal(t, 4, response.Usage.ReasoningTokens)
	require.Equal(t, 10, response.Usage.PromptTokens)
	stopDetails, ok := response.Choices[0].Message.Extra[providerName]["stop_details"].(providers.ProviderData)
	require.True(t, ok)
	require.Equal(t, "general_harms", stopDetails["category"])
}

func TestStreamUsageUsesCumulativeDeltaAndAppendsSignature(t *testing.T) {
	t.Parallel()

	state := newStreamState()
	state.handleMessageStart(anthropic.MessageStartEvent{
		Message: anthropic.Message{Usage: anthropic.Usage{InputTokens: 1}},
	})
	state.handleContentBlockStart(anthropic.ContentBlockStartEvent{
		ContentBlock: anthropic.ContentBlockStartEventContentBlockUnion{Type: blockTypeThinking},
	})
	for _, signature := range []string{"part-1", "part-2"} {
		var delta anthropic.ContentBlockDeltaEvent
		require.NoError(t, json.Unmarshal([]byte(fmt.Sprintf(
			`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":%q}}`,
			signature,
		)), &delta))
		state.handleContentBlockDelta(delta)
	}

	var event anthropic.MessageDeltaEvent
	require.NoError(t, json.Unmarshal([]byte(`{
		"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},
		"usage":{"input_tokens":2,"cache_creation_input_tokens":3,"cache_read_input_tokens":5,
			"output_tokens":7,"output_tokens_details":{"thinking_tokens":4}}
	}`), &event))
	chunk := state.handleMessageDelta(event)
	require.Equal(t, 10, chunk.Usage.PromptTokens)
	require.Equal(t, 4, chunk.Usage.ReasoningTokens)
	require.Equal(t, "part-1part-2", state.thinkingBlocks[0]["signature"])
}
