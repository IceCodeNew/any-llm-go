package anthropic

import (
	"maps"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/mozilla-ai/any-llm-go/providers"
)

// streamState tracks accumulated state during streaming.
// Note: Only accessed from a single goroutine, so no synchronization needed.
type streamState struct {
	messageID          string
	model              string
	content            strings.Builder
	reasoning          strings.Builder
	toolCalls          []providers.ToolCall
	currentToolIdx     int
	inputUsage         int64
	cacheCreationUsage int64
	cacheReadUsage     int64
	thinkingBlocks     []providers.ProviderData
	currentThinkingIdx int
}

// newStreamState creates a new stream state with default values.
func newStreamState() *streamState {
	return &streamState{
		currentToolIdx:     -1,
		currentThinkingIdx: -1,
	}
}

// chunk creates a ChatCompletionChunk with the given delta.
func (s *streamState) chunk(delta providers.ChunkDelta) providers.ChatCompletionChunk {
	return providers.ChatCompletionChunk{
		ID:     s.messageID,
		Object: "chat.completion.chunk",
		Model:  s.model,
		Choices: []providers.ChunkChoice{{
			Index: 0,
			Delta: delta,
		}},
	}
}

// handleContentBlockDelta processes a content_block_delta event and returns a chunk if applicable.
func (s *streamState) handleContentBlockDelta(event anthropic.ContentBlockDeltaEvent) *providers.ChatCompletionChunk {
	switch event.Delta.Type {
	case deltaTypeText:
		return s.handleTextDelta(event.Delta.Text)
	case deltaTypeThinking:
		return s.handleThinkingDelta(event.Delta.Thinking)
	case deltaTypeSignature:
		if s.currentThinkingIdx >= 0 {
			previous, _ := s.thinkingBlocks[s.currentThinkingIdx]["signature"].(string)
			s.thinkingBlocks[s.currentThinkingIdx]["signature"] = previous + event.Delta.Signature
		}
		chunk := s.chunk(providers.ChunkDelta{Extra: s.thinkingExtra()})
		return &chunk
	case deltaTypeInputJSON:
		return s.handleInputJSONDelta(event.Delta.PartialJSON)
	default:
		return nil
	}
}

// handleContentBlockStart processes a content_block_start event.
func (s *streamState) handleContentBlockStart(event anthropic.ContentBlockStartEvent) *providers.ChatCompletionChunk {
	switch event.ContentBlock.Type {
	case blockTypeThinking:
		s.thinkingBlocks = append(s.thinkingBlocks, providers.ProviderData{
			"type":      blockTypeThinking,
			"thinking":  event.ContentBlock.Thinking,
			"signature": event.ContentBlock.Signature,
		})
		s.currentThinkingIdx = len(s.thinkingBlocks) - 1
	case blockTypeRedactedThinking:
		s.thinkingBlocks = append(s.thinkingBlocks, providers.ProviderData{
			"type": blockTypeRedactedThinking, "data": event.ContentBlock.Data,
		})
		s.currentThinkingIdx = -1
		chunk := s.chunk(providers.ChunkDelta{Extra: s.thinkingExtra()})
		return &chunk
	case blockTypeToolUse:
		s.currentToolIdx++
		tc := providers.ToolCall{
			ID:   event.ContentBlock.ID,
			Type: "function",
			Function: providers.FunctionCall{
				Name: event.ContentBlock.Name,
			},
		}
		s.toolCalls = append(s.toolCalls, tc)
	}
	return nil
}

// handleInputJSONDelta processes a tool input JSON delta and returns a chunk if applicable.
func (s *streamState) handleInputJSONDelta(partialJSON string) *providers.ChatCompletionChunk {
	if s.currentToolIdx < 0 || s.currentToolIdx >= len(s.toolCalls) {
		return nil
	}

	s.toolCalls[s.currentToolIdx].Function.Arguments += partialJSON
	deltaToolCall := s.toolCalls[s.currentToolIdx]
	deltaToolCall.Function.Arguments = partialJSON
	chunk := s.chunk(providers.ChunkDelta{
		ToolCalls: []providers.ToolCall{deltaToolCall},
	})
	return &chunk
}

// handleMessageDelta processes a message_delta event and returns the final chunk.
func (s *streamState) handleMessageDelta(event anthropic.MessageDeltaEvent) providers.ChatCompletionChunk {
	if event.Usage.JSON.InputTokens.Valid() {
		s.inputUsage = event.Usage.InputTokens
	}
	if event.Usage.JSON.CacheCreationInputTokens.Valid() {
		s.cacheCreationUsage = event.Usage.CacheCreationInputTokens
	}
	if event.Usage.JSON.CacheReadInputTokens.Valid() {
		s.cacheReadUsage = event.Usage.CacheReadInputTokens
	}
	finishReason := convertStopReason(string(event.Delta.StopReason))
	delta := providers.ChunkDelta{Extra: refusalExtra(event.Delta.StopDetails)}
	chunk := s.chunk(delta)
	chunk.Choices[0].FinishReason = finishReason
	chunk.Usage = &providers.Usage{
		CacheCreationInputTokens: int(s.cacheCreationUsage),
		CacheReadInputTokens:     int(s.cacheReadUsage),
		PromptTokens:             int(s.inputUsage + s.cacheCreationUsage + s.cacheReadUsage),
		CompletionTokens:         int(event.Usage.OutputTokens),
		ReasoningTokens:          int(event.Usage.OutputTokensDetails.ThinkingTokens),
		TotalTokens: int(
			s.inputUsage + s.cacheCreationUsage + s.cacheReadUsage + event.Usage.OutputTokens,
		),
	}
	return chunk
}

func refusalExtra(details anthropic.RefusalStopDetails) map[string]providers.ProviderData {
	if !details.JSON.Type.Valid() {
		return nil
	}
	return map[string]providers.ProviderData{providerName: {
		"stop_details": providers.ProviderData{
			"type":        string(details.Type),
			"category":    string(details.Category),
			"explanation": details.Explanation,
		},
	}}
}

// handleMessageStart processes a message_start event and returns the initial chunk.
func (s *streamState) handleMessageStart(event anthropic.MessageStartEvent) providers.ChatCompletionChunk {
	s.messageID = event.Message.ID
	s.model = event.Message.Model
	s.inputUsage = event.Message.Usage.InputTokens
	s.cacheCreationUsage = event.Message.Usage.CacheCreationInputTokens
	s.cacheReadUsage = event.Message.Usage.CacheReadInputTokens

	return s.chunk(providers.ChunkDelta{Role: providers.RoleAssistant})
}

// handleThinkingDelta processes a thinking delta and returns a chunk.
func (s *streamState) handleThinkingDelta(thinking string) *providers.ChatCompletionChunk {
	s.reasoning.WriteString(thinking)
	if s.currentThinkingIdx >= 0 {
		previous, _ := s.thinkingBlocks[s.currentThinkingIdx]["thinking"].(string)
		s.thinkingBlocks[s.currentThinkingIdx]["thinking"] = previous + thinking
	}
	chunk := s.chunk(providers.ChunkDelta{
		Reasoning: &providers.Reasoning{Content: thinking}, Extra: s.thinkingExtra(),
	})
	return &chunk
}

func (s *streamState) thinkingExtra() map[string]providers.ProviderData {
	blocks := make([]providers.ProviderData, len(s.thinkingBlocks))
	for i, block := range s.thinkingBlocks {
		blocks[i] = make(providers.ProviderData, len(block))
		maps.Copy(blocks[i], block)
	}
	return map[string]providers.ProviderData{providerName: {"thinking_blocks": blocks}}
}

// handleTextDelta processes a text delta and returns a chunk.
func (s *streamState) handleTextDelta(text string) *providers.ChatCompletionChunk {
	s.content.WriteString(text)
	chunk := s.chunk(providers.ChunkDelta{Content: text})
	return &chunk
}
