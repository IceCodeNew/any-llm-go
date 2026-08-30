package anthropic

import (
	"encoding/json"
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
	responseBlocks     map[int]providers.ProviderData
	responseBlockOrder []int
	responseInputJSON  map[int]string
	preserveResponses  bool
}

// newStreamState creates a new stream state with default values.
func newStreamState() *streamState {
	return &streamState{
		currentToolIdx:     -1,
		currentThinkingIdx: -1,
		responseBlocks:     make(map[int]providers.ProviderData),
		responseInputJSON:  make(map[int]string),
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
		s.appendResponseBlockString(event.Index, "text", event.Delta.Text)
		return s.handleTextDelta(event.Delta.Text)
	case deltaTypeThinking:
		s.appendResponseBlockString(event.Index, "thinking", event.Delta.Thinking)
		return s.handleThinkingDelta(event.Delta.Thinking)
	case deltaTypeSignature:
		s.appendResponseBlockString(event.Index, "signature", event.Delta.Signature)
		if s.currentThinkingIdx >= 0 {
			previous, _ := s.thinkingBlocks[s.currentThinkingIdx]["signature"].(string)
			s.thinkingBlocks[s.currentThinkingIdx]["signature"] = previous + event.Delta.Signature
		}
		chunk := s.chunk(providers.ChunkDelta{Extra: mergeExtras(s.thinkingExtra(), s.responseExtra())})
		return &chunk
	case deltaTypeInputJSON:
		index := int(event.Index)
		s.responseInputJSON[index] += event.Delta.PartialJSON
		s.updateResponseBlockInput(event.Index)
		if block := s.responseBlocks[index]; block != nil && block["type"] != blockTypeToolUse {
			chunk := s.chunk(providers.ChunkDelta{Extra: s.responseExtra()})
			return &chunk
		}
		return s.handleInputJSONDelta(event.Delta.PartialJSON)
	case deltaTypeCitations:
		s.appendResponseBlockValue(event.Index, "citations", event.Delta.Citation)
		// Anthropic sends citations as first-class content deltas. Keep the SDK
		// union in provider metadata so every documented citation variant survives.
		// https://docs.anthropic.com/en/docs/build-with-claude/citations
		chunk := s.chunk(providers.ChunkDelta{Extra: mergeExtras(map[string]providers.ProviderData{
			providerName: {
				"citations": []any{event.Delta.Citation},
				"index":     event.Index,
			},
		}, s.responseExtra())})

		return &chunk
	default:
		return nil
	}
}

// handleContentBlockStart processes a content_block_start event.
func (s *streamState) handleContentBlockStart(event anthropic.ContentBlockStartEvent) *providers.ChatCompletionChunk {
	s.recordResponseBlock(event.Index, event.ContentBlock)
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
		chunk := s.chunk(providers.ChunkDelta{ToolCalls: []providers.ToolCall{tc}})

		return &chunk
	default:
		chunk := s.chunk(providers.ChunkDelta{Extra: s.responseExtra()})
		return &chunk
	}
	return nil
}

func (s *streamState) recordResponseBlock(
	index int64,
	block anthropic.ContentBlockStartEventContentBlockUnion,
) {
	raw := []byte(block.RawJSON())
	if len(raw) == 0 {
		var rawValue any = block.AsAny()
		encoded, err := json.Marshal(rawValue)
		if err != nil {
			return
		}
		raw = encoded
	}
	var data providers.ProviderData
	if err := json.Unmarshal(raw, &data); err != nil {
		return
	}
	blockIndex := int(index)
	if _, exists := s.responseBlocks[blockIndex]; !exists {
		s.responseBlockOrder = append(s.responseBlockOrder, blockIndex)
	}
	s.responseBlocks[blockIndex] = data
	if !normalizedResponseBlockType(block.Type) {
		s.preserveResponses = true
	}
}

func normalizedResponseBlockType(blockType string) bool {
	switch blockType {
	case blockTypeText, blockTypeThinking, blockTypeRedactedThinking, blockTypeToolUse:
		return true
	default:
		return false
	}
}

func (s *streamState) appendResponseBlockString(index int64, key, delta string) {
	block := s.responseBlocks[int(index)]
	if block == nil {
		return
	}
	previous, _ := block[key].(string)
	block[key] = previous + delta
}

func (s *streamState) appendResponseBlockValue(index int64, key string, value any) {
	block := s.responseBlocks[int(index)]
	if block == nil {
		return
	}
	values, _ := block[key].([]any)
	block[key] = append(values, value)
}

func (s *streamState) updateResponseBlockInput(index int64) {
	block := s.responseBlocks[int(index)]
	if block == nil {
		return
	}
	partial := s.responseInputJSON[int(index)]
	var input any
	if err := json.Unmarshal([]byte(partial), &input); err != nil {
		block["input_json_delta"] = partial
		return
	}
	delete(block, "input_json_delta")
	block["input"] = input
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
	delta := providers.ChunkDelta{Extra: mergeExtras(refusalExtra(event.Delta.StopDetails), s.responseExtra())}
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
		Reasoning: &providers.Reasoning{Content: thinking},
		Extra:     mergeExtras(s.thinkingExtra(), s.responseExtra()),
	})
	return &chunk
}

func (s *streamState) responseExtra() map[string]providers.ProviderData {
	if !s.preserveResponses {
		return nil
	}
	blocks := make([]providers.ProviderData, 0, len(s.responseBlockOrder))
	for _, index := range s.responseBlockOrder {
		block := s.responseBlocks[index]
		copy := make(providers.ProviderData, len(block))
		maps.Copy(copy, block)
		blocks = append(blocks, copy)
	}
	return map[string]providers.ProviderData{providerName: {"response_blocks": blocks}}
}

func mergeExtras(extras ...map[string]providers.ProviderData) map[string]providers.ProviderData {
	var result map[string]providers.ProviderData
	for _, extra := range extras {
		for provider, data := range extra {
			if result == nil {
				result = make(map[string]providers.ProviderData)
			}
			if result[provider] == nil {
				result[provider] = make(providers.ProviderData)
			}
			maps.Copy(result[provider], data)
		}
	}
	return result
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
	chunk := s.chunk(providers.ChunkDelta{Content: text, Extra: s.responseExtra()})
	return &chunk
}
