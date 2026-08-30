package gemini

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"google.golang.org/genai"

	"github.com/mozilla-ai/any-llm-go/providers"
)

type streamState struct {
	content          strings.Builder
	finishReason     genai.FinishReason
	messageID        string
	model            string
	reasoning        strings.Builder
	responseEvents   []json.RawMessage
	responseParts    []*genai.Part
	responsePartsRaw json.RawMessage
	toolCalls        []providers.ToolCall
	usage            *providers.Usage
}

func newStreamState(model string) (*streamState, error) {
	id, err := generateID(idPrefixCompletion)
	if err != nil {
		return nil, err
	}
	return &streamState{
		messageID: id,
		model:     model,
	}, nil
}

func (s *streamState) chunk(delta providers.ChunkDelta) providers.ChatCompletionChunk {
	return providers.ChatCompletionChunk{
		ID:     s.messageID,
		Object: objectChatCompletionChunk,
		Model:  s.model,
		Choices: []providers.ChunkChoice{{
			Index: 0,
			Delta: delta,
		}},
	}
}

func (s *streamState) finalChunk() *providers.ChatCompletionChunk {
	delta := providers.ChunkDelta{}
	if len(s.responseEvents) > 0 {
		delta.Extra = withGeminiExtra(delta.Extra, extraKeyResponseEvents, s.responseEvents)
	}
	if len(s.responsePartsRaw) > 0 {
		delta.Extra = withResponseParts(delta.Extra, s.responsePartsRaw)
	}
	chunk := s.chunk(delta)

	finishReason := convertFinishReason(s.finishReason)
	if len(s.toolCalls) > 0 && finishReason == providers.FinishReasonStop {
		finishReason = providers.FinishReasonToolCalls
	}

	chunk.Choices[0].FinishReason = finishReason
	chunk.Usage = s.usage
	return &chunk
}

func (s *streamState) processResponse(resp *genai.GenerateContentResponse) ([]providers.ChatCompletionChunk, error) {
	rawResponse, err := rawGeminiResponse(resp)
	if err != nil {
		return nil, err
	}
	s.responseEvents = append(s.responseEvents, rawResponse)

	if resp.UsageMetadata != nil {
		s.usage = &providers.Usage{
			PromptTokens:     int(resp.UsageMetadata.PromptTokenCount),
			CompletionTokens: int(resp.UsageMetadata.CandidatesTokenCount),
			TotalTokens:      usageTotal(resp),
			ReasoningTokens:  int(resp.UsageMetadata.ThoughtsTokenCount),
			CachedTokens:     int(resp.UsageMetadata.CachedContentTokenCount),
		}
	}

	if len(resp.Candidates) == 0 {
		if promptWasBlocked(resp) {
			s.finishReason = genai.FinishReasonSafety
		}
		return nil, nil
	}

	candidate := resp.Candidates[0]

	if candidate.FinishReason != "" {
		s.finishReason = candidate.FinishReason
	}

	if candidate.Content == nil {
		return nil, nil
	}
	s.responseParts = append(s.responseParts, candidate.Content.Parts...)
	raw, err := json.Marshal(s.responseParts)
	if err != nil {
		return nil, fmt.Errorf("serializing accumulated Gemini response parts: %w", err)
	}
	s.responsePartsRaw = raw

	var result []providers.ChatCompletionChunk
	for _, part := range candidate.Content.Parts {
		chunk, ok, err := s.processPart(part)
		if err != nil {
			return nil, err
		}
		if ok {
			result = append(result, chunk)
		}
	}

	return result, nil
}

func (s *streamState) processPart(part *genai.Part) (providers.ChatCompletionChunk, bool, error) {
	switch {
	case part.FunctionCall != nil:
		toolCall, err := convertFunctionCallToToolCall(part.FunctionCall)
		if err != nil {
			return providers.ChatCompletionChunk{}, false, err
		}
		if len(part.ThoughtSignature) > 0 {
			setProviderExtra(
				&toolCall,
				providerName,
				extraKeyThoughtSignature,
				base64.StdEncoding.EncodeToString(part.ThoughtSignature),
			)
		}
		s.toolCalls = append(s.toolCalls, toolCall)
		chunk := s.chunk(providers.ChunkDelta{ToolCalls: []providers.ToolCall{toolCall}})
		return chunk, true, nil
	case part.Thought:
		s.reasoning.WriteString(part.Text)
		chunk := s.chunk(providers.ChunkDelta{Reasoning: &providers.Reasoning{Content: part.Text}})
		return chunk, true, nil
	case part.Text != "":
		s.content.WriteString(part.Text)
		chunk := s.chunk(providers.ChunkDelta{Content: part.Text, Extra: thoughtSignatureExtra(part)})
		return chunk, true, nil
	default:
		raw, err := json.Marshal([]*genai.Part{part})
		if err != nil {
			return providers.ChatCompletionChunk{}, false, fmt.Errorf("serializing Gemini response part: %w", err)
		}
		chunk := s.chunk(providers.ChunkDelta{Extra: withResponseParts(nil, raw)})
		return chunk, true, nil
	}
}
