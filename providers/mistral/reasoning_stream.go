package mistral

import (
	"bytes"
	"context"
	"encoding/json"

	"github.com/mozilla-ai/any-llm-go/providers"
)

type streamedReasoning struct {
	chunks []json.RawMessage
	closed bool
}

func assembleReasoningStream(
	ctx context.Context,
	cancel context.CancelFunc,
	upstreamChunks <-chan providers.ChatCompletionChunk,
	upstreamErrs <-chan error,
) (<-chan providers.ChatCompletionChunk, <-chan error) {
	chunks := make(chan providers.ChatCompletionChunk)
	errs := make(chan error, 1)

	go func() {
		defer cancel()
		defer close(chunks)
		defer close(errs)

		states := make(map[int]*streamedReasoning)
		pending := make([]providers.ChatCompletionChunk, 0, 1)
		chunkInput, errInput := upstreamChunks, upstreamErrs
		buffering := false

		for chunkInput != nil || errInput != nil {
			select {
			case <-ctx.Done():
				errs <- ctx.Err()

				return
			case chunk, ok := <-chunkInput:
				if !ok {
					chunkInput = nil

					continue
				}

				accumulateReasoning(chunk, states)

				if hasFinish(chunk) {
					attachReasoningSnapshots(&chunk, states)

					buffering = true
				}

				if buffering {
					pending = append(pending, chunk)

					continue
				}

				if !forwardChunk(ctx, chunks, chunk) {
					errs <- ctx.Err()

					return
				}
			case err, ok := <-errInput:
				if !ok {
					errInput = nil

					continue
				}

				if err != nil {
					errs <- err

					return
				}
			}
		}

		for _, chunk := range pending {
			if !forwardChunk(ctx, chunks, chunk) {
				errs <- ctx.Err()

				return
			}
		}
	}()

	return chunks, errs
}

func accumulateReasoning(chunk providers.ChatCompletionChunk, states map[int]*streamedReasoning) {
	for _, choice := range chunk.Choices {
		state := states[choice.Index]
		if choice.Delta.Reasoning != nil && len(choice.Delta.Reasoning.ProviderRaw) > 0 {
			var fragments []json.RawMessage
			if json.Unmarshal(choice.Delta.Reasoning.ProviderRaw, &fragments) == nil {
				if state == nil {
					state = &streamedReasoning{}
					states[choice.Index] = state
				}

				for _, fragment := range fragments {
					state.chunks = append(state.chunks, bytes.Clone(fragment))

					var metadata struct {
						Type   *string `json:"type"`
						Closed *bool   `json:"closed"`
					}
					if json.Unmarshal(fragment, &metadata) == nil && metadata.Type != nil &&
						*metadata.Type == "thinking" {
						state.closed = metadata.Closed != nil && *metadata.Closed
					}
				}
			}
		}

		if state != nil && choice.Delta.Content != "" && choice.Delta.Reasoning == nil {
			text, err := json.Marshal(struct {
				Type string `json:"type"`
				Text string `json:"text"`
			}{Type: "text", Text: choice.Delta.Content})
			if err == nil {
				state.chunks = append(state.chunks, text)
			}
		}
	}
}

func hasFinish(chunk providers.ChatCompletionChunk) bool {
	for _, choice := range chunk.Choices {
		if choice.FinishReason != "" {
			return true
		}
	}

	return false
}

func attachReasoningSnapshots(chunk *providers.ChatCompletionChunk, states map[int]*streamedReasoning) {
	for choiceIndex := range chunk.Choices {
		choice := &chunk.Choices[choiceIndex]

		state := states[choice.Index]
		if choice.FinishReason == "" || state == nil || !state.closed {
			continue
		}

		raw, err := json.Marshal(state.chunks)
		if err != nil {
			continue
		}

		_, reasoning, chunked, err := decodeContent(raw)
		if err == nil && chunked && reasoning != nil {
			reasoning.Content = ""
			if choice.Delta.Reasoning != nil {
				reasoning.Content = choice.Delta.Reasoning.Content
			}

			choice.Delta.Reasoning = reasoning
		}
	}
}

func forwardChunk(
	ctx context.Context,
	chunks chan<- providers.ChatCompletionChunk,
	chunk providers.ChatCompletionChunk,
) bool {
	select {
	case chunks <- chunk:
		return true
	case <-ctx.Done():
		return false
	}
}
