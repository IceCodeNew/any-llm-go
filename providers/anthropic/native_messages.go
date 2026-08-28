package anthropic

import (
	"context"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// Messages sends a native Anthropic Messages request without OpenAI normalization.
func (p *Provider) Messages(
	ctx context.Context,
	params anthropic.MessageNewParams,
	opts ...option.RequestOption,
) (*anthropic.Message, error) {
	message, err := p.client.Messages.New(ctx, params, opts...)
	if err != nil {
		return nil, p.ConvertError(err)
	}
	return message, nil
}

// MessagesStreaming sends a native request and yields official SDK event unions.
func (p *Provider) MessagesStreaming(
	ctx context.Context,
	params anthropic.MessageNewParams,
	opts ...option.RequestOption,
) (<-chan anthropic.MessageStreamEventUnion, <-chan error) {
	events := make(chan anthropic.MessageStreamEventUnion)
	errs := make(chan error, 1)
	go func() {
		defer close(events)
		defer close(errs)
		stream := p.client.Messages.NewStreaming(ctx, params, opts...)
		defer func() {
			if err := stream.Close(); err != nil {
				reportStreamError(errs, p.ConvertError(err))
			}
		}()
		for stream.Next() {
			select {
			case events <- stream.Current():
			case <-ctx.Done():
				reportStreamError(errs, ctx.Err())
				return
			}
		}
		if err := stream.Err(); err != nil {
			reportStreamError(errs, p.ConvertError(err))
		}
	}()
	return events, errs
}

// BetaMessages sends an official SDK beta Messages request. This exposes beta
// fields such as context management without duplicating SDK unions.
func (p *Provider) BetaMessages(
	ctx context.Context,
	params anthropic.BetaMessageNewParams,
	opts ...option.RequestOption,
) (*anthropic.BetaMessage, error) {
	message, err := p.client.Beta.Messages.New(ctx, params, opts...)
	if err != nil {
		return nil, p.ConvertError(err)
	}
	return message, nil
}

// BetaMessagesStreaming yields official SDK beta stream event unions.
func (p *Provider) BetaMessagesStreaming(
	ctx context.Context,
	params anthropic.BetaMessageNewParams,
	opts ...option.RequestOption,
) (<-chan anthropic.BetaRawMessageStreamEventUnion, <-chan error) {
	events := make(chan anthropic.BetaRawMessageStreamEventUnion)
	errs := make(chan error, 1)
	go func() {
		defer close(events)
		defer close(errs)
		stream := p.client.Beta.Messages.NewStreaming(ctx, params, opts...)
		defer func() {
			if err := stream.Close(); err != nil {
				reportStreamError(errs, p.ConvertError(err))
			}
		}()
		for stream.Next() {
			select {
			case events <- stream.Current():
			case <-ctx.Done():
				reportStreamError(errs, ctx.Err())
				return
			}
		}
		if err := stream.Err(); err != nil {
			reportStreamError(errs, p.ConvertError(err))
		}
	}()
	return events, errs
}
