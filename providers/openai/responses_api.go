package openai

import (
	"context"

	"github.com/openai/openai-go/v3/packages/pagination"
	"github.com/openai/openai-go/v3/responses"
)

// CreateResponse creates a response without translating SDK unions, preserving
// multimodal inputs, built-in and function tools, structured output, and metadata.
func (p *Provider) CreateResponse(
	ctx context.Context,
	params responses.ResponseNewParams,
) (*responses.Response, error) {
	if params.Model == "" {
		return nil, invalid("model is required")
	}
	response, err := p.client.Responses.New(ctx, params)
	if err != nil {
		return nil, p.ConvertError(err)
	}
	return response, nil
}

// StreamResponse streams every SDK event without flattening event-specific data.
func (p *Provider) StreamResponse(
	ctx context.Context,
	params responses.ResponseNewParams,
) (<-chan responses.ResponseStreamEventUnion, <-chan error) {
	events := make(chan responses.ResponseStreamEventUnion)
	errs := make(chan error, 1)
	go func() {
		defer close(events)
		defer close(errs)
		if params.Model == "" {
			reportStreamError(errs, invalid("model is required"))
			return
		}
		stream := p.client.Responses.NewStreaming(ctx, params)
		p.pumpResponseStream(ctx, stream, events, errs)
	}()
	return events, errs
}

// RetrieveResponse retrieves a response by ID.
func (p *Provider) RetrieveResponse(
	ctx context.Context,
	id string,
	params responses.ResponseGetParams,
) (*responses.Response, error) {
	if id == "" {
		return nil, invalid("response ID is required")
	}
	response, err := p.client.Responses.Get(ctx, id, params)
	if err != nil {
		return nil, p.ConvertError(err)
	}
	return response, nil
}

// DeleteResponse deletes a response by ID.
func (p *Provider) DeleteResponse(ctx context.Context, id string) error {
	if id == "" {
		return invalid("response ID is required")
	}
	if err := p.client.Responses.Delete(ctx, id); err != nil {
		return p.ConvertError(err)
	}
	return nil
}

// CancelResponse cancels an in-progress response by ID.
func (p *Provider) CancelResponse(ctx context.Context, id string) (*responses.Response, error) {
	if id == "" {
		return nil, invalid("response ID is required")
	}
	response, err := p.client.Responses.Cancel(ctx, id)
	if err != nil {
		return nil, p.ConvertError(err)
	}
	return response, nil
}

// ListResponseInputItems lists the input items for a response.
func (p *Provider) ListResponseInputItems(
	ctx context.Context,
	id string,
	params responses.InputItemListParams,
) (*pagination.CursorPage[responses.ResponseItemUnion], error) {
	if id == "" {
		return nil, invalid("response ID is required")
	}
	items, err := p.client.Responses.InputItems.List(ctx, id, params)
	if err != nil {
		return nil, p.ConvertError(err)
	}
	return items, nil
}

// ReplayResponse resumes an interrupted response event stream.
func (p *Provider) ReplayResponse(
	ctx context.Context,
	id string,
	params responses.ResponseGetParams,
) (<-chan responses.ResponseStreamEventUnion, <-chan error) {
	events := make(chan responses.ResponseStreamEventUnion)
	errs := make(chan error, 1)
	go func() {
		defer close(events)
		defer close(errs)
		if id == "" {
			reportStreamError(errs, invalid("response ID is required"))
			return
		}
		stream := p.client.Responses.GetStreaming(ctx, id, params)
		p.pumpResponseStream(ctx, stream, events, errs)
	}()
	return events, errs
}

type responseStream interface {
	Next() bool
	Current() responses.ResponseStreamEventUnion
	Err() error
	Close() error
}

func (p *Provider) pumpResponseStream(
	ctx context.Context,
	stream responseStream,
	events chan<- responses.ResponseStreamEventUnion,
	errs chan<- error,
) {
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
}
