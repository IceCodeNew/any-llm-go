package openai

import (
	"context"

	"github.com/mozilla-ai/any-llm-go/providers"
	"github.com/mozilla-ai/any-llm-go/providers/internal/openairesponses"
)

// Responses implements the provider-neutral Responses interface. Use the
// SDK-native lifecycle methods when callers need the full input and output unions.
func (p *Provider) Responses(
	ctx context.Context,
	params providers.ResponsesParams,
) (*providers.ResponsesResult, error) {
	// The helper returns provider-neutral validation and API errors unchanged.
	return openairesponses.Create( //nolint:wrapcheck
		ctx,
		&p.client,
		providerName,
		p.ConvertError,
		params,
	)
}
