package anthropic

import (
	"context"
	"encoding/json"
	stderrors "errors"
	"fmt"
	"maps"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

// CreateBatch validates and creates an Anthropic Message Batch.
func (p *Provider) CreateBatch(ctx context.Context, params providers.CreateBatchParams) (*providers.Batch, error) {
	requests, err := convertBatchRequests(params)
	if err != nil {
		return nil, err
	}
	batch, err := p.client.Messages.Batches.New(ctx, anthropic.MessageBatchNewParams{Requests: requests})
	if err != nil {
		return nil, p.ConvertError(err)
	}
	return convertBatch(batch), nil
}

func validateCreateBatchParams(params providers.CreateBatchParams) error {
	if len(params.Requests) == 0 {
		return errors.NewInvalidRequestError(providerName, fmt.Errorf("batch requests are required"))
	}
	if params.CompletionWindow != "" && params.CompletionWindow != "24h" {
		return errors.NewInvalidRequestError(
			providerName,
			fmt.Errorf("anthropic batch completion_window must be 24h"),
		)
	}
	if params.Endpoint != "" && params.Endpoint != "/v1/messages" {
		return errors.NewInvalidRequestError(
			providerName,
			fmt.Errorf("anthropic batch endpoint must be %q", "/v1/messages"),
		)
	}
	if len(params.Metadata) != 0 {
		return errors.NewInvalidRequestError(providerName, fmt.Errorf("anthropic batch metadata is unsupported"))
	}
	seen := make(map[string]struct{}, len(params.Requests))
	for _, item := range params.Requests {
		if item.CustomID == "" {
			return errors.NewInvalidRequestError(providerName, fmt.Errorf("batch custom_id is required"))
		}
		if _, ok := seen[item.CustomID]; ok {
			return errors.NewInvalidRequestError(
				providerName,
				fmt.Errorf("duplicate batch custom_id %q", item.CustomID),
			)
		}
		seen[item.CustomID] = struct{}{}
	}
	return nil
}

func convertBatchRequests(params providers.CreateBatchParams) ([]anthropic.MessageBatchNewParamsRequest, error) {
	if err := validateCreateBatchParams(params); err != nil {
		return nil, err
	}
	requests := make([]anthropic.MessageBatchNewParamsRequest, 0, len(params.Requests))
	for _, item := range params.Requests {
		request, err := convertBatchRequest(item, params.Model)
		if err != nil {
			return nil, err
		}
		requests = append(requests, request)
	}
	return requests, nil
}

func convertBatchRequest(
	item providers.BatchRequestItem,
	model string,
) (anthropic.MessageBatchNewParamsRequest, error) {
	body := item.Body
	if _, ok := body["model"]; !ok && model != "" {
		body = cloneMap(body)
		body["model"] = model
	}
	data, err := json.Marshal(body)
	if err != nil {
		return anthropic.MessageBatchNewParamsRequest{}, batchRequestValidationError(item.CustomID, err)
	}
	var native anthropic.MessageBatchNewParamsRequestParams
	if err := json.Unmarshal(data, &native); err != nil {
		return anthropic.MessageBatchNewParamsRequest{}, batchRequestValidationError(item.CustomID, err)
	}
	if native.Model == "" || len(native.Messages) == 0 || native.MaxTokens <= 0 {
		return anthropic.MessageBatchNewParamsRequest{}, errors.NewInvalidRequestError(
			providerName,
			fmt.Errorf("batch request %q requires model, messages, and positive max_tokens", item.CustomID),
		)
	}
	return anthropic.MessageBatchNewParamsRequest{CustomID: item.CustomID, Params: native}, nil
}

func batchRequestValidationError(customID string, err error) error {
	return errors.NewInvalidRequestError(providerName, fmt.Errorf("batch request %q: %w", customID, err))
}

// RetrieveBatch retrieves an Anthropic Message Batch by ID.
func (p *Provider) RetrieveBatch(ctx context.Context, batchID, provider string) (*providers.Batch, error) {
	if err := validateBatchArgs(batchID, provider); err != nil {
		return nil, err
	}
	batch, err := p.client.Messages.Batches.Get(ctx, batchID)
	if err != nil {
		return nil, p.ConvertError(err)
	}
	return convertBatch(batch), nil
}

// CancelBatch requests cancellation of an Anthropic Message Batch.
func (p *Provider) CancelBatch(ctx context.Context, batchID, provider string) (*providers.Batch, error) {
	if err := validateBatchArgs(batchID, provider); err != nil {
		return nil, err
	}
	batch, err := p.client.Messages.Batches.Cancel(ctx, batchID)
	if err != nil {
		return nil, p.ConvertError(err)
	}
	return convertBatch(batch), nil
}

// ListBatches lists Anthropic Message Batches, optionally starting after a batch ID.
func (p *Provider) ListBatches(
	ctx context.Context,
	provider string,
	opts providers.ListBatchesOptions,
) ([]providers.Batch, error) {
	params, err := batchListParams(provider, opts)
	if err != nil {
		return nil, err
	}
	pager := p.client.Messages.Batches.ListAutoPaging(ctx, params)
	batches := make([]providers.Batch, 0)
	for pager.Next() {
		current := pager.Current()
		batches = append(batches, *convertBatch(&current))
		if opts.Limit != nil && len(batches) >= *opts.Limit {
			break
		}
	}
	if err := pager.Err(); err != nil {
		return nil, p.ConvertError(err)
	}
	return batches, nil
}

func batchListParams(provider string, opts providers.ListBatchesOptions) (anthropic.MessageBatchListParams, error) {
	if provider != "" && provider != providerName {
		return anthropic.MessageBatchListParams{}, errors.NewInvalidRequestError(
			providerName, fmt.Errorf("provider must be %q", providerName),
		)
	}
	params := anthropic.MessageBatchListParams{}
	if opts.After != "" {
		params.AfterID = anthropic.String(opts.After)
	}
	if opts.Limit == nil {
		return params, nil
	}
	if *opts.Limit < 1 || *opts.Limit > 1000 {
		return anthropic.MessageBatchListParams{}, errors.NewInvalidRequestError(
			providerName, fmt.Errorf("limit must be between 1 and 1000"),
		)
	}
	params.Limit = anthropic.Int(int64(*opts.Limit))
	return params, nil
}

// RetrieveBatchResults retrieves the results of a terminal Anthropic Message Batch.
func (p *Provider) RetrieveBatchResults(
	ctx context.Context,
	batchID string,
	provider string,
) (_ *providers.BatchResult, err error) {
	batch, err := p.RetrieveBatch(ctx, batchID, provider)
	if err != nil {
		return nil, err
	}
	if !isTerminalBatchStatus(batch.Status) {
		return nil, errors.NewInvalidRequestError(
			providerName,
			fmt.Errorf("batch %q is not complete: %s", batchID, batch.Status),
		)
	}
	stream := p.client.Messages.Batches.ResultsStreaming(ctx, batchID)
	defer func() {
		if closeErr := stream.Close(); closeErr != nil {
			err = stderrors.Join(err, p.ConvertError(closeErr))
		}
	}()
	result := &providers.BatchResult{}
	for stream.Next() {
		item, err := convertBatchResultItem(stream.Current())
		if err != nil {
			return nil, err
		}
		result.Results = append(result.Results, item)
	}
	if err := stream.Err(); err != nil {
		return nil, p.ConvertError(err)
	}
	return result, nil
}

func isTerminalBatchStatus(status providers.BatchStatus) bool {
	return status == providers.BatchStatusCompleted || status == providers.BatchStatusFailed ||
		status == providers.BatchStatusCancelled || status == providers.BatchStatusExpired
}

func convertBatchResultItem(entry anthropic.MessageBatchIndividualResponse) (providers.BatchResultItem, error) {
	item := providers.BatchResultItem{CustomID: entry.CustomID}
	switch entry.Result.Type {
	case "succeeded":
		result, err := convertResponse(&entry.Result.Message)
		if err != nil {
			return providers.BatchResultItem{}, err
		}
		item.Result = result
	case "errored":
		item.Error = &providers.BatchResultError{
			Code:    entry.Result.Error.Error.Type,
			Message: entry.Result.Error.Error.Message,
		}
	case "canceled", "expired":
		item.Error = &providers.BatchResultError{Code: entry.Result.Type, Message: "Request " + entry.Result.Type}
	default:
		item.Error = &providers.BatchResultError{Code: "unknown", Message: "Unknown batch result"}
	}
	return item, nil
}

func cloneMap(source map[string]any) map[string]any {
	result := maps.Clone(source)
	if result == nil {
		result = make(map[string]any)
	}
	return result
}

func validateBatchArgs(batchID, provider string) error {
	if batchID == "" {
		return errors.NewInvalidRequestError(providerName, fmt.Errorf("batch ID is required"))
	}
	if provider != "" && provider != providerName {
		return errors.NewInvalidRequestError(providerName, fmt.Errorf("provider must be %q", providerName))
	}
	return nil
}

func convertBatch(batch *anthropic.MessageBatch) *providers.Batch {
	counts := batch.RequestCounts
	total := counts.Processing + counts.Succeeded + counts.Errored + counts.Canceled + counts.Expired
	status := providers.BatchStatusInProgress
	if batch.ProcessingStatus == anthropic.MessageBatchProcessingStatusCanceling {
		status = providers.BatchStatusCancelling
	}
	if batch.ProcessingStatus == anthropic.MessageBatchProcessingStatusEnded {
		status = providers.BatchStatusCompleted
	}
	result := &providers.Batch{
		CompletionWindow: "24h",
		CreatedAt:        batch.CreatedAt.Unix(),
		Endpoint:         "/v1/messages",
		ID:               batch.ID,
		Object:           "batch",
		Provider:         providerName,
		RequestCounts: &providers.BatchRequestCounts{
			Completed: int(counts.Succeeded),
			Failed:    int(counts.Errored + counts.Canceled + counts.Expired),
			Total:     int(total),
		},
		Status: status,
	}
	if !batch.EndedAt.IsZero() {
		result.CompletedAt = new(batch.EndedAt.Unix())
	}
	return result
}
