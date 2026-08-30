package gemini

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"

	"google.golang.org/genai"

	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

const batchEndpoint = "/v1/chat/completions"

var _ providers.BatchProvider = (*Provider)(nil)

// CreateBatch creates a Gemini batch job.
func (p *Provider) CreateBatch(ctx context.Context, params providers.CreateBatchParams) (*providers.Batch, error) {
	source, cfg, err := p.convertCreateBatchParams(params)
	if err != nil {
		return nil, err
	}
	job, err := p.client.Batches.Create(ctx, params.Model, source, cfg)
	if err != nil {
		return nil, p.ConvertError(err)
	}
	return convertBatch(job), nil
}

func (p *Provider) convertCreateBatchParams(
	params providers.CreateBatchParams,
) (*genai.BatchJobSource, *genai.CreateBatchJobConfig, error) {
	if err := validateCreateBatchParams(params); err != nil {
		return nil, nil, err
	}

	requests := make([]*genai.InlinedRequest, 0, len(params.Requests))
	customIDs := make(map[string]struct{}, len(params.Requests))
	for _, item := range params.Requests {
		request, err := p.convertBatchRequest(params.Model, item, customIDs)
		if err != nil {
			return nil, nil, err
		}
		requests = append(requests, request)
	}

	cfg := &genai.CreateBatchJobConfig{DisplayName: params.Metadata["display_name"]}
	for _, request := range requests {
		if hasThoughtSignatureBypass(request.Contents) {
			cfg.HTTPOptions = &genai.HTTPOptions{ExtrasRequestProvider: rewriteThoughtSignatureBypass}
			break
		}
	}

	return &genai.BatchJobSource{InlinedRequests: requests}, cfg, nil
}

func validateCreateBatchParams(params providers.CreateBatchParams) error {
	if params.Model == "" || len(params.Requests) == 0 {
		return errors.NewInvalidRequestError(providerName, fmt.Errorf("batch model and requests are required"))
	}
	if params.Endpoint != "" && params.Endpoint != batchEndpoint {
		return errors.NewInvalidRequestError(
			providerName,
			fmt.Errorf("gemini batch endpoint must be %q", batchEndpoint),
		)
	}
	if params.CompletionWindow != "" && params.CompletionWindow != "24h" {
		return errors.NewInvalidRequestError(providerName, fmt.Errorf("gemini batch completion_window must be 24h"))
	}
	for key := range params.Metadata {
		if key != "display_name" {
			return errors.NewInvalidRequestError(
				providerName,
				fmt.Errorf("gemini batch metadata key %q is unsupported", key),
			)
		}
	}
	return nil
}

func (p *Provider) convertBatchRequest(
	model string,
	item providers.BatchRequestItem,
	customIDs map[string]struct{},
) (*genai.InlinedRequest, error) {
	if item.CustomID == "" {
		return nil, errors.NewInvalidRequestError(providerName, fmt.Errorf("batch custom_id is required"))
	}
	if _, exists := customIDs[item.CustomID]; exists {
		return nil, errors.NewInvalidRequestError(
			providerName,
			fmt.Errorf("batch custom_id %q is duplicated", item.CustomID),
		)
	}
	customIDs[item.CustomID] = struct{}{}

	body, err := json.Marshal(item.Body)
	if err != nil {
		return nil, errors.NewInvalidRequestError(providerName, err)
	}
	var completion providers.CompletionParams
	if err = json.Unmarshal(body, &completion); err != nil {
		return nil, errors.NewInvalidRequestError(providerName, err)
	}
	if completion.Model != "" && completion.Model != model {
		return nil, errors.NewInvalidRequestError(
			providerName,
			fmt.Errorf("batch request model %q does not match batch model %q", completion.Model, model),
		)
	}
	completion.Model = model
	contents, cfg, err := p.convertParams(completion)
	if err != nil {
		return nil, err
	}

	return &genai.InlinedRequest{
		Model: model, Contents: contents, Config: cfg,
		Metadata: map[string]string{"custom_id": item.CustomID},
	}, nil
}

// RetrieveBatch retrieves a Gemini batch job.
func (p *Provider) RetrieveBatch(ctx context.Context, batchID, provider string) (*providers.Batch, error) {
	if err := validateBatchProvider(provider); err != nil {
		return nil, err
	}
	if batchID == "" {
		return nil, errors.NewInvalidRequestError(providerName, fmt.Errorf("batch ID is required"))
	}
	job, err := p.client.Batches.Get(ctx, batchID, nil)
	if err != nil {
		return nil, p.ConvertError(err)
	}
	return convertBatch(job), nil
}

// CancelBatch cancels a Gemini batch job.
func (p *Provider) CancelBatch(ctx context.Context, batchID, provider string) (*providers.Batch, error) {
	if err := validateBatchProvider(provider); err != nil {
		return nil, err
	}
	if batchID == "" {
		return nil, errors.NewInvalidRequestError(providerName, fmt.Errorf("batch ID is required"))
	}
	if err := p.client.Batches.Cancel(ctx, batchID, nil); err != nil {
		return nil, p.ConvertError(err)
	}
	return p.RetrieveBatch(ctx, batchID, providerName)
}

// ListBatches lists Gemini batch jobs.
func (p *Provider) ListBatches(
	ctx context.Context,
	provider string,
	opts providers.ListBatchesOptions,
) ([]providers.Batch, error) {
	if err := validateBatchProvider(provider); err != nil {
		return nil, err
	}
	cfg, limit, err := convertListBatchesOptions(opts)
	if err != nil {
		return nil, err
	}
	page, err := p.client.Batches.List(ctx, cfg)
	if err != nil {
		return nil, p.ConvertError(err)
	}
	var result []providers.Batch
	for {
		for _, job := range page.Items {
			result = append(result, *convertBatch(job))
			if limit > 0 && len(result) == limit {
				return result, nil
			}
		}
		if page.NextPageToken == "" {
			break
		}
		page, err = page.Next(ctx)
		if err != nil {
			return nil, p.ConvertError(err)
		}
	}
	return result, nil
}

func convertListBatchesOptions(opts providers.ListBatchesOptions) (*genai.ListBatchJobsConfig, int, error) {
	cfg := &genai.ListBatchJobsConfig{PageToken: opts.After}
	if opts.Limit == nil {
		return cfg, 0, nil
	}
	if *opts.Limit <= 0 {
		return nil, 0, errors.NewInvalidRequestError(providerName, fmt.Errorf("batch limit must be positive"))
	}
	pageSize, err := positiveInt32(*opts.Limit, "batch limit")
	if err != nil {
		return nil, 0, err
	}
	cfg.PageSize = pageSize
	return cfg, *opts.Limit, nil
}

// RetrieveBatchResults retrieves the results of a completed Gemini batch job.
func (p *Provider) RetrieveBatchResults(ctx context.Context, batchID, provider string) (*providers.BatchResult, error) {
	if err := validateBatchProvider(provider); err != nil {
		return nil, err
	}
	if batchID == "" {
		return nil, errors.NewInvalidRequestError(providerName, fmt.Errorf("batch ID is required"))
	}
	job, err := p.client.Batches.Get(ctx, batchID, nil)
	if err != nil {
		return nil, p.ConvertError(err)
	}
	if !batchResultsAvailable(job) {
		return nil, errors.NewInvalidRequestError(providerName, fmt.Errorf("batch %s is not complete", batchID))
	}
	return convertBatchResults(job)
}

func batchResultsAvailable(job *genai.BatchJob) bool {
	return job.Dest != nil &&
		(job.State == genai.JobStateSucceeded || job.State == genai.JobStatePartiallySucceeded)
}

func convertBatchResults(job *genai.BatchJob) (*providers.BatchResult, error) {
	results := make([]providers.BatchResultItem, 0, len(job.Dest.InlinedResponses))
	for _, response := range job.Dest.InlinedResponses {
		item := providers.BatchResultItem{CustomID: response.Metadata["custom_id"]}
		switch {
		case response.Error != nil:
			item.Error = &providers.BatchResultError{Message: response.Error.Message}
			if response.Error.Code != nil {
				item.Error.Code = strconv.Itoa(int(*response.Error.Code))
			}
		case response.Response != nil:
			converted, err := convertResponse(response.Response, job.Model)
			if err != nil {
				return nil, err
			}
			item.Result = converted
		default:
			item.Error = &providers.BatchResultError{Message: "batch result has no response payload"}
		}
		results = append(results, item)
	}
	return &providers.BatchResult{Results: results}, nil
}

func convertBatch(job *genai.BatchJob) *providers.Batch {
	batch := &providers.Batch{
		ID: job.Name, Object: "batch", Endpoint: batchEndpoint, Provider: providerName,
		Status: convertBatchStatus(job.State), CompletionWindow: "24h",
	}
	if !job.CreateTime.IsZero() {
		batch.CreatedAt = job.CreateTime.Unix()
	}
	if job.DisplayName != "" {
		batch.Metadata = map[string]string{"display_name": job.DisplayName}
	}
	if job.Src != nil && job.Src.FileName != "" {
		batch.InputFileID = job.Src.FileName
	}
	if job.Dest != nil {
		batch.OutputFileID = job.Dest.FileName
	}
	if !job.StartTime.IsZero() {
		batch.InProgressAt = new(job.StartTime.Unix())
	}
	if !job.EndTime.IsZero() {
		batch.CompletedAt = new(job.EndTime.Unix())
	}
	if job.CompletionStats != nil {
		batch.RequestCounts = &providers.BatchRequestCounts{
			Completed: int(job.CompletionStats.SuccessfulCount),
			Failed: int(
				job.CompletionStats.FailedCount,
			),
			Total: int(
				job.CompletionStats.SuccessfulCount + job.CompletionStats.FailedCount + job.CompletionStats.IncompleteCount,
			),
		}
	}
	return batch
}

func validateBatchProvider(provider string) error {
	if provider != "" && provider != providerName {
		return errors.NewInvalidRequestError(
			providerName,
			fmt.Errorf("batch belongs to provider %q, not %q", providerName, provider),
		)
	}
	return nil
}

func convertBatchStatus(state genai.JobState) providers.BatchStatus {
	switch state {
	case genai.JobStateSucceeded, genai.JobStatePartiallySucceeded:
		return providers.BatchStatusCompleted
	case genai.JobStateFailed:
		return providers.BatchStatusFailed
	case genai.JobStateCancelling:
		return providers.BatchStatusCancelling
	case genai.JobStateCancelled:
		return providers.BatchStatusCancelled
	case genai.JobStateExpired:
		return providers.BatchStatusExpired
	case genai.JobStateQueued, genai.JobStatePending:
		return providers.BatchStatusValidating
	case genai.JobStateUnspecified, genai.JobStateRunning, genai.JobStatePaused, genai.JobStateUpdating:
		return providers.BatchStatusInProgress
	default:
		return providers.BatchStatusInProgress
	}
}
