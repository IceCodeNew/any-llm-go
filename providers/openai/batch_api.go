package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	stderrors "errors"
	"fmt"

	openaisdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/pagination"

	"github.com/mozilla-ai/any-llm-go/providers"
)

const (
	defaultBatchEndpoint       = "/v1/chat/completions"
	initialBatchScanBufferSize = 64 * 1024
	maxBatchScanBufferSize     = 16 * 1024 * 1024
)

// CreateOpenAIBatch creates a batch using official SDK parameters.
func (p *Provider) CreateOpenAIBatch(
	ctx context.Context,
	params openaisdk.BatchNewParams,
) (*openaisdk.Batch, error) {
	if params.CompletionWindow == "" || params.Endpoint == "" || params.InputFileID == "" {
		return nil, invalid("batch completion_window, endpoint, and input_file_id are required")
	}
	result, err := p.client.Batches.New(ctx, params)
	if err != nil {
		return nil, p.ConvertError(err)
	}
	return result, nil
}

// RetrieveOpenAIBatch retrieves an official SDK batch by ID.
func (p *Provider) RetrieveOpenAIBatch(ctx context.Context, id string) (*openaisdk.Batch, error) {
	if id == "" {
		return nil, invalid("batch ID is required")
	}
	result, err := p.client.Batches.Get(ctx, id)
	if err != nil {
		return nil, p.ConvertError(err)
	}
	return result, nil
}

// CancelOpenAIBatch cancels an official SDK batch by ID.
func (p *Provider) CancelOpenAIBatch(ctx context.Context, id string) (*openaisdk.Batch, error) {
	if id == "" {
		return nil, invalid("batch ID is required")
	}
	result, err := p.client.Batches.Cancel(ctx, id)
	if err != nil {
		return nil, p.ConvertError(err)
	}
	return result, nil
}

// ListOpenAIBatches lists official SDK batches.
func (p *Provider) ListOpenAIBatches(
	ctx context.Context,
	params openaisdk.BatchListParams,
) (*pagination.CursorPage[openaisdk.Batch], error) {
	result, err := p.client.Batches.List(ctx, params)
	if err != nil {
		return nil, p.ConvertError(err)
	}
	return result, nil
}

// CreateBatch creates a normalized batch from in-memory requests.
func (p *Provider) CreateBatch(
	ctx context.Context,
	params providers.CreateBatchParams,
) (*providers.Batch, error) {
	normalized, err := normalizeBatch(params)
	if err != nil {
		return nil, err
	}
	file, err := p.UploadFile(
		ctx,
		openaisdk.FileNewParams{
			File:    bytes.NewReader(normalized.jsonl),
			Purpose: openaisdk.FilePurposeBatch,
		},
	)
	if err != nil {
		return nil, err
	}
	batch, err := p.CreateOpenAIBatch(
		ctx,
		openaisdk.BatchNewParams{
			CompletionWindow: openaisdk.BatchNewParamsCompletionWindow(normalized.completionWindow),
			Endpoint:         openaisdk.BatchNewParamsEndpoint(normalized.endpoint),
			InputFileID:      file.ID,
			Metadata:         params.Metadata,
		},
	)
	if err != nil {
		_, cleanupErr := p.DeleteFile(context.WithoutCancel(ctx), file.ID)
		return nil, stderrors.Join(err, cleanupErr)
	}
	return convertBatch(batch), nil
}

// RetrieveBatch retrieves a normalized batch by ID.
func (p *Provider) RetrieveBatch(
	ctx context.Context,
	batchID, provider string,
) (*providers.Batch, error) {
	if err := validateBatchIdentity(batchID, provider); err != nil {
		return nil, err
	}
	batch, err := p.RetrieveOpenAIBatch(ctx, batchID)
	if err != nil {
		return nil, err
	}
	return convertBatch(batch), nil
}

// CancelBatch cancels a normalized batch by ID.
func (p *Provider) CancelBatch(
	ctx context.Context,
	batchID, provider string,
) (*providers.Batch, error) {
	if err := validateBatchIdentity(batchID, provider); err != nil {
		return nil, err
	}
	batch, err := p.CancelOpenAIBatch(ctx, batchID)
	if err != nil {
		return nil, err
	}
	return convertBatch(batch), nil
}

// ListBatches lists normalized batches.
func (p *Provider) ListBatches(
	ctx context.Context,
	provider string,
	opts providers.ListBatchesOptions,
) ([]providers.Batch, error) {
	query, err := batchListQuery(provider, opts)
	if err != nil {
		return nil, err
	}
	pager := p.client.Batches.ListAutoPaging(ctx, query)
	result := []providers.Batch{}
	for pager.Next() {
		batch := pager.Current()
		result = append(result, *convertBatch(&batch))
		if opts.Limit != nil && len(result) == *opts.Limit {
			break
		}
	}
	if err := pager.Err(); err != nil {
		return nil, p.ConvertError(err)
	}
	return result, nil
}

// RetrieveBatchResults downloads and parses a completed batch's JSONL output.
func (p *Provider) RetrieveBatchResults(
	ctx context.Context,
	batchID, provider string,
) (_ *providers.BatchResult, err error) {
	batch, err := p.RetrieveBatch(ctx, batchID, provider)
	if err != nil {
		return nil, err
	}
	if batch.Status != providers.BatchStatusCompleted || batch.OutputFileID == "" {
		return nil, invalid("batch is not complete")
	}
	response, err := p.FileContent(ctx, batch.OutputFileID)
	if err != nil {
		return nil, err
	}
	defer func() {
		err = stderrors.Join(err, response.Body.Close())
	}()
	result := &providers.BatchResult{}
	scanner := bufio.NewScanner(response.Body)
	scanner.Buffer(make([]byte, initialBatchScanBufferSize), maxBatchScanBufferSize)
	for scanner.Scan() {
		item, parseErr := parseBatchOutputItem(scanner.Bytes(), batch.Endpoint)
		if parseErr != nil {
			return nil, parseErr
		}
		result.Results = append(result.Results, item)
	}
	if err := scanner.Err(); err != nil {
		return nil, p.ConvertError(err)
	}
	return result, nil
}

type normalizedBatch struct {
	endpoint         string
	completionWindow string
	jsonl            []byte
}

func normalizeBatch(params providers.CreateBatchParams) (normalizedBatch, error) {
	if params.Model == "" || len(params.Requests) == 0 {
		return normalizedBatch{}, invalid("batch model and requests are required")
	}
	endpoint := params.Endpoint
	if endpoint == "" {
		endpoint = defaultBatchEndpoint
	}
	if !isSupportedBatchEndpoint(endpoint) {
		return normalizedBatch{}, invalid("unsupported batch endpoint")
	}
	window := params.CompletionWindow
	if window == "" {
		window = "24h"
	}
	if window != "24h" {
		return normalizedBatch{}, invalid("completion_window must be 24h")
	}
	jsonl, err := buildBatchJSONL(params.Model, endpoint, params.Requests)
	if err != nil {
		return normalizedBatch{}, err
	}
	return normalizedBatch{endpoint: endpoint, completionWindow: window, jsonl: jsonl}, nil
}

func buildBatchJSONL(model, endpoint string, requests []providers.BatchRequestItem) ([]byte, error) {
	var data bytes.Buffer
	encoder := json.NewEncoder(&data)
	customIDs := make(map[string]struct{}, len(requests))
	for _, request := range requests {
		if request.CustomID == "" {
			return nil, invalid("batch custom_id is required")
		}
		if _, exists := customIDs[request.CustomID]; exists {
			return nil, invalid("batch custom_id must be unique")
		}
		customIDs[request.CustomID] = struct{}{}
		body, err := normalizeBatchRequest(model, request.Body)
		if err != nil {
			return nil, err
		}
		line := map[string]any{"custom_id": request.CustomID, "method": "POST", "url": endpoint, "body": body}
		if err := encoder.Encode(line); err != nil {
			return nil, invalid("batch request cannot be encoded")
		}
	}
	return data.Bytes(), nil
}

func normalizeBatchRequest(
	model string,
	requestBody map[string]any,
) (map[string]any, error) {
	body := make(map[string]any)

	if requestBody != nil {
		encoded, err := json.Marshal(requestBody)
		if err != nil {
			return nil, invalid("batch request cannot be encoded")
		}

		err = json.Unmarshal(encoded, &body)
		if err != nil {
			return nil, invalid("batch request cannot be encoded")
		}
	}

	if requestModel, exists := body["model"]; exists {
		modelString, valid := requestModel.(string)
		if !valid || modelString != model {
			return nil, invalid("batch request model must match batch model")
		}
	} else {
		body["model"] = model
	}

	return body, nil
}

func batchListQuery(provider string, opts providers.ListBatchesOptions) (openaisdk.BatchListParams, error) {
	if provider != "" && provider != providerName {
		return openaisdk.BatchListParams{}, invalid("provider must be openai")
	}
	if opts.Limit != nil && (*opts.Limit < 1 || *opts.Limit > 100) {
		return openaisdk.BatchListParams{}, invalid("limit must be between 1 and 100")
	}
	query := openaisdk.BatchListParams{}
	if opts.After != "" {
		query.After = openaisdk.String(opts.After)
	}
	if opts.Limit != nil {
		query.Limit = openaisdk.Int(int64(*opts.Limit))
	}
	return query, nil
}

type batchOutputLine struct {
	CustomID string `json:"custom_id"`
	Response *struct {
		StatusCode int             `json:"status_code"`
		Body       json.RawMessage `json:"body"`
	} `json:"response"`
	Error *providers.BatchResultError `json:"error"`
}

func parseBatchOutputItem(data []byte, endpoint string) (providers.BatchResultItem, error) {
	var line batchOutputLine
	if err := json.Unmarshal(data, &line); err != nil {
		return providers.BatchResultItem{}, invalid("invalid batch output JSONL")
	}
	item := providers.BatchResultItem{CustomID: line.CustomID}
	switch {
	case line.Error != nil:
		item.Error = line.Error
	case line.Response == nil:
		item.Error = batchOutputError("malformed_response", "batch output has neither response nor error")
	case line.Response.StatusCode < 200 || line.Response.StatusCode >= 300:
		item.Raw = bytes.Clone(line.Response.Body)
		item.Error = parseBatchHTTPError(line.Response.StatusCode, line.Response.Body)
	case endpoint == defaultBatchEndpoint:
		parseChatBatchOutput(&item, line.Response.Body)
	case len(line.Response.Body) == 0:
		item.Error = batchOutputError("malformed_response", "batch response body is missing")
	default:
		item.Raw = bytes.Clone(line.Response.Body)
	}
	return item, nil
}

func parseBatchHTTPError(statusCode int, body json.RawMessage) *providers.BatchResultError {
	var envelope struct {
		Error *providers.BatchResultError `json:"error"`
	}
	if json.Unmarshal(body, &envelope) == nil && envelope.Error != nil &&
		(envelope.Error.Code != "" || envelope.Error.Message != "") {
		return envelope.Error
	}
	return batchOutputError("http_error", fmt.Sprintf("batch request returned HTTP %d", statusCode))
}

func parseChatBatchOutput(item *providers.BatchResultItem, body json.RawMessage) {
	var completion openaisdk.ChatCompletion
	if err := json.Unmarshal(body, &completion); err != nil || completion.ID == "" {
		item.Error = batchOutputError(
			"malformed_response",
			"batch response body is not a chat completion",
		)
		return
	}
	item.Result = convertResponse(&completion, providerName)
}

func batchOutputError(code, message string) *providers.BatchResultError {
	return &providers.BatchResultError{Code: code, Message: message}
}

func isSupportedBatchEndpoint(endpoint string) bool {
	switch endpoint {
	case "/v1/responses",
		defaultBatchEndpoint,
		"/v1/embeddings",
		"/v1/completions",
		"/v1/moderations",
		"/v1/images/generations",
		"/v1/images/edits",
		"/v1/videos":
		return true
	default:
		return false
	}
}

func validateBatchIdentity(id, provider string) error {
	if id == "" {
		return invalid("batch ID is required")
	}
	if provider != "" && provider != providerName {
		return invalid("provider must be openai")
	}
	return nil
}

func convertBatch(batch *openaisdk.Batch) *providers.Batch {
	result := &providers.Batch{
		ID:               batch.ID,
		Object:           string(batch.Object),
		Endpoint:         batch.Endpoint,
		InputFileID:      batch.InputFileID,
		OutputFileID:     batch.OutputFileID,
		ErrorFileID:      batch.ErrorFileID,
		CompletionWindow: batch.CompletionWindow,
		CreatedAt:        batch.CreatedAt,
		Metadata:         batch.Metadata,
		Provider:         providerName,
		Status:           providers.BatchStatus(batch.Status),
	}
	if batch.JSON.CompletedAt.Valid() {
		result.CompletedAt = &batch.CompletedAt
	}
	if batch.JSON.InProgressAt.Valid() {
		result.InProgressAt = &batch.InProgressAt
	}
	if batch.JSON.RequestCounts.Valid() {
		result.RequestCounts = &providers.BatchRequestCounts{
			Completed: int(batch.RequestCounts.Completed),
			Failed:    int(batch.RequestCounts.Failed),
			Total:     int(batch.RequestCounts.Total),
		}
	}
	return result
}
