package openai

import (
	"context"
	"encoding/json"
	"net/http"

	openaisdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/pagination"
	"github.com/openai/openai-go/v3/packages/param"

	"github.com/mozilla-ai/any-llm-go/providers"
)

// Moderate submits lossless official SDK moderation parameters.
func (p *Provider) Moderate(
	ctx context.Context,
	params openaisdk.ModerationNewParams,
) (*openaisdk.ModerationNewResponse, error) {
	result, err := p.client.Moderations.New(ctx, params)
	if err != nil {
		return nil, p.ConvertError(err)
	}
	return result, nil
}

// Moderation implements the provider-wide moderation interface. Moderate is
// preferred for lossless access to the official SDK response.
func (p *Provider) Moderation(
	ctx context.Context,
	params providers.ModerationParams,
) (*providers.ModerationResponse, error) {
	sdkParams, err := convertModerationParams(params)
	if err != nil {
		return nil, err
	}
	result, err := p.Moderate(ctx, sdkParams)
	if err != nil {
		return nil, err
	}
	normalized := &providers.ModerationResponse{
		ID:      result.ID,
		Model:   result.Model,
		Results: make([]providers.ModerationResult, 0, len(result.Results)),
	}
	for _, item := range result.Results {
		n := convertModerationResult(item)
		if params.IncludeRaw {
			n.ProviderRaw = json.RawMessage(item.RawJSON())
		}
		normalized.Results = append(normalized.Results, n)
	}
	return normalized, nil
}

// GenerateImage generates images using official SDK parameters.
func (p *Provider) GenerateImage(
	ctx context.Context,
	params openaisdk.ImageGenerateParams,
) (*openaisdk.ImagesResponse, error) {
	if params.Prompt == "" {
		return nil, invalid("image prompt is required")
	}
	result, err := p.client.Images.Generate(ctx, params)
	if err != nil {
		return nil, p.ConvertError(err)
	}
	return result, nil
}

// Transcribe transcribes audio using official SDK parameters.
func (p *Provider) Transcribe(
	ctx context.Context,
	params openaisdk.AudioTranscriptionNewParams,
) (*openaisdk.AudioTranscriptionNewResponseUnion, error) {
	if params.File == nil || params.Model == "" {
		return nil, invalid("transcription file and model are required")
	}
	result, err := p.client.Audio.Transcriptions.New(ctx, params)
	if err != nil {
		return nil, p.ConvertError(err)
	}
	return result, nil
}

// Speech generates audio and returns a response whose body the caller must close.
func (p *Provider) Speech(
	ctx context.Context,
	params openaisdk.AudioSpeechNewParams,
) (*http.Response, error) {
	if params.Input == "" || params.Model == "" || param.IsOmitted(params.Voice) {
		return nil, invalid("speech input, model, and voice are required")
	}
	result, err := p.client.Audio.Speech.New(ctx, params)
	if err != nil {
		return nil, p.ConvertError(err)
	}
	return result, nil
}

// UploadFile uploads a file using official SDK parameters.
func (p *Provider) UploadFile(
	ctx context.Context,
	params openaisdk.FileNewParams,
) (*openaisdk.FileObject, error) {
	if params.File == nil || params.Purpose == "" {
		return nil, invalid("file and purpose are required")
	}
	result, err := p.client.Files.New(ctx, params)
	if err != nil {
		return nil, p.ConvertError(err)
	}
	return result, nil
}

// RetrieveFile retrieves a file by ID.
func (p *Provider) RetrieveFile(ctx context.Context, id string) (*openaisdk.FileObject, error) {
	if id == "" {
		return nil, invalid("file ID is required")
	}
	result, err := p.client.Files.Get(ctx, id)
	if err != nil {
		return nil, p.ConvertError(err)
	}
	return result, nil
}

// DeleteFile deletes a file by ID.
func (p *Provider) DeleteFile(ctx context.Context, id string) (*openaisdk.FileDeleted, error) {
	if id == "" {
		return nil, invalid("file ID is required")
	}
	result, err := p.client.Files.Delete(ctx, id)
	if err != nil {
		return nil, p.ConvertError(err)
	}
	return result, nil
}

// ListFiles lists uploaded files.
func (p *Provider) ListFiles(
	ctx context.Context,
	params openaisdk.FileListParams,
) (*pagination.CursorPage[openaisdk.FileObject], error) {
	result, err := p.client.Files.List(ctx, params)
	if err != nil {
		return nil, p.ConvertError(err)
	}
	return result, nil
}

// FileContent returns file content in a response whose body the caller must close.
func (p *Provider) FileContent(ctx context.Context, id string) (*http.Response, error) {
	if id == "" {
		return nil, invalid("file ID is required")
	}
	result, err := p.client.Files.Content(ctx, id)
	if err != nil {
		return nil, p.ConvertError(err)
	}
	return result, nil
}

func convertModerationParams(
	params providers.ModerationParams,
) (openaisdk.ModerationNewParams, error) {
	result := openaisdk.ModerationNewParams{Model: params.Model}
	switch input := params.Input.(type) {
	case string:
		if input == "" {
			return result, invalid("moderation input is required")
		}
		result.Input.OfString = openaisdk.String(input)
	case []string:
		if len(input) == 0 {
			return result, invalid("moderation input is required")
		}
		result.Input.OfStringArray = input
	case []providers.ContentPart:
		if len(input) == 0 {
			return result, invalid("moderation input is required")
		}
		for _, part := range input {
			converted, err := convertModerationContentPart(part)
			if err != nil {
				return result, err
			}
			result.Input.OfModerationMultiModalArray = append(
				result.Input.OfModerationMultiModalArray,
				converted,
			)
		}
	default:
		return result, invalid("moderation input must be a string, []string, or []ContentPart")
	}
	return result, nil
}

func convertModerationContentPart(
	part providers.ContentPart,
) (openaisdk.ModerationMultiModalInputUnionParam, error) {
	switch part.Type {
	case contentTypeText:
		return openaisdk.ModerationMultiModalInputParamOfText(part.Text), nil
	case contentTypeImageURL:
		if part.ImageURL == nil || part.ImageURL.URL == "" {
			return openaisdk.ModerationMultiModalInputUnionParam{}, invalid(
				"moderation image_url is required",
			)
		}
		return openaisdk.ModerationMultiModalInputParamOfImageURL(
			openaisdk.ModerationImageURLInputImageURLParam{URL: part.ImageURL.URL},
		), nil
	default:
		return openaisdk.ModerationMultiModalInputUnionParam{}, invalid(
			"unsupported moderation content type",
		)
	}
}

func convertModerationResult(item openaisdk.Moderation) providers.ModerationResult {
	c, s, a := item.Categories, item.CategoryScores, item.CategoryAppliedInputTypes
	return providers.ModerationResult{
		Flagged: item.Flagged,
		Categories: map[string]bool{
			"harassment": c.Harassment, "harassment/threatening": c.HarassmentThreatening,
			"hate": c.Hate, "hate/threatening": c.HateThreatening,
			"illicit": c.Illicit, "illicit/violent": c.IllicitViolent,
			"self-harm": c.SelfHarm, "self-harm/instructions": c.SelfHarmInstructions,
			"self-harm/intent": c.SelfHarmIntent, "sexual": c.Sexual,
			"sexual/minors": c.SexualMinors, "violence": c.Violence,
			"violence/graphic": c.ViolenceGraphic,
		},
		CategoryScores: map[string]float64{
			"harassment": s.Harassment, "harassment/threatening": s.HarassmentThreatening,
			"hate": s.Hate, "hate/threatening": s.HateThreatening,
			"illicit": s.Illicit, "illicit/violent": s.IllicitViolent,
			"self-harm": s.SelfHarm, "self-harm/instructions": s.SelfHarmInstructions,
			"self-harm/intent": s.SelfHarmIntent, "sexual": s.Sexual,
			"sexual/minors": s.SexualMinors, "violence": s.Violence,
			"violence/graphic": s.ViolenceGraphic,
		},
		CategoryAppliedInputTypes: map[string][]string{
			"harassment": a.Harassment, "harassment/threatening": a.HarassmentThreatening,
			"hate": a.Hate, "hate/threatening": a.HateThreatening,
			"illicit": a.Illicit, "illicit/violent": a.IllicitViolent,
			"self-harm": a.SelfHarm, "self-harm/instructions": a.SelfHarmInstructions,
			"self-harm/intent": a.SelfHarmIntent, "sexual": a.Sexual,
			"sexual/minors": a.SexualMinors, "violence": a.Violence,
			"violence/graphic": a.ViolenceGraphic,
		},
	}
}
