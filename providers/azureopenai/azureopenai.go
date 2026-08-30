// Package azureopenai provides an Azure OpenAI provider implementation for any-llm.
package azureopenai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/policy"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/runtime"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/pagination"
	"github.com/openai/openai-go/v3/responses"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
	"github.com/mozilla-ai/any-llm-go/providers/internal/openairesponses"
	anyopenai "github.com/mozilla-ai/any-llm-go/providers/openai"
)

// Provider configuration constants.
const (
	envAPIKey            = "AZURE_OPENAI_API_KEY"
	envADToken           = "AZURE_OPENAI_AD_TOKEN"
	envBaseURL           = "AZURE_OPENAI_ENDPOINT"
	extraTokenCredential = "token_credential"
	providerName         = "azureopenai"
	previewAPIVersion    = "preview"
	v1TokenScope         = "https://ai.azure.com/.default"
)

// Ensure Provider implements the required interfaces.
var (
	_ providers.CapabilityProvider = (*Provider)(nil)
	_ providers.EmbeddingProvider  = (*Provider)(nil)
	_ providers.ErrorConverter     = (*Provider)(nil)
	_ providers.ModelLister        = (*Provider)(nil)
	_ providers.Provider           = (*Provider)(nil)
	_ providers.ResponsesProvider  = (*Provider)(nil)
)

// Provider implements the providers.Provider interface for Azure OpenAI.
// It embeds openai.CompatibleProvider so chat completions, embeddings, and
// model listing reuse the OpenAI-compatible client, but it intentionally does
// not embed openai.Provider: that avoids inheriting methods (such as the
// Responses API) that openai.Provider may gain in the future but Azure should
// not expose until explicitly enabled here.
type Provider struct {
	*anyopenai.CompatibleProvider

	client      openai.Client
	mediaClient openai.Client
}

// WithTokenCredential configures Microsoft Entra ID authentication. It cannot
// be combined with an API key supplied through options or the environment.
func WithTokenCredential(credential azcore.TokenCredential) config.Option {
	return func(cfg *config.Config) error {
		if credential == nil {
			return fmt.Errorf("token credential cannot be nil")
		}
		return config.WithExtra(extraTokenCredential, credential)(cfg)
	}
}

// New creates a new Azure OpenAI provider.
func New(opts ...config.Option) (*Provider, error) {
	cfg, err := config.New(opts...)
	if err != nil {
		return nil, fmt.Errorf("invalid options: %w", err)
	}

	auth, err := resolveAuthentication(cfg)
	if err != nil {
		return nil, err
	}

	endpoint, err := cfg.ResolveBaseURL(envBaseURL, "")
	if err != nil {
		return nil, fmt.Errorf("resolve endpoint: %w", err)
	}
	if endpoint == "" {
		return nil, fmt.Errorf(
			"%s endpoint is required (set via WithBaseURL option or %q env var)",
			providerName,
			envBaseURL,
		)
	}

	clientOptions := v1Options(endpoint, cfg.HTTPClient(), auth)
	mediaClientOptions := append(
		v1Options(endpoint, cfg.HTTPClient(), auth),
		option.WithQuery("api-version", previewAPIVersion),
	)
	base, err := anyopenai.NewCompatible(anyopenai.CompatibleConfig{
		APIKeyEnvVar:   envAPIKey,
		BaseURLEnvVar:  envBaseURL,
		Capabilities:   capabilities(),
		ClientOptions:  clientOptions,
		DefaultAPIKey:  auth.apiKey,
		DefaultBaseURL: endpoint,
		Name:           providerName,
		RequireAPIKey:  false,
		RequireBaseURL: true,
	})
	if err != nil {
		return nil, fmt.Errorf("create compatible provider: %w", err)
	}

	return &Provider{
		CompatibleProvider: base,
		client:             openai.NewClient(clientOptions...),
		mediaClient:        openai.NewClient(mediaClientOptions...),
	}, nil
}

type authentication struct {
	apiKey     string
	credential azcore.TokenCredential
}

func resolveAuthentication(cfg *config.Config) (authentication, error) {
	apiKey := cfg.ResolveAPIKey(envAPIKey)
	credential, hasCredential, err := resolveTokenCredential(cfg)
	if err != nil {
		return authentication{}, err
	}
	if !hasCredential {
		if token := cfg.ResolveEnv(envADToken); token != "" {
			// This adapter wraps a token acquired by the caller. It cannot refresh it.
			credential, hasCredential = staticTokenCredential{token: token}, true
		}
	}
	if hasCredential && apiKey != "" {
		return authentication{}, fmt.Errorf(
			"%s authentication is ambiguous: API key and token credential are both configured",
			providerName,
		)
	}
	if !hasCredential && apiKey == "" {
		return authentication{}, errors.NewMissingAPIKeyError(providerName, envAPIKey)
	}
	return authentication{apiKey: apiKey, credential: credential}, nil
}

func v1Options(endpoint string, httpClient *http.Client, auth authentication) []option.RequestOption {
	options := []option.RequestOption{
		option.WithBaseURL(v1BaseURL(endpoint)),
		option.WithHTTPClient(httpClient),
	}
	if auth.credential != nil {
		return append(options, withV1TokenCredential(auth.credential))
	}
	// openai-go's WithAPIKey sends bearer auth. Azure's preview media API
	// requires API keys in `api-key`, which the GA v1 API also accepts.
	// https://learn.microsoft.com/azure/foundry/openai/reference-preview-latest
	return append(options, option.WithHeader("api-key", auth.apiKey))
}

func v1BaseURL(endpoint string) string {
	baseURL := strings.TrimRight(endpoint, "/")
	if !strings.HasSuffix(baseURL, "/openai/v1") {
		baseURL += "/openai/v1"
	}
	return baseURL + "/"
}

type policyAdapter option.MiddlewareNext

func (adapter policyAdapter) Do(req *policy.Request) (*http.Response, error) {
	return option.MiddlewareNext(adapter)(req.Raw())
}

func withV1TokenCredential(credential azcore.TokenCredential) option.RequestOption {
	// Microsoft's v1 guide pairs option.WithBaseURL with azure.WithTokenCredential,
	// while the SDK documents token credentials together with its legacy endpoint
	// option. Apply azcore's public bearer policy to preserve the documented v1 route.
	// https://learn.microsoft.com/azure/foundry/openai/api-version-lifecycle?tabs=go
	// https://github.com/openai/openai-go/blob/v3.32.0/azure/azure.go#L110-L113
	tokenPolicy := runtime.NewBearerTokenPolicy(credential, []string{v1TokenScope}, nil)
	return option.WithMiddleware(func(req *http.Request, next option.MiddlewareNext) (*http.Response, error) {
		pipeline := runtime.NewPipeline(
			providerName,
			"v1",
			runtime.PipelineOptions{},
			&policy.ClientOptions{PerRetryPolicies: []policy.Policy{tokenPolicy, policyAdapter(next)}},
		)
		azRequest, err := runtime.NewRequestFromRequest(req)
		if err != nil {
			return nil, fmt.Errorf("create Azure authentication request: %w", err)
		}
		return pipeline.Do(azRequest)
	})
}

// capabilities returns the capabilities for Azure OpenAI.
func capabilities() providers.Capabilities {
	return providers.Capabilities{
		Completion:          true,
		CompletionImage:     true,
		CompletionPDF:       false,
		CompletionReasoning: true,
		CompletionStreaming: true,
		CompletionTools:     true,
		Embedding:           true,
		AudioSpeech:         true,
		AudioTranscription:  true,
		ImageGeneration:     true,
		ListModels:          true,
		Responses:           true,
		ResponsesStreaming:  true,
	}
}

// ListModels returns Azure model metadata using Azure's created_at field.
func (p *Provider) ListModels(ctx context.Context) (*providers.ModelsResponse, error) {
	resp, err := p.client.Models.List(ctx)
	if err != nil {
		return nil, fmt.Errorf("list models: %w", p.ConvertError(err))
	}

	models := make([]providers.Model, 0, len(resp.Data))
	for _, model := range resp.Data {
		created := model.Created
		if field, ok := model.JSON.ExtraFields["created_at"]; ok {
			if err := json.Unmarshal([]byte(field.Raw()), &created); err != nil {
				return nil, fmt.Errorf("decode created_at for model %q: %w", model.ID, err)
			}
		}
		models = append(models, providers.Model{
			ID:      model.ID,
			Object:  string(model.Object),
			Created: created,
			OwnedBy: model.OwnedBy,
		})
	}

	return &providers.ModelsResponse{Object: resp.Object, Data: models}, nil
}

// GenerateImage generates images using an Azure deployment named by params.Model.
func (p *Provider) GenerateImage(
	ctx context.Context,
	params openai.ImageGenerateParams,
) (*openai.ImagesResponse, error) {
	if params.Prompt == "" {
		return nil, invalid("prompt is required")
	}
	resp, err := p.mediaClient.Images.Generate(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("generate image: %w", p.ConvertError(err))
	}
	return resp, nil
}

// Responses implements the provider-neutral Responses interface. Azure's v1
// API uses the official OpenAI SDK route and supports the same streaming form:
// https://learn.microsoft.com/azure/foundry/openai/how-to/responses
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

// CreateResponse creates a response using the resource-level Responses endpoint.
func (p *Provider) CreateResponse(
	ctx context.Context,
	params responses.ResponseNewParams,
) (*responses.Response, error) {
	if params.Model == "" {
		return nil, invalid("model is required")
	}
	resp, err := p.client.Responses.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("create response: %w", p.ConvertError(err))
	}
	return resp, nil
}

// StreamResponse creates a streaming response.
func (p *Provider) StreamResponse(
	ctx context.Context,
	params responses.ResponseNewParams,
) (<-chan responses.ResponseStreamEventUnion, <-chan error) {
	return p.streamResponse(ctx, func() error {
		if params.Model == "" {
			return invalid("model is required")
		}
		return nil
	}, func() responseStream { return p.client.Responses.NewStreaming(ctx, params) })
}

// RetrieveResponse retrieves a stored response.
func (p *Provider) RetrieveResponse(
	ctx context.Context,
	responseID string,
	params responses.ResponseGetParams,
) (*responses.Response, error) {
	if responseID == "" {
		return nil, invalid("response ID is required")
	}
	resp, err := p.client.Responses.Get(ctx, responseID, params)
	if err != nil {
		return nil, fmt.Errorf("retrieve response: %w", p.ConvertError(err))
	}
	return resp, nil
}

// DeleteResponse deletes a stored response.
func (p *Provider) DeleteResponse(ctx context.Context, responseID string) error {
	if responseID == "" {
		return invalid("response ID is required")
	}
	if err := p.client.Responses.Delete(ctx, responseID); err != nil {
		return fmt.Errorf("delete response: %w", p.ConvertError(err))
	}
	return nil
}

// CancelResponse cancels a stored background response.
func (p *Provider) CancelResponse(ctx context.Context, responseID string) (*responses.Response, error) {
	if responseID == "" {
		return nil, invalid("response ID is required")
	}
	resp, err := p.client.Responses.Cancel(ctx, responseID)
	if err != nil {
		return nil, fmt.Errorf("cancel response: %w", p.ConvertError(err))
	}
	return resp, nil
}

// ListResponseInputItems lists the input items for a stored response.
func (p *Provider) ListResponseInputItems(
	ctx context.Context,
	responseID string,
	params responses.InputItemListParams,
) (*pagination.CursorPage[responses.ResponseItemUnion], error) {
	if responseID == "" {
		return nil, invalid("response ID is required")
	}
	page, err := p.client.Responses.InputItems.List(ctx, responseID, params)
	if err != nil {
		return nil, fmt.Errorf("list response input items: %w", p.ConvertError(err))
	}
	return page, nil
}

// ReplayResponse replays the event stream for a stored response.
func (p *Provider) ReplayResponse(
	ctx context.Context,
	responseID string,
	params responses.ResponseGetParams,
) (<-chan responses.ResponseStreamEventUnion, <-chan error) {
	return p.streamResponse(ctx, func() error {
		if responseID == "" {
			return invalid("response ID is required")
		}
		return nil
	}, func() responseStream { return p.client.Responses.GetStreaming(ctx, responseID, params) })
}

// Speech generates speech using an Azure deployment named by params.Model.
func (p *Provider) Speech(ctx context.Context, params openai.AudioSpeechNewParams) (*http.Response, error) {
	if params.Input == "" {
		return nil, invalid("input is required")
	}
	if params.Model == "" {
		return nil, invalid("model is required")
	}
	if params.Voice.OfString.Value == "" &&
		params.Voice.OfAudioSpeechNewsVoiceString2.Value == "" &&
		params.Voice.OfAudioSpeechNewsVoiceID == nil {
		return nil, invalid("voice is required")
	}
	resp, err := p.mediaClient.Audio.Speech.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("generate speech: %w", p.ConvertError(err))
	}
	return resp, nil
}

// Transcribe transcribes audio using an Azure deployment named by params.Model.
func (p *Provider) Transcribe(
	ctx context.Context,
	params openai.AudioTranscriptionNewParams,
) (*openai.Transcription, error) {
	if params.File == nil {
		return nil, invalid("file is required")
	}
	if params.Model == "" {
		return nil, invalid("model is required")
	}
	resp, err := p.mediaClient.Audio.Transcriptions.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("transcribe audio: %w", p.ConvertError(err))
	}
	transcription := resp.AsTranscription()
	return &transcription, nil
}

type responseStream interface {
	Next() bool
	Current() responses.ResponseStreamEventUnion
	Err() error
	Close() error
}

func (p *Provider) streamResponse(
	ctx context.Context,
	validate func() error,
	newStream func() responseStream,
) (<-chan responses.ResponseStreamEventUnion, <-chan error) {
	events := make(chan responses.ResponseStreamEventUnion)
	errs := make(chan error, 1)
	go func() {
		defer close(events)
		defer close(errs)
		if err := validate(); err != nil {
			errs <- err
			return
		}
		stream := newStream()
		defer func() {
			if err := stream.Close(); err != nil {
				select {
				case errs <- fmt.Errorf("close response stream: %w", p.ConvertError(err)):
				default:
				}
			}
		}()
		for stream.Next() {
			select {
			case events <- stream.Current():
			case <-ctx.Done():
				errs <- ctx.Err()
				return
			}
		}
		if err := ctx.Err(); err != nil {
			errs <- err
			return
		}
		if err := stream.Err(); err != nil {
			errs <- fmt.Errorf("read response stream: %w", p.ConvertError(err))
		}
	}()
	return events, errs
}

func invalid(message string) error {
	return errors.NewInvalidRequestError(providerName, fmt.Errorf("%s", message))
}

func resolveTokenCredential(cfg *config.Config) (azcore.TokenCredential, bool, error) {
	value, ok := cfg.ExtraValue(extraTokenCredential)
	if !ok {
		return nil, false, nil
	}
	credential, ok := value.(azcore.TokenCredential)
	if !ok || credential == nil {
		return nil, false, fmt.Errorf("%s %q must implement azcore.TokenCredential", providerName, extraTokenCredential)
	}
	return credential, true, nil
}

type staticTokenCredential struct {
	token string
}

func (c staticTokenCredential) GetToken(context.Context, policy.TokenRequestOptions) (azcore.AccessToken, error) {
	// Microsoft requires clients to treat access tokens as opaque, so this static
	// adapter cannot derive expiry or refresh the token: https://learn.microsoft.com/entra/identity-platform/access-tokens
	return azcore.AccessToken{Token: c.token, ExpiresOn: time.Now().Add(time.Hour)}, nil
}
