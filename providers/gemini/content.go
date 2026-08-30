package gemini

import (
	"encoding/base64"
	"encoding/json"
	stderrors "errors"
	"fmt"
	"math"
	"mime"
	"net/url"
	"path/filepath"
	"reflect"
	"strings"

	"google.golang.org/genai"

	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

var errUnsupportedContentPartType = stderrors.New("unsupported content part type")

func convertAssistantMessage(msg providers.Message) (*genai.Content, error) {
	if parts, ok, err := responsePartsFromExtra(msg.Extra); err != nil {
		return nil, err
	} else if ok {
		if err := validateResponseParts(msg, parts); err != nil {
			return nil, err
		}
		return &genai.Content{Role: roleModel, Parts: parts}, nil
	}

	var parts []*genai.Part

	if msg.IsMultiModal() {
		converted, err := convertContentParts(msg.ContentParts())
		if err != nil {
			return nil, err
		}

		parts = append(parts, converted...)
		if len(parts) > 0 {
			parts[0].ThoughtSignature = thoughtSignatureFromExtra(msg.Extra)
		}
	} else if msg.Content != nil && (msg.ContentString() != "" || len(msg.ToolCalls) == 0) {
		parts = append(parts, &genai.Part{
			Text:             msg.ContentString(),
			ThoughtSignature: thoughtSignatureFromExtra(msg.Extra),
		})
	}

	for i, tc := range msg.ToolCalls {
		args := map[string]any{}
		if tc.Function.Arguments != "" {
			_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)
		}

		part := genai.NewPartFromFunctionCall(tc.Function.Name, args)
		part.FunctionCall.ID = tc.ID

		// Replay the thought signature that Gemini returned on the original turn.
		// Thinking models (2.5+) require this on every function call part when
		// replaying conversation history; omitting it causes a 400 error.
		// If no real signature was captured, use the documented bypass value.
		if sig := thoughtSignatureFromExtra(tc.Extra); sig != nil {
			part.ThoughtSignature = sig
		} else if i == 0 {
			part.ThoughtSignature = []byte(thoughtSignatureBypass)
		}

		parts = append(parts, part)
	}

	if len(parts) == 0 && msg.Content == nil {
		// An empty normalized assistant turn is intentionally omitted.
		//nolint:nilnil
		return nil, nil
	}

	return &genai.Content{
		Role:  roleModel,
		Parts: parts,
	}, nil
}

func responsePartsFromExtra(extra map[string]providers.ProviderData) ([]*genai.Part, bool, error) {
	geminiData, ok := extra[providerName]
	if !ok {
		return nil, false, nil
	}
	value, ok := geminiData[extraKeyResponseParts]
	if !ok {
		return nil, false, nil
	}

	raw, err := json.Marshal(value)
	if err != nil {
		return nil, false, fmt.Errorf("serializing Gemini response part metadata: %w", err)
	}
	var parts []*genai.Part
	if err := json.Unmarshal(raw, &parts); err != nil {
		return nil, false, fmt.Errorf("decoding Gemini response part metadata: %w", err)
	}
	return parts, true, nil
}

func validateResponseParts(msg providers.Message, parts []*genai.Part) error {
	content, reasoning, toolCalls, _, err := extractResponseContent(&genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{{Content: &genai.Content{Parts: parts}}},
	})
	if err != nil {
		return err
	}
	if len(toolCalls) == len(msg.ToolCalls) {
		toolIndex := 0
		for _, part := range parts {
			if part.FunctionCall == nil {
				continue
			}
			if part.FunctionCall.ID == "" {
				toolCalls[toolIndex].ID = msg.ToolCalls[toolIndex].ID
			}
			toolIndex++
		}
	}
	if msg.ContentString() != content || !reflect.DeepEqual(msg.Reasoning, reasoning) ||
		!reflect.DeepEqual(msg.ToolCalls, toolCalls) {
		return errors.NewInvalidRequestError(
			providerName,
			fmt.Errorf("gemini response_parts metadata conflicts with normalized message fields"),
		)
	}
	return nil
}

func hasThoughtSignatureBypass(contents []*genai.Content) bool {
	for _, content := range contents {
		for _, part := range content.Parts {
			if string(part.ThoughtSignature) == thoughtSignatureBypass {
				return true
			}
		}
	}
	return false
}

func isEncodedThoughtSignatureBypass(value any, encoded string) bool {
	switch signature := value.(type) {
	case string:
		return signature == encoded
	case []byte:
		return string(signature) == thoughtSignatureBypass
	default:
		return false
	}
}

func rewriteThoughtSignatureBypass(body map[string]any) map[string]any {
	// Google documents a literal bypass value, but go-genai models
	// ThoughtSignature as []byte and base64-encodes it. Use the SDK's public
	// request hook until https://github.com/googleapis/go-genai/issues/711 is fixed.
	// https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures#faqs
	encoded := base64.StdEncoding.EncodeToString([]byte(thoughtSignatureBypass))
	var rewrite func(any)
	rewrite = func(value any) {
		switch node := value.(type) {
		case map[string]any:
			for key, child := range node {
				if key == "thoughtSignature" && isEncodedThoughtSignatureBypass(child, encoded) {
					node[key] = thoughtSignatureBypass
					continue
				}
				rewrite(child)
			}
		case []any:
			for _, child := range node {
				rewrite(child)
			}
		case []map[string]any:
			for _, child := range node {
				rewrite(child)
			}
		}
	}
	rewrite(body)
	return body
}

func convertEmbeddingInputs(input any) ([]*genai.Content, error) {
	switch v := input.(type) {
	case string:
		return []*genai.Content{genai.NewContentFromText(v, roleUser)}, nil
	case []string:
		if len(v) == 0 {
			return nil, errors.NewInvalidRequestError(providerName, fmt.Errorf("embedding input must not be empty"))
		}
		contents := make([]*genai.Content, len(v))
		for i, s := range v {
			contents[i] = genai.NewContentFromText(s, roleUser)
		}
		return contents, nil
	default:
		return nil, errors.NewInvalidRequestError(
			providerName,
			fmt.Errorf("embedding input must be a string or []string"),
		)
	}
}

func convertFinishReason(reason genai.FinishReason) string {
	switch reason {
	case genai.FinishReasonStop:
		return providers.FinishReasonStop
	case genai.FinishReasonMaxTokens:
		return providers.FinishReasonLength
	case genai.FinishReasonTooManyToolCalls:
		return providers.FinishReasonToolCalls
	case genai.FinishReasonSafety, genai.FinishReasonBlocklist, genai.FinishReasonProhibitedContent,
		genai.FinishReasonSPII, genai.FinishReasonImageSafety, genai.FinishReasonRecitation,
		genai.FinishReasonLanguage, genai.FinishReasonMalformedFunctionCall,
		genai.FinishReasonUnexpectedToolCall, genai.FinishReasonImageProhibitedContent,
		genai.FinishReasonImageRecitation:
		return providers.FinishReasonContentFilter
	case genai.FinishReasonUnspecified, genai.FinishReasonOther, genai.FinishReasonNoImage,
		genai.FinishReasonImageOther:
		return providers.FinishReasonStop
	default:
		return providers.FinishReasonStop
	}
}

func convertFunctionCallToToolCall(fc *genai.FunctionCall) (providers.ToolCall, error) {
	argsJSON := ""
	if fc.Args != nil {
		if b, err := json.Marshal(fc.Args); err == nil {
			argsJSON = string(b)
		}
	}

	id := fc.ID
	if id == "" {
		var err error
		id, err = generateID(idPrefixToolCall)
		if err != nil {
			return providers.ToolCall{}, err
		}
	}

	return providers.ToolCall{
		ID:   id,
		Type: toolCallType,
		Function: providers.FunctionCall{
			Name:      fc.Name,
			Arguments: argsJSON,
		},
	}, nil
}

func convertImagePart(img *providers.ImageURL) (*genai.Part, error) {
	if img == nil || img.URL == "" {
		return nil, errors.NewInvalidRequestError(providerName, fmt.Errorf("image_url.url is required"))
	}
	return convertURIToPart(img.URL, defaultImageMIMEType)
}

func convertFilePart(file *providers.File) (*genai.Part, error) {
	if file == nil || file.FileData == "" {
		return nil, errors.NewInvalidRequestError(providerName, fmt.Errorf("file.file_data is required"))
	}
	return convertURIToPart(file.FileData, defaultFileMIMEType)
}

func convertURIToPart(value, fallbackMIME string) (*genai.Part, error) {
	if strings.HasPrefix(value, "data:") {
		return convertDataURIToPart(value)
	}

	parsed, err := url.Parse(value)
	if err != nil || parsed.Scheme == "" {
		return nil, errors.NewInvalidRequestError(providerName, fmt.Errorf("file URI must be absolute"))
	}
	mediaType := mime.TypeByExtension(filepath.Ext(parsed.Path))
	if mediaType == "" {
		mediaType = fallbackMIME
	}
	return genai.NewPartFromURI(value, mediaType), nil
}

func convertDataURIToPart(value string) (*genai.Part, error) {
	header, encoded, ok := strings.Cut(value, ",")
	if !ok || !strings.HasSuffix(header, ";base64") {
		return nil, errors.NewInvalidRequestError(providerName, fmt.Errorf("content must be a base64-encoded data URI"))
	}
	mediaType := strings.TrimSuffix(strings.TrimPrefix(header, "data:"), ";base64")
	if mediaType == "" || encoded == "" {
		return nil, errors.NewInvalidRequestError(
			providerName,
			fmt.Errorf("data URI must include a MIME type and data"),
		)
	}
	data, err := base64.StdEncoding.Strict().DecodeString(encoded)
	if err != nil {
		return nil, errors.NewInvalidRequestError(providerName, fmt.Errorf("invalid base64 data: %w", err))
	}
	if len(data) > inlineSizeLimit {
		return nil, errors.NewInvalidRequestError(providerName, fmt.Errorf("inline content exceeds 20 MB limit"))
	}

	return genai.NewPartFromBytes(data, mediaType), nil
}

func int32Value(value int, field string) (int32, error) {
	if value < math.MinInt32 || value > math.MaxInt32 {
		return 0, errors.NewInvalidRequestError(providerName, fmt.Errorf("%s must fit in a 32-bit integer", field))
	}
	return int32(value), nil
}

func positiveInt32(value int, field string) (int32, error) {
	if value <= 0 {
		return 0, errors.NewInvalidRequestError(providerName, fmt.Errorf("%s must be positive", field))
	}
	return int32Value(value, field)
}

func convertMessage(msg providers.Message) (*genai.Content, error) {
	switch msg.Role {
	case providers.RoleUser:
		return convertUserMessage(msg)
	case providers.RoleAssistant:
		return convertAssistantMessage(msg)
	case providers.RoleTool:
		return convertToolMessage(msg), nil
	default:
		return nil, errors.NewInvalidRequestError(providerName, fmt.Errorf("unsupported message role %q", msg.Role))
	}
}

func convertMessages(messages []providers.Message) ([]*genai.Content, *genai.Content, error) {
	var contents []*genai.Content
	var systemParts []*genai.Part

	toolNames := make(map[string]string)
	for _, msg := range messages {
		if msg.Role == providers.RoleSystem {
			// Gemini systemInstruction is an ordered Content.Parts value. Reject
			// portable parts this adapter cannot represent without data loss.
			// https://ai.google.dev/api/generate-content#systeminstruction
			if msg.IsMultiModal() {
				for _, part := range msg.ContentParts() {
					if part.Type != contentPartTypeText {
						return nil, nil, errors.NewInvalidRequestError(
							providerName,
							fmt.Errorf("unsupported system content part type %q", part.Type),
						)
					}
					systemParts = append(systemParts, genai.NewPartFromText(part.Text))
				}
			} else {
				systemParts = append(systemParts, genai.NewPartFromText(msg.ContentString()))
			}
			continue
		}

		if msg.Role == providers.RoleAssistant {
			for _, call := range msg.ToolCalls {
				toolNames[call.ID] = call.Function.Name
			}
		}
		if msg.Role == providers.RoleTool && msg.Name == "" {
			msg.Name = toolNames[msg.ToolCallID]
		}
		converted, err := convertMessage(msg)
		if err != nil {
			return nil, nil, err
		}
		if converted != nil {
			contents = append(contents, converted)
		}
	}

	var systemInstruction *genai.Content
	if len(systemParts) > 0 {
		systemInstruction = &genai.Content{Role: roleUser, Parts: systemParts}
	}

	return contents, systemInstruction, nil
}

func validateMessages(messages []providers.Message) error {
	for _, msg := range messages {
		if err := validateMessage(msg); err != nil {
			return err
		}
	}
	return nil
}

func validateMessage(msg providers.Message) error {
	switch msg.Role {
	case providers.RoleSystem, providers.RoleUser, providers.RoleAssistant, providers.RoleTool:
	default:
		return errors.NewInvalidRequestError(providerName, fmt.Errorf("unsupported message role %q", msg.Role))
	}
	if err := validateMessageContent(msg.Content); err != nil {
		return err
	}
	for _, call := range msg.ToolCalls {
		if err := validateToolCall(call); err != nil {
			return err
		}
	}
	for _, part := range msg.ContentParts() {
		if err := validateContentPart(part); err != nil {
			return err
		}
	}
	return nil
}

func validateMessageContent(content any) error {
	switch value := content.(type) {
	case nil, string, []providers.ContentPart:
		return nil
	case []any:
		for _, rawPart := range value {
			partMap, ok := rawPart.(map[string]any)
			if !ok {
				return errors.NewInvalidRequestError(providerName, fmt.Errorf("content part must be an object"))
			}
			body, err := json.Marshal(partMap)
			if err != nil {
				return errors.NewInvalidRequestError(providerName, fmt.Errorf("invalid content part: %w", err))
			}
			var part providers.ContentPart
			if err = json.Unmarshal(body, &part); err != nil {
				return errors.NewInvalidRequestError(providerName, fmt.Errorf("invalid content part: %w", err))
			}
		}
		return nil
	default:
		return errors.NewInvalidRequestError(
			providerName,
			fmt.Errorf("message content must be a string or content parts"),
		)
	}
}

func validateToolCall(call providers.ToolCall) error {
	if call.Function.Name == "" {
		return errors.NewInvalidRequestError(providerName, fmt.Errorf("tool call function name is required"))
	}
	if call.Function.Arguments != "" {
		var args map[string]any
		if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil || args == nil {
			return errors.NewInvalidRequestError(providerName, fmt.Errorf("tool call arguments must be a JSON object"))
		}
	}
	if data, ok := call.Extra[providerName]; ok {
		if _, present := data[extraKeyThoughtSignature]; present && thoughtSignatureFromExtra(call.Extra) == nil {
			return errors.NewInvalidRequestError(providerName, fmt.Errorf("thought_signature must be valid base64"))
		}
	}
	return nil
}

func validateContentPart(part providers.ContentPart) error {
	switch part.Type {
	case contentPartTypeText:
		return nil
	case contentPartTypeImageURL:
		return nil
	case contentPartTypeFile:
		return nil
	default:
		return errors.NewInvalidRequestError(providerName, fmt.Errorf("unsupported content part type %q", part.Type))
	}
}

func convertToolChoice(choice any) (*genai.ToolConfig, error) {
	switch v := choice.(type) {
	case string:
		switch v {
		case "auto":
			return &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode: genai.FunctionCallingConfigModeAuto,
				},
			}, nil
		case "none":
			return &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode: genai.FunctionCallingConfigModeNone,
				},
			}, nil
		case "required", "any":
			return &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode: genai.FunctionCallingConfigModeAny,
				},
			}, nil
		}
	case providers.ToolChoice:
		if v.Function != nil {
			return &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode:                 genai.FunctionCallingConfigModeAny,
					AllowedFunctionNames: []string{v.Function.Name},
				},
			}, nil
		}
	}

	return nil, errors.NewInvalidRequestError(providerName, fmt.Errorf("unsupported tool_choice"))
}

func convertToolMessage(msg providers.Message) *genai.Content {
	name := msg.Name
	if name == "" {
		name = toolCallFallbackName
	}

	content := msg.ContentString()

	// Try to parse content as JSON first (structured tool responses).
	// If parsing fails, wrap the raw content as {"result": content}.
	var response map[string]any
	if err := json.Unmarshal([]byte(content), &response); err != nil {
		response = map[string]any{
			"result": content,
		}
	}

	part := genai.NewPartFromFunctionResponse(name, response)
	part.FunctionResponse.ID = msg.ToolCallID

	return &genai.Content{
		Role:  roleUser,
		Parts: []*genai.Part{part},
	}
}

func convertTools(tools []providers.Tool) []*genai.Tool {
	declarations := make([]*genai.FunctionDeclaration, 0, len(tools))

	for _, tool := range tools {
		decl := &genai.FunctionDeclaration{
			Name:        tool.Function.Name,
			Description: tool.Function.Description,
		}

		if tool.Function.Parameters != nil {
			decl.ParametersJsonSchema = tool.Function.Parameters
		}

		declarations = append(declarations, decl)
	}

	return []*genai.Tool{{
		FunctionDeclarations: declarations,
	}}
}

func convertUserMessage(msg providers.Message) (*genai.Content, error) {
	if !msg.IsMultiModal() {
		return genai.NewContentFromText(msg.ContentString(), roleUser), nil
	}

	parts, err := convertContentParts(msg.ContentParts())
	if err != nil {
		return nil, err
	}

	return genai.NewContentFromParts(parts, roleUser), nil
}

func convertContentParts(contentParts []providers.ContentPart) ([]*genai.Part, error) {
	parts := make([]*genai.Part, 0, len(contentParts))
	for _, part := range contentParts {
		switch part.Type {
		case contentPartTypeText:
			parts = append(parts, genai.NewPartFromText(part.Text))
		case contentPartTypeImageURL:
			converted, err := convertImagePart(part.ImageURL)
			if err != nil {
				return nil, err
			}
			parts = append(parts, converted)
		case contentPartTypeFile:
			converted, err := convertFilePart(part.File)
			if err != nil {
				return nil, err
			}
			parts = append(parts, converted)
		default:
			return nil, errors.NewInvalidRequestError(
				providerName,
				fmt.Errorf("%w %q", errUnsupportedContentPartType, part.Type),
			)
		}
	}

	return parts, nil
}

func thoughtSignatureExtra(part *genai.Part) map[string]providers.ProviderData {
	if len(part.ThoughtSignature) == 0 {
		return nil
	}
	return map[string]providers.ProviderData{
		providerName: {extraKeyThoughtSignature: base64.StdEncoding.EncodeToString(part.ThoughtSignature)},
	}
}

func thoughtSignatureFromExtra(extra map[string]providers.ProviderData) []byte {
	if extra == nil {
		return nil
	}

	geminiData, ok := extra[providerName]
	if !ok {
		return nil
	}

	sigStr, ok := geminiData[extraKeyThoughtSignature].(string)
	if !ok {
		return nil
	}

	sig, err := base64.StdEncoding.DecodeString(sigStr)
	if err != nil {
		return nil
	}

	return sig
}
