package anthropic

import (
	"encoding/base64"
	"encoding/json"
	stderrors "errors"
	"fmt"
	"net/url"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

var errToolCallArgumentsNotObject = stderrors.New("tool call arguments must be a JSON object")

// applyThinking configures thinking/reasoning on the request if applicable.
// Matches Python any-llm: none disables thinking, empty and auto leave it unset,
// and other efforts use adaptive thinking with output_config.effort.
// Go's empty string is the omitted field (Python's default auto), not Python's None.
func applyThinking(req *anthropic.MessageNewParams, effort providers.ReasoningEffort) error {
	if effort == "" || effort == providers.ReasoningEffortAuto {
		return nil
	}
	if effort == providers.ReasoningEffortNone {
		disabled := anthropic.NewThinkingConfigDisabledParam()
		req.Thinking = anthropic.ThinkingConfigParamUnion{OfDisabled: &disabled}
		return nil
	}

	level, ok := thinkingEffort(effort)
	if !ok {
		return errors.NewUnsupportedParamError(providerName, "reasoning_effort")
	}

	req.Thinking = anthropic.ThinkingConfigParamUnion{
		OfAdaptive: new(anthropic.ThinkingConfigAdaptiveParam),
	}
	req.OutputConfig.Effort = level
	return nil
}

// applyResponseFormat configures structured output on the request if applicable.
func applyResponseFormat(req *anthropic.MessageNewParams, format *providers.ResponseFormat) error {
	if format == nil {
		return nil
	}
	switch format.Type {
	case responseFormatJSONSchema:
		if format.JSONSchema == nil || len(format.JSONSchema.Schema) == 0 {
			return fmt.Errorf("json_schema response format requires a non-empty schema")
		}
		// JSONOutputFormatParam only carries Schema and Type; Name, Description, and Strict
		// from providers.JSONSchema are not supported by the Anthropic API.
		req.OutputConfig = anthropic.OutputConfigParam{
			Format: anthropic.JSONOutputFormatParam{
				Schema: format.JSONSchema.Schema,
			},
		}
	case responseFormatJSONObject:
		return fmt.Errorf("response format %q is unsupported without a JSON schema", format.Type)
	default:
		return fmt.Errorf("unsupported response format %q", format.Type)
	}
	return nil
}

// convertAssistantMessage converts an assistant message to Anthropic format.
func convertAssistantMessage(msg providers.Message) (*anthropic.MessageParam, error) {
	thinking, replayedThinking, err := convertThinkingMetadata(msg.Extra)
	if err != nil {
		return nil, err
	}
	if !replayedThinking && msg.Reasoning != nil && msg.Reasoning.Signature != "" {
		thinking = append(thinking, anthropic.NewThinkingBlock(msg.Reasoning.Signature, msg.Reasoning.Content))
	}
	content, err := convertAssistantContent(msg)
	if err != nil {
		return nil, err
	}
	thinking = append(thinking, content...)
	message := anthropic.NewAssistantMessage(thinking...)
	return &message, nil
}

func convertAssistantContent(msg providers.Message) ([]anthropic.ContentBlockParamUnion, error) {
	if len(msg.ToolCalls) == 0 && (msg.ContentString() != "" || msg.IsMultiModal()) {
		return convertContent(msg)
	}
	content := make([]anthropic.ContentBlockParamUnion, 0, len(msg.ToolCalls)+1)
	if msg.ContentString() != "" {
		content = append(content, anthropic.NewTextBlock(msg.ContentString()))
	}
	for _, tc := range msg.ToolCalls {
		converted, err := convertToolCall(tc)
		if err != nil {
			return nil, err
		}
		content = append(content, converted)
	}
	return content, nil
}

func convertThinkingMetadata(
	extra map[string]providers.ProviderData,
) ([]anthropic.ContentBlockParamUnion, bool, error) {
	data, exists := extra[providerName]
	if !exists {
		return nil, false, nil
	}
	raw, exists := data["thinking_blocks"]
	if !exists {
		return nil, false, nil
	}
	rawBlocks, ok := thinkingBlockValues(raw)
	if !ok {
		return nil, false, fmt.Errorf("anthropic thinking_blocks metadata must be an array")
	}
	blocks := make([]anthropic.ContentBlockParamUnion, 0, len(rawBlocks))
	for _, rawBlock := range rawBlocks {
		block, replay, err := replayThinkingBlock(rawBlock)
		if err != nil {
			return nil, false, err
		}
		if replay {
			blocks = append(blocks, block)
		}
	}
	return blocks, len(blocks) > 0, nil
}

// convertImagePart converts an image URL to Anthropic format.
func convertImagePart(img *providers.ImageURL) (anthropic.ContentBlockParamUnion, error) {
	imgURL := img.URL

	if strings.HasPrefix(imgURL, "data:") {
		return convertBase64Image(imgURL)
	}

	if err := validateHTTPSURL(imgURL); err != nil {
		return anthropic.ContentBlockParamUnion{}, fmt.Errorf("invalid image URL: %w", err)
	}
	return anthropic.NewImageBlock(anthropic.URLImageSourceParam{URL: imgURL}), nil
}

func convertBase64Image(dataURL string) (anthropic.ContentBlockParamUnion, error) {
	header, data, ok := strings.Cut(dataURL, ",")
	if !ok || !strings.HasSuffix(header, ";base64") {
		return anthropic.ContentBlockParamUnion{}, fmt.Errorf("invalid image data URL")
	}
	mediaType := strings.TrimSuffix(strings.TrimPrefix(header, "data:"), ";base64")
	if !validImageMediaType(mediaType) {
		return anthropic.ContentBlockParamUnion{}, fmt.Errorf("unsupported image media type %q", mediaType)
	}
	if _, err := base64.StdEncoding.DecodeString(data); err != nil {
		return anthropic.ContentBlockParamUnion{}, fmt.Errorf("invalid base64 image: %w", err)
	}
	return anthropic.NewImageBlockBase64(mediaType, data), nil
}

// convertMessage converts a single message to Anthropic format.
func convertMessage(msg providers.Message) (*anthropic.MessageParam, error) {
	switch msg.Role {
	case providers.RoleUser:
		return convertUserMessage(msg)
	case providers.RoleAssistant:
		return convertAssistantMessage(msg)
	case providers.RoleTool:
		return convertToolMessage(msg)
	default:
		return nil, fmt.Errorf("unsupported message role %q", msg.Role)
	}
}

// convertMessages converts providers messages to Anthropic format.
// Returns the messages and the combined system message.
func convertMessages(messages []providers.Message) ([]anthropic.MessageParam, []anthropic.TextBlockParam, error) {
	result := make([]anthropic.MessageParam, 0, len(messages))
	var systemParts []anthropic.TextBlockParam
	previousWasTool := false

	for _, msg := range messages {
		if msg.Role == providers.RoleSystem {
			previousWasTool = false
			blocks, err := convertSystemContent(msg)
			if err != nil {
				return nil, nil, err
			}
			systemParts = append(systemParts, blocks...)
			continue
		}

		converted, err := convertMessage(msg)
		if err != nil {
			return nil, nil, err
		}
		if msg.Role == providers.RoleTool && previousWasTool && len(result) > 0 {
			result[len(result)-1].Content = append(result[len(result)-1].Content, converted.Content...)
		} else if converted != nil {
			result = append(result, *converted)
		}
		previousWasTool = msg.Role == providers.RoleTool
	}

	return result, systemParts, nil
}

func convertResponseContent(
	blocks []anthropic.ContentBlockUnion,
) (string, *providers.Reasoning, []any, []providers.ToolCall) {
	var content strings.Builder
	var reasoning *providers.Reasoning
	thinkingBlocks := make([]any, 0)
	var toolCalls []providers.ToolCall

	for _, block := range blocks {
		switch block.Type {
		case blockTypeText:
			content.WriteString(block.Text)
		case blockTypeThinking:
			reasoning, thinkingBlocks = appendThinkingBlock(reasoning, thinkingBlocks, block)
		case blockTypeRedactedThinking:
			thinkingBlocks = append(
				thinkingBlocks,
				map[string]any{"type": blockTypeRedactedThinking, "data": block.Data},
			)
		case blockTypeToolUse:
			toolCalls = append(toolCalls, responseToolCall(block))
		}
	}

	return content.String(), reasoning, thinkingBlocks, toolCalls
}

func appendThinkingBlock(
	reasoning *providers.Reasoning,
	thinkingBlocks []any,
	block anthropic.ContentBlockUnion,
) (*providers.Reasoning, []any) {
	if reasoning == nil {
		reasoning = &providers.Reasoning{}
	}
	reasoning.Content += block.Thinking
	if block.Signature != "" {
		reasoning.Signature = block.Signature
		thinkingBlocks = append(
			thinkingBlocks,
			map[string]any{"type": blockTypeThinking, "thinking": block.Thinking, "signature": block.Signature},
		)
	}
	return reasoning, thinkingBlocks
}

func responseToolCall(block anthropic.ContentBlockUnion) providers.ToolCall {
	inputJSON := ""
	if block.Input != nil {
		if inputBytes, err := json.Marshal(block.Input); err == nil {
			inputJSON = string(inputBytes)
		}
	}
	return providers.ToolCall{
		ID:   block.ID,
		Type: "function",
		Function: providers.FunctionCall{
			Name:      block.Name,
			Arguments: inputJSON,
		},
	}
}

// convertResponse converts an Anthropic response to providers format.
func convertResponse(resp *anthropic.Message) *providers.ChatCompletion {
	content, reasoning, thinkingBlocks, toolCalls := convertResponseContent(resp.Content)
	message := providers.Message{
		Role:      providers.RoleAssistant,
		Content:   content,
		ToolCalls: toolCalls,
		Reasoning: reasoning,
	}
	if len(thinkingBlocks) > 0 {
		message.Extra = map[string]providers.ProviderData{providerName: {"thinking_blocks": thinkingBlocks}}
	}
	if resp.StopDetails.JSON.Type.Valid() {
		if message.Extra == nil {
			message.Extra = make(map[string]providers.ProviderData)
		}
		if message.Extra[providerName] == nil {
			message.Extra[providerName] = make(providers.ProviderData)
		}
		message.Extra[providerName]["stop_details"] = providers.ProviderData{
			"type":        string(resp.StopDetails.Type),
			"category":    string(resp.StopDetails.Category),
			"explanation": resp.StopDetails.Explanation,
		}
	}

	finishReason := convertStopReason(string(resp.StopReason))

	return &providers.ChatCompletion{
		ID:     resp.ID,
		Object: "chat.completion",
		Model:  resp.Model,
		Choices: []providers.Choice{{
			Index:        0,
			Message:      message,
			FinishReason: finishReason,
		}},
		Usage: &providers.Usage{
			CacheCreationInputTokens:            int(resp.Usage.CacheCreationInputTokens),
			CacheCreationInputTokensEphemeral1h: int(resp.Usage.CacheCreation.Ephemeral1hInputTokens),
			CacheCreationInputTokensEphemeral5m: int(resp.Usage.CacheCreation.Ephemeral5mInputTokens),
			CacheReadInputTokens:                int(resp.Usage.CacheReadInputTokens),
			ReasoningTokens:                     int(resp.Usage.OutputTokensDetails.ThinkingTokens),
			PromptTokens: int(
				resp.Usage.InputTokens + resp.Usage.CacheCreationInputTokens + resp.Usage.CacheReadInputTokens,
			),
			CompletionTokens: int(resp.Usage.OutputTokens),
			TotalTokens: int(
				resp.Usage.InputTokens + resp.Usage.CacheCreationInputTokens +
					resp.Usage.CacheReadInputTokens + resp.Usage.OutputTokens,
			),
		},
	}
}

// convertStopReason converts Anthropic stop reason to OpenAI finish reason.
func convertStopReason(reason string) string {
	switch reason {
	case stopReasonEndTurn:
		return providers.FinishReasonStop
	case stopReasonMaxTokens:
		return providers.FinishReasonLength
	case stopReasonContextLimit:
		return providers.FinishReasonLength
	case stopReasonRefusal:
		return providers.FinishReasonContentFilter
	case stopReasonToolUse:
		return providers.FinishReasonToolCalls
	case stopReasonStopSequence:
		return providers.FinishReasonStop
	default:
		return providers.FinishReasonStop
	}
}

// convertTool converts a providers.Tool to Anthropic format.
func convertTool(tool providers.Tool) (anthropic.ToolUnionParam, error) {
	inputSchema := anthropic.ToolInputSchemaParam{
		Type: "object",
	}

	if tool.Function.Parameters == nil {
		return buildToolParam(tool, inputSchema)
	}
	inputSchema.ExtraFields = make(map[string]any)
	for key, value := range tool.Function.Parameters {
		if key != schemaFieldProperties && key != schemaFieldRequired && key != "type" {
			inputSchema.ExtraFields[key] = value
		}
	}

	if props, ok := tool.Function.Parameters[schemaFieldProperties]; ok {
		inputSchema.Properties = props
	}

	req, ok := tool.Function.Parameters[schemaFieldRequired]
	if !ok {
		return buildToolParam(tool, inputSchema)
	}

	required, err := toStringSlice(req)
	if err != nil {
		return anthropic.ToolUnionParam{}, fmt.Errorf(
			"tool %s: invalid required field: %w",
			tool.Function.Name,
			err,
		)
	}
	inputSchema.Required = required

	return buildToolParam(tool, inputSchema)
}

// buildToolParam constructs the final ToolUnionParam from tool metadata and schema.
func buildToolParam(tool providers.Tool, schema anthropic.ToolInputSchemaParam) (anthropic.ToolUnionParam, error) {
	param := &anthropic.ToolParam{
		Name: tool.Function.Name, Description: anthropic.String(tool.Function.Description), InputSchema: schema,
	}
	if tool.CacheControl != nil {
		cache, err := convertCacheControl(tool.CacheControl)
		if err != nil {
			return anthropic.ToolUnionParam{}, err
		}
		param.CacheControl = cache
	}
	return anthropic.ToolUnionParam{
		OfTool: param,
	}, nil
}

// convertToolCall converts a tool call to Anthropic content block format.
func convertToolCall(tc providers.ToolCall) (anthropic.ContentBlockParamUnion, error) {
	// Anthropic requires tool_use.input to be an object. Empty normalized
	// arguments therefore encode as an empty object, not null.
	// https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls
	input := map[string]any{}

	arguments := strings.TrimSpace(tc.Function.Arguments)
	if arguments != "" {
		err := json.Unmarshal([]byte(arguments), &input)
		if err != nil {
			return anthropic.ContentBlockParamUnion{}, fmt.Errorf("tool call %q has invalid arguments: %w", tc.ID, err)
		}

		if input == nil {
			return anthropic.ContentBlockParamUnion{}, fmt.Errorf(
				"tool call %q: %w", tc.ID, errToolCallArgumentsNotObject,
			)
		}
	}

	return anthropic.ContentBlockParamUnion{
		OfToolUse: &anthropic.ToolUseBlockParam{
			Type:  "tool_use",
			ID:    tc.ID,
			Name:  tc.Function.Name,
			Input: input,
		},
	}, nil
}

// convertToolChoice converts providers tool choice to Anthropic format.
func convertToolChoice(choice any, parallelToolCalls *bool, hasTools bool) (anthropic.ToolChoiceUnionParam, error) {
	if !hasTools {
		return anthropic.ToolChoiceUnionParam{}, fmt.Errorf("tool choice and parallel_tool_calls require tools")
	}
	disableParallel := parallelToolCalls != nil && !*parallelToolCalls
	if choice == nil {
		choice = "auto"
	}

	switch v := choice.(type) {
	case string:
		return convertStringToolChoice(v, disableParallel)
	case providers.ToolChoice:
		return convertNamedToolChoice(v, disableParallel)
	}
	return anthropic.ToolChoiceUnionParam{}, fmt.Errorf("unsupported tool choice %v", choice)
}

func convertStringToolChoice(choice string, disableParallel bool) (anthropic.ToolChoiceUnionParam, error) {
	switch choice {
	case "auto":
		return anthropic.ToolChoiceUnionParam{OfAuto: &anthropic.ToolChoiceAutoParam{
			DisableParallelToolUse: anthropic.Bool(disableParallel),
		}}, nil
	case "none":
		return anthropic.ToolChoiceUnionParam{OfNone: &anthropic.ToolChoiceNoneParam{}}, nil
	case "required", "any":
		return anthropic.ToolChoiceUnionParam{OfAny: &anthropic.ToolChoiceAnyParam{
			DisableParallelToolUse: anthropic.Bool(disableParallel),
		}}, nil
	default:
		return anthropic.ToolChoiceUnionParam{}, fmt.Errorf("unsupported tool choice %q", choice)
	}
}

func convertNamedToolChoice(
	choice providers.ToolChoice,
	disableParallel bool,
) (anthropic.ToolChoiceUnionParam, error) {
	if choice.Function == nil {
		return anthropic.ToolChoiceUnionParam{}, fmt.Errorf("tool choice function is required")
	}
	if choice.Type != "" && choice.Type != "function" {
		return anthropic.ToolChoiceUnionParam{}, fmt.Errorf("unsupported tool choice type %q", choice.Type)
	}
	if choice.Function.Name == "" {
		return anthropic.ToolChoiceUnionParam{}, fmt.Errorf("tool choice function name is required")
	}
	return anthropic.ToolChoiceUnionParam{OfTool: &anthropic.ToolChoiceToolParam{
		Name:                   choice.Function.Name,
		DisableParallelToolUse: anthropic.Bool(disableParallel),
	}}, nil
}

// convertToolMessage converts a tool result message to Anthropic format.
func convertToolMessage(msg providers.Message) (*anthropic.MessageParam, error) {
	if msg.ToolCallID == "" {
		return nil, fmt.Errorf("tool_call_id is required")
	}
	blocks, err := convertContent(msg)
	if err != nil {
		return nil, err
	}
	resultContent := make([]anthropic.ToolResultBlockParamContentUnion, 0, len(blocks))
	for _, block := range blocks {
		resultContent = append(
			resultContent,
			anthropic.ToolResultBlockParamContentUnion{
				OfText:     block.OfText,
				OfImage:    block.OfImage,
				OfDocument: block.OfDocument,
			},
		)
	}
	toolResult := &anthropic.ToolResultBlockParam{ToolUseID: msg.ToolCallID, Content: resultContent}
	if data := msg.Extra[providerName]; data != nil {
		if value, ok := data["is_error"].(bool); ok {
			toolResult.IsError = anthropic.Bool(value)
		}
	}
	m := anthropic.NewUserMessage(anthropic.ContentBlockParamUnion{OfToolResult: toolResult})
	return &m, nil
}

// convertUserMessage converts a user message to Anthropic format.
func convertUserMessage(msg providers.Message) (*anthropic.MessageParam, error) {
	content, err := convertContent(msg)
	if err != nil {
		return nil, err
	}
	m := anthropic.NewUserMessage(content...)
	return &m, nil
}

func convertContent(msg providers.Message) ([]anthropic.ContentBlockParamUnion, error) {
	if !msg.IsMultiModal() {
		return []anthropic.ContentBlockParamUnion{anthropic.NewTextBlock(msg.ContentString())}, nil
	}
	parts := msg.ContentParts()
	content := make([]anthropic.ContentBlockParamUnion, 0, len(parts))
	for _, part := range parts {
		block, err := convertContentPart(part)
		if err != nil {
			return nil, err
		}
		if part.CacheControl != nil {
			cache, err := convertCacheControl(part.CacheControl)
			if err != nil {
				return nil, err
			}
			applyCacheControl(&block, cache)
		}
		content = append(content, block)
	}
	return content, nil
}

func convertContentPart(part providers.ContentPart) (anthropic.ContentBlockParamUnion, error) {
	switch part.Type {
	case "", blockTypeText:
		return anthropic.NewTextBlock(part.Text), nil
	case "image_url":
		if part.ImageURL == nil {
			return anthropic.ContentBlockParamUnion{}, fmt.Errorf("image_url content requires image_url")
		}
		return convertImagePart(part.ImageURL)
	case "file":
		if part.File == nil {
			return anthropic.ContentBlockParamUnion{}, fmt.Errorf("file content requires file_data")
		}
		return convertPDFPart(part.File.FileData)
	default:
		return anthropic.ContentBlockParamUnion{}, fmt.Errorf("unsupported content part type %q", part.Type)
	}
}

func applyCacheControl(block *anthropic.ContentBlockParamUnion, cache anthropic.CacheControlEphemeralParam) {
	switch {
	case block.OfText != nil:
		block.OfText.CacheControl = cache
	case block.OfImage != nil:
		block.OfImage.CacheControl = cache
	case block.OfDocument != nil:
		block.OfDocument.CacheControl = cache
	}
}

func convertPDFPart(fileData string) (anthropic.ContentBlockParamUnion, error) {
	doc := &anthropic.DocumentBlockParam{}
	if strings.HasPrefix(fileData, "data:") {
		prefix, data, ok := strings.Cut(fileData, ",")
		if !ok || prefix != "data:application/pdf;base64" {
			return anthropic.ContentBlockParamUnion{}, fmt.Errorf("PDF data must use data:application/pdf;base64")
		}
		if _, err := base64.StdEncoding.DecodeString(data); err != nil {
			return anthropic.ContentBlockParamUnion{}, fmt.Errorf("invalid base64 PDF: %w", err)
		}
		doc.Source.OfBase64 = &anthropic.Base64PDFSourceParam{Data: data}
	} else {
		if err := validateHTTPSURL(fileData); err != nil {
			return anthropic.ContentBlockParamUnion{}, fmt.Errorf("invalid PDF URL: %w", err)
		}
		doc.Source.OfURL = &anthropic.URLPDFSourceParam{URL: fileData}
	}
	return anthropic.ContentBlockParamUnion{OfDocument: doc}, nil
}

func convertCacheControl(cache *providers.CacheControl) (anthropic.CacheControlEphemeralParam, error) {
	if cache.Type != "ephemeral" {
		return anthropic.CacheControlEphemeralParam{}, fmt.Errorf("unsupported cache control type %q", cache.Type)
	}
	if cache.TTL != "" && cache.TTL != "5m" && cache.TTL != "1h" {
		return anthropic.CacheControlEphemeralParam{}, fmt.Errorf("unsupported cache control TTL %q", cache.TTL)
	}
	return anthropic.CacheControlEphemeralParam{
		TTL:  anthropic.CacheControlEphemeralTTL(cache.TTL),
		Type: "ephemeral",
	}, nil
}

func validateHTTPSURL(raw string) error {
	parsed, err := url.Parse(raw)
	if err != nil || parsed.Scheme != "https" || parsed.Host == "" {
		return fmt.Errorf("must be an absolute HTTPS URL")
	}
	return nil
}

func validImageMediaType(mediaType string) bool {
	switch mediaType {
	case "image/jpeg", "image/png", "image/gif", "image/webp":
		return true
	default:
		return false
	}
}

func replayThinkingBlock(raw any) (anthropic.ContentBlockParamUnion, bool, error) {
	var block map[string]any
	switch value := raw.(type) {
	case providers.ProviderData:
		block = value
	case map[string]any:
		block = value
	default:
		return anthropic.ContentBlockParamUnion{}, false, fmt.Errorf("invalid Anthropic thinking block metadata")
	}
	switch block["type"] {
	case blockTypeThinking:
		thinking, tok := block["thinking"].(string)
		signature, sok := block["signature"].(string)
		if !tok || !sok {
			return anthropic.ContentBlockParamUnion{}, false, fmt.Errorf(
				"thinking block requires thinking and signature",
			)
		}
		// A stream starts with an empty signature and emits the complete signature later.
		// Only completed blocks can be replayed: https://platform.claude.com/docs/en/build-with-claude/thinking
		if signature == "" {
			return anthropic.ContentBlockParamUnion{}, false, nil
		}
		return anthropic.NewThinkingBlock(signature, thinking), true, nil
	case blockTypeRedactedThinking:
		data, ok := block["data"].(string)
		if !ok {
			return anthropic.ContentBlockParamUnion{}, false, fmt.Errorf("redacted thinking block requires data")
		}
		return anthropic.ContentBlockParamUnion{
			OfRedactedThinking: &anthropic.RedactedThinkingBlockParam{Data: data},
		}, true, nil
	default:
		return anthropic.ContentBlockParamUnion{}, false, fmt.Errorf(
			"unsupported thinking block type %v",
			block["type"],
		)
	}
}

func thinkingBlockValues(raw any) ([]any, bool) {
	switch blocks := raw.(type) {
	case []any:
		return blocks, true
	case []providers.ProviderData:
		values := make([]any, len(blocks))
		for i := range blocks {
			values[i] = blocks[i]
		}
		return values, true
	default:
		return nil, false
	}
}

func convertSystemContent(msg providers.Message) ([]anthropic.TextBlockParam, error) {
	if !msg.IsMultiModal() {
		return []anthropic.TextBlockParam{{Text: msg.ContentString()}}, nil
	}
	parts := msg.ContentParts()
	result := make([]anthropic.TextBlockParam, 0, len(parts))
	for _, part := range parts {
		if part.Type != "" && part.Type != blockTypeText {
			return nil, fmt.Errorf("system messages only support text content")
		}
		block := anthropic.TextBlockParam{Text: part.Text}
		if part.CacheControl != nil {
			cache, err := convertCacheControl(part.CacheControl)
			if err != nil {
				return nil, err
			}
			block.CacheControl = cache
		}
		result = append(result, block)
	}
	return result, nil
}

// thinkingEffort maps canonical reasoning effort onto Anthropic output_config.effort.
// minimal maps to low, matching Python any-llm. xhigh and max stay distinct.
func thinkingEffort(effort providers.ReasoningEffort) (anthropic.OutputConfigEffort, bool) {
	switch effort {
	case "", providers.ReasoningEffortAuto, providers.ReasoningEffortNone:
		return "", false
	case providers.ReasoningEffortLow, providers.ReasoningEffortMinimal:
		return anthropic.OutputConfigEffortLow, true
	case providers.ReasoningEffortMedium:
		return anthropic.OutputConfigEffortMedium, true
	case providers.ReasoningEffortHigh:
		return anthropic.OutputConfigEffortHigh, true
	case providers.ReasoningEffortXHigh:
		return anthropic.OutputConfigEffortXhigh, true
	case providers.ReasoningEffortMax:
		return anthropic.OutputConfigEffortMax, true
	default:
		return "", false
	}
}

// toStringSlice converts a value to []string.
// Accepts []string (returned as-is) or []any (each element must be string).
func toStringSlice(v any) ([]string, error) {
	switch typed := v.(type) {
	case []string:
		return typed, nil
	case []any:
		result := make([]string, len(typed))
		for i, elem := range typed {
			s, ok := elem.(string)
			if !ok {
				return nil, fmt.Errorf("element %d: expected string, got %T", i, elem)
			}
			result[i] = s
		}
		return result, nil
	default:
		return nil, fmt.Errorf("expected []string or []any, got %T", v)
	}
}
