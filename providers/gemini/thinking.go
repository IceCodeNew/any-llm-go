package gemini

import (
	"strconv"
	"strings"

	"google.golang.org/genai"

	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

// Default thinking budgets for reasoning effort levels.
// These match the Python any-llm library.
const (
	geminiThinkingLevelMajor        = 3
	thinkingBudgetHigh        int32 = 24576
	thinkingBudgetLow         int32 = 1024
	thinkingBudgetMax         int32 = 32768
	thinkingBudgetMedium      int32 = 8192
	thinkingBudgetMinimal     int32 = 256
	thinkingBudgetMinimalLite int32 = 512
	thinkingBudgetXHigh       int32 = 32768
)

// applyThinking configures thinking/reasoning on the config if applicable.
// Empty and auto leave thinking unset. none uses thinking_budget=0 only for the
// Gemini 2.5 Flash families that Google documents as supporting disabled thinking.
// https://ai.google.dev/gemini-api/docs/generate-content/thinking#thinking-budgets
func applyThinking(cfg *genai.GenerateContentConfig, model string, effort providers.ReasoningEffort) error {
	if effort == "" || effort == providers.ReasoningEffortAuto {
		return nil
	}
	if effort == providers.ReasoningEffortNone {
		if !supportsDisabledThinkingBudget(model) {
			return errors.NewUnsupportedParamError(providerName, "reasoning_effort")
		}

		cfg.ThinkingConfig = &genai.ThinkingConfig{
			IncludeThoughts: false,
			ThinkingBudget:  new(int32(0)),
		}
		return nil
	}

	if usesThinkingLevel(model) {
		level, ok := thinkingLevel(effort)
		if !ok {
			return errors.NewUnsupportedParamError(providerName, "reasoning_effort")
		}
		cfg.ThinkingConfig = &genai.ThinkingConfig{
			IncludeThoughts: true,
			ThinkingLevel:   level,
		}
		return nil
	}

	budget, ok := thinkingBudget(effort)
	if !ok {
		return errors.NewUnsupportedParamError(providerName, "reasoning_effort")
	}

	budget = max(budget, minThinkingBudget(model))
	budget = min(budget, maxThinkingBudget(model))

	cfg.ThinkingConfig = &genai.ThinkingConfig{
		IncludeThoughts: true,
		ThinkingBudget:  &budget,
	}
	return nil
}

// thinkingBudget returns the token budget for the given reasoning effort.
func thinkingBudget(effort providers.ReasoningEffort) (int32, bool) {
	switch effort {
	case providers.ReasoningEffortMinimal:
		return thinkingBudgetMinimal, true
	case providers.ReasoningEffortLow:
		return thinkingBudgetLow, true
	case providers.ReasoningEffortMedium:
		return thinkingBudgetMedium, true
	case providers.ReasoningEffortHigh:
		return thinkingBudgetHigh, true
	case providers.ReasoningEffortXHigh:
		return thinkingBudgetXHigh, true
	case providers.ReasoningEffortMax:
		return thinkingBudgetMax, true
	case providers.ReasoningEffortNone, providers.ReasoningEffortAuto:
		return 0, false
	default:
		return 0, false
	}
}

// minThinkingBudget applies Google's model-specific lower bounds. Flash-Lite
// accepts 0 to disable thinking, but positive budgets start at 512 tokens.
func minThinkingBudget(modelID string) int32 {
	if isGemini25FlashLite(modelID) {
		return thinkingBudgetMinimalLite
	}

	return 0
}

// maxThinkingBudget applies Google's model-specific Gemini 2.5 limits. Flash
// and Flash-Lite allow 24,576 tokens, while Pro allows 32,768.
// https://ai.google.dev/gemini-api/docs/generate-content/thinking#thinking-budgets
func maxThinkingBudget(modelID string) int32 {
	if isGemini25Flash(modelID) {
		return thinkingBudgetHigh
	}

	return thinkingBudgetMax
}

// thinkingLevel returns the Gemini thinking_level for the given reasoning effort.
func thinkingLevel(effort providers.ReasoningEffort) (genai.ThinkingLevel, bool) {
	switch effort {
	case providers.ReasoningEffortMinimal:
		return genai.ThinkingLevelMinimal, true
	case providers.ReasoningEffortLow:
		return genai.ThinkingLevelLow, true
	case providers.ReasoningEffortMedium:
		return genai.ThinkingLevelMedium, true
	case providers.ReasoningEffortHigh, providers.ReasoningEffortXHigh, providers.ReasoningEffortMax:
		return genai.ThinkingLevelHigh, true
	case providers.ReasoningEffortNone, providers.ReasoningEffortAuto:
		return "", false
	default:
		return "", false
	}
}

// usesThinkingLevel reports whether the model expects thinking_level instead of
// thinking_budget. Gemini 3 and newer reject thinking_budget.
func usesThinkingLevel(modelID string) bool {
	major, ok := geminiMajorVersion(modelID)
	if !ok {
		return false
	}
	return major >= geminiThinkingLevelMajor
}

// supportsDisabledThinkingBudget reports whether thinking_budget=0 can disable thoughts.
// Gemini 2.5 Flash and Flash-Lite accept 0. Gemini 2.5 Pro requires a positive minimum.
func supportsDisabledThinkingBudget(modelID string) bool {
	if usesThinkingLevel(modelID) {
		return false
	}

	return isGemini25Flash(modelID)
}

func isGemini25Flash(modelID string) bool {
	modelName := geminiModelName(modelID)

	return modelName == "gemini-2.5-flash" ||
		strings.HasPrefix(modelName, "gemini-2.5-flash-preview") ||
		isGemini25FlashLite(modelName)
}

func isGemini25FlashLite(modelID string) bool {
	modelName := geminiModelName(modelID)

	return modelName == "gemini-2.5-flash-lite" ||
		strings.HasPrefix(modelName, "gemini-2.5-flash-lite-preview")
}

func geminiModelName(modelID string) string {
	modelName := strings.ToLower(modelID)
	if slash := strings.LastIndexByte(modelName, '/'); slash >= 0 {
		modelName = modelName[slash+1:]
	}

	return modelName
}

// geminiMajorVersion extracts the major version from a Gemini model ID.
func geminiMajorVersion(modelID string) (int, bool) {
	_, version, ok := strings.Cut(strings.ToLower(modelID), "gemini-")
	if !ok {
		return 0, false
	}

	end := strings.IndexFunc(version, func(char rune) bool {
		return char < '0' || char > '9'
	})
	if end < 0 {
		end = len(version)
	}
	if end == 0 {
		return 0, false
	}

	major, err := strconv.Atoi(version[:end])
	return major, err == nil
}
