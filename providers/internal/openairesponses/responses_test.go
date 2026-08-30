package openairesponses

import (
	"encoding/json"
	"testing"

	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

const (
	testInput        = "think"
	testModel        = "gpt-4o-mini"
	testProviderName = "openai"
)

func TestConvertParams(t *testing.T) {
	t.Parallel()

	instructions := "you are a shell assistant"
	params := providers.ResponsesParams{
		Model:        testModel,
		Instructions: &instructions,
		Input: []providers.ResponsesInputItem{
			{Role: providers.RoleUser, Content: "list files"},
			{Role: providers.RoleAssistant, Content: "=ls"},
		},
	}

	req, err := convertParams(testProviderName, params)
	require.NoError(t, err)
	require.Equal(t, testModel, req.Model)
	require.Equal(t, "you are a shell assistant", req.Instructions.Value)
	require.Len(t, req.Input.OfInputItemList, 2)
}

func TestConvertParamsPreservesExplicitEmptyInstructions(t *testing.T) {
	t.Parallel()

	empty := ""
	params := providers.ResponsesParams{
		Model:        testModel,
		Instructions: &empty,
		Input:        []providers.ResponsesInputItem{{Role: providers.RoleUser, Content: testInput}},
	}

	req, err := convertParams(testProviderName, params)
	require.NoError(t, err)
	wire, err := json.Marshal(req)
	require.NoError(t, err)
	var body map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(wire, &body))
	require.JSONEq(t, `""`, string(body["instructions"]))

	params.Instructions = nil
	req, err = convertParams(testProviderName, params)
	require.NoError(t, err)
	wire, err = json.Marshal(req)
	require.NoError(t, err)
	body = nil
	require.NoError(t, json.Unmarshal(wire, &body))
	require.NotContains(t, body, "instructions")
}

func TestConvertParamsPreservesDeveloperRole(t *testing.T) {
	t.Parallel()

	params := providers.ResponsesParams{
		Model: testModel,
		Input: []providers.ResponsesInputItem{
			{Role: providers.RoleDeveloper, Content: "Answer in JSON."},
			{Role: providers.RoleUser, Content: "List files."},
		},
	}

	req, err := convertParams(testProviderName, params)
	require.NoError(t, err)

	wire, err := json.Marshal(req)
	require.NoError(t, err)

	var body struct {
		Input []struct {
			Role string `json:"role"`
		} `json:"input"`
	}
	require.NoError(t, json.Unmarshal(wire, &body))
	require.Len(t, body.Input, 2)
	require.Equal(t, providers.RoleDeveloper, body.Input[0].Role)
	require.Equal(t, providers.RoleUser, body.Input[1].Role)
}

func TestConvertOutputItemsPreservesAnnotationsAndProviderFields(t *testing.T) {
	t.Parallel()

	var item responses.ResponseOutputItemUnion
	require.NoError(t, json.Unmarshal([]byte(`{
		"id":"msg_1",
		"type":"message",
		"role":"assistant",
		"status":"completed",
		"provider_extension":{"trace_id":"trace_1"},
		"content":[{
			"type":"output_text",
			"text":"source",
			"annotations":[{
				"type":"url_citation",
				"start_index":0,
				"end_index":6,
				"title":"Example",
				"url":"https://example.test"
			}]
		}]
	}`), &item))

	converted, err := convertOutputItems([]responses.ResponseOutputItemUnion{item})
	require.NoError(t, err)
	require.Len(t, converted, 1)
	require.Equal(t, "source", converted[0].Content)

	var raw struct {
		Content []struct {
			Annotations []struct {
				EndIndex   int    `json:"end_index"`
				StartIndex int    `json:"start_index"`
				Title      string `json:"title"`
				Type       string `json:"type"`
				URL        string `json:"url"`
			} `json:"annotations"`
		} `json:"content"`
		ProviderExtension struct {
			TraceID string `json:"trace_id"`
		} `json:"provider_extension"`
	}
	require.NoError(t, json.Unmarshal(converted[0].ProviderRaw, &raw))
	require.Len(t, raw.Content, 1)
	require.Len(t, raw.Content[0].Annotations, 1)
	require.Equal(t, "url_citation", raw.Content[0].Annotations[0].Type)
	require.Equal(t, "https://example.test", raw.Content[0].Annotations[0].URL)
	require.Equal(t, "trace_1", raw.ProviderExtension.TraceID)
}

func TestConvertParamsSetsDocumentedReasoningEfforts(t *testing.T) {
	t.Parallel()

	efforts := []providers.ReasoningEffort{
		providers.ReasoningEffortNone,
		providers.ReasoningEffortLow,
		providers.ReasoningEffortMedium,
		providers.ReasoningEffortHigh,
		providers.ReasoningEffortXHigh,
		providers.ReasoningEffortMax,
	}

	for _, effort := range efforts {
		t.Run(string(effort), func(t *testing.T) {
			t.Parallel()
			params := providers.ResponsesParams{
				Model:     "gpt-5.6",
				Reasoning: effort,
				Input:     []providers.ResponsesInputItem{{Role: providers.RoleUser, Content: testInput}},
			}

			req, err := convertParams(testProviderName, params)
			require.NoError(t, err)
			require.Equal(t, string(effort), string(req.Reasoning.Effort))
		})
	}
}

func TestConvertParamsAllowsSDKReasoningEffortForOtherModels(t *testing.T) {
	t.Parallel()

	params := providers.ResponsesParams{
		Model:     "gpt-5",
		Reasoning: providers.ReasoningEffortMinimal,
		Input:     []providers.ResponsesInputItem{{Role: providers.RoleUser, Content: testInput}},
	}

	req, err := convertParams(testProviderName, params)
	require.NoError(t, err)
	require.Equal(t, responses.ReasoningEffortMinimal, req.Reasoning.Effort)
}

func TestConvertParamsRejectsUnknownReasoningEffort(t *testing.T) {
	t.Parallel()

	params := providers.ResponsesParams{
		Model:     "o3-mini",
		Reasoning: "banana",
		Input:     []providers.ResponsesInputItem{{Role: providers.RoleUser, Content: testInput}},
	}

	_, err := convertParams(testProviderName, params)
	require.Error(t, err)
	require.ErrorIs(t, err, errors.ErrUnsupportedParam)
}

func TestConvertParamsOmitsAutoReasoning(t *testing.T) {
	t.Parallel()

	params := providers.ResponsesParams{
		Model:     "o3-mini",
		Reasoning: providers.ReasoningEffortAuto,
		Input:     []providers.ResponsesInputItem{{Role: providers.RoleUser, Content: testInput}},
	}

	req, err := convertParams(testProviderName, params)
	require.NoError(t, err)
	require.Empty(t, req.Reasoning.Effort)
}

func TestConvertParamsRejectsUnknownRoles(t *testing.T) {
	t.Parallel()

	params := providers.ResponsesParams{
		Model: testModel,
		Input: []providers.ResponsesInputItem{{Role: "moderator", Content: "nope"}},
	}

	_, err := convertParams(testProviderName, params)
	require.Error(t, err)
	require.ErrorIs(t, err, errors.ErrInvalidRequest)
	require.Contains(t, err.Error(), "unsupported responses role")

	var invalidErr *errors.InvalidRequestError
	require.ErrorAs(t, err, &invalidErr)
	require.Equal(t, testProviderName, invalidErr.Provider)
}

func TestValidateParams(t *testing.T) {
	t.Parallel()

	t.Run("requires model", func(t *testing.T) {
		t.Parallel()
		err := validateParams(testProviderName, providers.ResponsesParams{
			Input: []providers.ResponsesInputItem{{Role: providers.RoleUser, Content: "hi"}},
		})
		require.Error(t, err)
		require.Contains(t, err.Error(), "model is required")

		var invalidErr *errors.InvalidRequestError
		require.ErrorAs(t, err, &invalidErr)
		require.Equal(t, testProviderName, invalidErr.Provider)
	})

	t.Run("requires input", func(t *testing.T) {
		t.Parallel()
		err := validateParams(testProviderName, providers.ResponsesParams{Model: testModel})
		require.Error(t, err)
		require.Contains(t, err.Error(), "at least one input item is required")
	})
}
