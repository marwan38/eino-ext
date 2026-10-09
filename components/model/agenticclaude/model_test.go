/*
 * Copyright 2026 CloudWeGo Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package agenticclaude

import (
	"bytes"
	"context"
	"encoding/gob"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
	"github.com/eino-contrib/jsonschema"
)

func TestGetAllowedToolNames(t *testing.T) {
	t.Run("deduplicate function and server tools", func(t *testing.T) {
		got, err := getAllowedToolNames(&schema.AgenticToolChoice{
			Type: schema.ToolChoiceAllowed,
			Allowed: &schema.AgenticAllowedToolChoice{
				Tools: []*schema.AllowedTool{
					{FunctionName: "fn"},
					{FunctionName: "fn"},
					{ServerTool: &schema.AllowedServerTool{Name: string(ServerToolNameWebFetch)}},
					{ServerTool: &schema.AllowedServerTool{Name: string(ServerToolNameWebFetch)}},
				},
			},
		})
		if err != nil {
			t.Fatalf("getAllowedToolNames() error = %v", err)
		}
		if len(got) != 2 {
			t.Fatalf("len(getAllowedToolNames()) = %d, want 2", len(got))
		}
		if got[0] != "fn" || got[1] != string(ServerToolNameWebFetch) {
			t.Fatalf("getAllowedToolNames() = %#v", got)
		}
	})

	t.Run("empty server tool name", func(t *testing.T) {
		_, err := getAllowedToolNames(&schema.AgenticToolChoice{
			Type: schema.ToolChoiceForced,
			Forced: &schema.AgenticForcedToolChoice{
				Tools: []*schema.AllowedTool{
					{ServerTool: &schema.AllowedServerTool{}},
				},
			},
		})
		if err == nil || err.Error() != "server tool name is empty" {
			t.Fatalf("getAllowedToolNames() error = %v", err)
		}
	})

	t.Run("mcp tool unsupported", func(t *testing.T) {
		_, err := getAllowedToolNames(&schema.AgenticToolChoice{
			Type: schema.ToolChoiceAllowed,
			Allowed: &schema.AgenticAllowedToolChoice{
				Tools: []*schema.AllowedTool{
					{MCPTool: &schema.AllowedMCPTool{ServerLabel: "srv", Name: "tool"}},
				},
			},
		})
		if err == nil || err.Error() != "mcp tools are not supported yet" {
			t.Fatalf("getAllowedToolNames() error = %v", err)
		}
	})
}

func TestPopulateToolChoice(t *testing.T) {
	disableParallelToolUse := true
	m := &Model{disableParallelToolUse: &disableParallelToolUse}

	t.Run("forbidden", func(t *testing.T) {
		req := &anthropic.MessageNewParams{}
		err := m.populateToolChoice(req, &model.Options{
			AgenticToolChoice: &schema.AgenticToolChoice{Type: schema.ToolChoiceForbidden},
		})
		if err != nil {
			t.Fatalf("populateToolChoice() error = %v", err)
		}
		if req.ToolChoice.OfNone == nil {
			t.Fatalf("populateToolChoice() did not set none tool choice")
		}
	})

	t.Run("allowed with specific tools is unsupported", func(t *testing.T) {
		req := &anthropic.MessageNewParams{}
		err := m.populateToolChoice(req, &model.Options{
			AgenticToolChoice: &schema.AgenticToolChoice{
				Type: schema.ToolChoiceAllowed,
				Allowed: &schema.AgenticAllowedToolChoice{
					Tools: []*schema.AllowedTool{{FunctionName: "fn"}},
				},
			},
		})
		if err == nil || err.Error() != "tool choice 'allowed' with specific tools is not supported" {
			t.Fatalf("populateToolChoice() error = %v", err)
		}
	})

	t.Run("forced without tools", func(t *testing.T) {
		req := &anthropic.MessageNewParams{}
		err := m.populateToolChoice(req, &model.Options{
			AgenticToolChoice: &schema.AgenticToolChoice{Type: schema.ToolChoiceForced},
		})
		if err == nil || err.Error() != "tool choice is forced but no tools are provided" {
			t.Fatalf("populateToolChoice() error = %v", err)
		}
	})

	t.Run("forced with explicit function name", func(t *testing.T) {
		req := &anthropic.MessageNewParams{
			Tools: []anthropic.ToolUnionParam{
				{OfTool: &anthropic.ToolParam{Name: "fn"}},
			},
		}
		err := m.populateToolChoice(req, &model.Options{
			AgenticToolChoice: &schema.AgenticToolChoice{
				Type: schema.ToolChoiceForced,
				Forced: &schema.AgenticForcedToolChoice{
					Tools: []*schema.AllowedTool{{FunctionName: "fn"}},
				},
			},
		})
		if err != nil {
			t.Fatalf("populateToolChoice() error = %v", err)
		}
		if req.ToolChoice.OfTool == nil {
			t.Fatalf("populateToolChoice() did not set tool choice")
		}
		if req.ToolChoice.OfTool.Name != "fn" {
			t.Fatalf("tool choice name = %q, want %q", req.ToolChoice.OfTool.Name, "fn")
		}
		if !req.ToolChoice.OfTool.DisableParallelToolUse.Valid() || !req.ToolChoice.OfTool.DisableParallelToolUse.Value {
			t.Fatalf("disable_parallel_tool_use = %#v, want true", req.ToolChoice.OfTool.DisableParallelToolUse)
		}
	})

	t.Run("forced with explicit server tool name", func(t *testing.T) {
		req := &anthropic.MessageNewParams{
			Tools: []anthropic.ToolUnionParam{
				{OfWebSearchTool20260209: &anthropic.WebSearchTool20260209Param{}},
			},
		}
		err := m.populateToolChoice(req, &model.Options{
			AgenticToolChoice: &schema.AgenticToolChoice{
				Type: schema.ToolChoiceForced,
				Forced: &schema.AgenticForcedToolChoice{
					Tools: []*schema.AllowedTool{{ServerTool: &schema.AllowedServerTool{Name: string(ServerToolNameWebSearch)}}},
				},
			},
		})
		if err != nil {
			t.Fatalf("populateToolChoice() error = %v", err)
		}
		if req.ToolChoice.OfTool == nil {
			t.Fatalf("populateToolChoice() did not set tool choice")
		}
		if req.ToolChoice.OfTool.Name != string(ServerToolNameWebSearch) {
			t.Fatalf("tool choice name = %q, want %q", req.ToolChoice.OfTool.Name, string(ServerToolNameWebSearch))
		}
	})

	t.Run("forced with expanded code execution tool name", func(t *testing.T) {
		req := &anthropic.MessageNewParams{
			Tools: []anthropic.ToolUnionParam{
				{OfCodeExecutionTool20260120: &anthropic.CodeExecutionTool20260120Param{}},
			},
		}
		err := m.populateToolChoice(req, &model.Options{
			AgenticToolChoice: &schema.AgenticToolChoice{
				Type: schema.ToolChoiceForced,
				Forced: &schema.AgenticForcedToolChoice{
					Tools: []*schema.AllowedTool{{ServerTool: &schema.AllowedServerTool{Name: string(ServerToolNameBashCodeExecution)}}},
				},
			},
		})
		if err != nil {
			t.Fatalf("populateToolChoice() error = %v", err)
		}
		if req.ToolChoice.OfTool == nil {
			t.Fatalf("populateToolChoice() did not set tool choice")
		}
		if req.ToolChoice.OfTool.Name != string(ServerToolNameBashCodeExecution) {
			t.Fatalf("tool choice name = %q, want %q", req.ToolChoice.OfTool.Name, string(ServerToolNameBashCodeExecution))
		}
	})

	t.Run("forced with single tool falls back to tool", func(t *testing.T) {
		req := &anthropic.MessageNewParams{
			Tools: []anthropic.ToolUnionParam{
				{OfTool: &anthropic.ToolParam{Name: "fn"}},
			},
		}
		err := m.populateToolChoice(req, &model.Options{
			AgenticToolChoice: &schema.AgenticToolChoice{Type: schema.ToolChoiceForced},
		})
		if err != nil {
			t.Fatalf("populateToolChoice() error = %v", err)
		}
		if req.ToolChoice.OfTool == nil || req.ToolChoice.OfTool.Name != "fn" {
			t.Fatalf("tool choice = %#v, want tool fn", req.ToolChoice)
		}
	})

	t.Run("forced with multiple tools falls back to any", func(t *testing.T) {
		req := &anthropic.MessageNewParams{
			Tools: []anthropic.ToolUnionParam{
				{OfTool: &anthropic.ToolParam{Name: "fn1"}},
				{OfTool: &anthropic.ToolParam{Name: "fn2"}},
			},
		}
		err := m.populateToolChoice(req, &model.Options{
			AgenticToolChoice: &schema.AgenticToolChoice{Type: schema.ToolChoiceForced},
		})
		if err != nil {
			t.Fatalf("populateToolChoice() error = %v", err)
		}
		if req.ToolChoice.OfAny == nil {
			t.Fatalf("populateToolChoice() did not set any tool choice")
		}
	})

	t.Run("forced tool not found", func(t *testing.T) {
		req := &anthropic.MessageNewParams{
			Tools: []anthropic.ToolUnionParam{
				{OfTool: &anthropic.ToolParam{Name: "fn"}},
			},
		}
		err := m.populateToolChoice(req, &model.Options{
			AgenticToolChoice: &schema.AgenticToolChoice{
				Type: schema.ToolChoiceForced,
				Forced: &schema.AgenticForcedToolChoice{
					Tools: []*schema.AllowedTool{{FunctionName: "other"}},
				},
			},
		})
		if err == nil || err.Error() != `forced tool "other" is not found in the provided tools` {
			t.Fatalf("populateToolChoice() error = %v", err)
		}
	})
}

func TestPopulateTools(t *testing.T) {
	m := &Model{}
	req := &anthropic.MessageNewParams{}
	options := &model.Options{
		Tools:             []*schema.ToolInfo{testToolInfo("call_tool")},
		DeferredTools:     []*schema.ToolInfo{testToolInfo("deferred_tool")},
		ToolSearchTool:    testToolInfo("client_tool_search"),
		AgenticToolChoice: nil,
	}
	specOptions := &claudeOptions{
		serverTools: []*ServerToolConfig{
			{
				WebFetch20260309: &anthropic.WebFetchTool20260309Param{},
			},
		},
	}

	if err := m.populateTools(req, options, specOptions); err != nil {
		t.Fatalf("populateTools() error = %v", err)
	}
	if len(req.Tools) != 4 {
		t.Fatalf("len(req.Tools) = %d, want 4", len(req.Tools))
	}

	gotByName := make(map[string]anthropic.ToolUnionParam, len(req.Tools))
	foundWebFetch := false
	for _, tool := range req.Tools {
		name := tool.GetName()
		if tool.OfWebFetchTool20260309 != nil {
			foundWebFetch = true
			continue
		}
		if name == nil {
			t.Fatalf("tool name is nil for %#v", tool)
		}
		gotByName[*name] = tool
	}

	if gotByName["call_tool"].OfTool == nil || gotByName["call_tool"].OfTool.DeferLoading.Valid() {
		t.Fatalf("call_tool defer_loading = %#v, want omitted", gotByName["call_tool"].OfTool)
	}
	if gotByName["deferred_tool"].OfTool == nil || !gotByName["deferred_tool"].OfTool.DeferLoading.Valid() || !gotByName["deferred_tool"].OfTool.DeferLoading.Value {
		t.Fatalf("deferred_tool defer_loading = %#v, want true", gotByName["deferred_tool"].OfTool)
	}
	if gotByName["client_tool_search"].OfTool == nil {
		t.Fatalf("client_tool_search tool = %#v, want custom function tool", gotByName["client_tool_search"])
	}
	if !foundWebFetch {
		t.Fatalf("req.Tools = %#v, want one web_fetch server tool", req.Tools)
	}
}

func TestServerToolSelection(t *testing.T) {
	t.Run("toServerTools keeps selected versions", func(t *testing.T) {
		first := &anthropic.WebFetchTool20260309Param{AllowedDomains: []string{"first.example"}}
		second := &anthropic.WebSearchTool20260209Param{AllowedDomains: []string{"search.example"}}
		tools, err := toServerTools([]*ServerToolConfig{
			{WebFetch20260309: first},
			{WebSearch20260209: second},
		})
		if err != nil {
			t.Fatalf("toServerTools() error = %v", err)
		}
		if len(tools) != 2 {
			t.Fatalf("len(tools) = %d, want 2", len(tools))
		}
		if tools[0].OfWebSearchTool20260209 != second {
			t.Fatalf("web search tool = %p, want %p", tools[0].OfWebSearchTool20260209, second)
		}
		if tools[1].OfWebFetchTool20260309 != first {
			t.Fatalf("web fetch tool = %p, want %p", tools[1].OfWebFetchTool20260309, first)
		}
	})

	t.Run("collect beta headers only for web fetch", func(t *testing.T) {
		got := collectServerToolBetaHeaders([]*ServerToolConfig{
			{WebSearch20260209: &anthropic.WebSearchTool20260209Param{}},
			{WebFetch20260309: &anthropic.WebFetchTool20260309Param{}},
			{ToolSearchToolBm25_20251119: &anthropic.ToolSearchToolBm25_20251119Param{}},
		})
		if len(got) != 1 || got[0] != betaHeaderWebFetch20260309 {
			t.Fatalf("collectServerToolBetaHeaders() = %#v", got)
		}
	})

	t.Run("split header values trims blanks", func(t *testing.T) {
		got := splitHeaderValues(" foo , , bar,baz ")
		if len(got) != 3 {
			t.Fatalf("len(splitHeaderValues()) = %d, want 3", len(got))
		}
		for _, key := range []string{"foo", "bar", "baz"} {
			if _, ok := got[key]; !ok {
				t.Fatalf("splitHeaderValues() missing %q", key)
			}
		}
	})
}

func TestGenRequestAndOptions(t *testing.T) {
	temperature := float32(0.4)
	topP := float32(0.8)
	modelName := "claude-opus-4.1"
	maxTokens := 2048
	thinking := &anthropic.ThinkingConfigParamUnion{
		OfEnabled: &anthropic.ThinkingConfigEnabledParam{BudgetTokens: 1024},
	}

	m := &Model{
		thinking: thinking,
	}
	options := &model.Options{
		Model:       &modelName,
		MaxTokens:   &maxTokens,
		Temperature: &temperature,
		TopP:        &topP,
		Stop:        []string{"STOP"},
		Tools:       []*schema.ToolInfo{testToolInfo("call_tool")},
	}
	specOptions := &claudeOptions{
		serverTools: []*ServerToolConfig{
			{WebFetch20260309: &anthropic.WebFetchTool20260309Param{}},
		},
		customHeaders: map[string]string{
			headerAnthropicBeta: "custom-beta",
			"x-trace-id":        "trace-1",
		},
		extraFields: map[string]any{
			"extra_body": map[string]any{"enabled": true},
		},
	}

	req, reqOpts, err := m.genParamsAndOptions([]*schema.AgenticMessage{
		schema.SystemAgenticMessage("system prompt"),
		schema.UserAgenticMessage("hello"),
	}, options, specOptions)
	if err != nil {
		t.Fatalf("genParamsAndOptions() error = %v", err)
	}
	if req.Model != anthropic.Model(modelName) || req.MaxTokens != int64(maxTokens) {
		t.Fatalf("request model/max_tokens = (%q, %d)", req.Model, req.MaxTokens)
	}
	if !req.Temperature.Valid() || req.Temperature.Value != float64(temperature) {
		t.Fatalf("temperature = %#v", req.Temperature)
	}
	if !req.TopP.Valid() || req.TopP.Value != float64(topP) {
		t.Fatalf("top_p = %#v", req.TopP)
	}
	if req.Thinking.OfEnabled == nil || req.Thinking.OfEnabled.BudgetTokens != 1024 {
		t.Fatalf("thinking = %#v", req.Thinking)
	}
	if len(req.System) != 1 || req.System[0].Text != "system prompt" {
		t.Fatalf("system = %#v", req.System)
	}
	if len(req.Messages) != 1 || len(req.Tools) != 2 {
		t.Fatalf("messages/tools = (%d, %d)", len(req.Messages), len(req.Tools))
	}
	if len(reqOpts) != 4 {
		t.Fatalf("len(reqOpts) = %d, want 4", len(reqOpts))
	}
}

func TestGetOptions(t *testing.T) {
	m := &Model{
		model:         "claude-sonnet-4",
		maxTokens:     4096,
		stopSequences: []string{"done"},
		customHeaders: map[string]string{"x-default": "1"},
		extraFields:   map[string]any{"foo": "bar"},
	}

	options, specOptions, err := m.getOptions([]model.Option{
		model.WithTemperature(0.2),
		WithServerTools([]*ServerToolConfig{
			{WebSearch20260209: &anthropic.WebSearchTool20260209Param{}},
		}),
		WithCustomHeaders(map[string]string{"x-call": "2"}),
	})
	if err != nil {
		t.Fatalf("getOptions() error = %v", err)
	}
	if options.Model == nil || *options.Model != "claude-sonnet-4" {
		t.Fatalf("options.Model = %#v", options.Model)
	}
	if options.MaxTokens == nil || *options.MaxTokens != 4096 {
		t.Fatalf("options.MaxTokens = %#v", options.MaxTokens)
	}
	if len(specOptions.serverTools) != 1 || specOptions.serverTools[0].WebSearch20260209 == nil {
		t.Fatalf("specOptions.serverTools = %#v", specOptions.serverTools)
	}
	if specOptions.customHeaders["x-call"] != "2" {
		t.Fatalf("specOptions.customHeaders = %#v", specOptions.customHeaders)
	}

	_, _, err = m.getOptions([]model.Option{
		model.WithToolChoice("auto"),
	})
	if err == nil || err.Error() != "'ToolChoice' option is not supported, please use 'AgenticToolChoice'" {
		t.Fatalf("getOptions() error = %v", err)
	}
}

func TestToCallbackConfigAndModelTokenUsage(t *testing.T) {
	req := &anthropic.MessageNewParams{
		Model:       "claude-opus-4",
		MaxTokens:   1024,
		Temperature: anthropic.Float(0.3),
		TopP:        anthropic.Float(0.9),
	}
	config := toCallbackConfig(req)
	if config.Model != "claude-opus-4" || config.MaxTokens != 1024 {
		t.Fatalf("config = %#v", config)
	}
	if config.Temperature != 0.3 || config.TopP != 0.9 {
		t.Fatalf("config = %#v", config)
	}

	usage := toModelTokenUsage(&schema.AgenticResponseMeta{
		TokenUsage: &schema.TokenUsage{
			PromptTokens: 10,
			PromptTokenDetails: schema.PromptTokenDetails{
				CachedTokens: 3,
			},
			CompletionTokens: 5,
			CompletionTokensDetails: schema.CompletionTokensDetails{
				ReasoningTokens: 2,
			},
			TotalTokens: 15,
		},
	})
	if usage == nil || usage.PromptTokens != 10 || usage.CompletionTokensDetails.ReasoningTokens != 2 {
		t.Fatalf("usage = %#v", usage)
	}
	if toModelTokenUsage(nil) != nil {
		t.Fatalf("toModelTokenUsage(nil) should return nil")
	}
}

func TestConfigAndModelBasics(t *testing.T) {
	t.Run("config check", func(t *testing.T) {
		err := (&Config{}).check()
		if err == nil || err.Error() != "model is required" {
			t.Fatalf("check() error = %v", err)
		}

		err = (&Config{Model: "claude-sonnet-4"}).check()
		if err == nil || err.Error() != "max_tokens must be positive" {
			t.Fatalf("check() error = %v", err)
		}

		err = (&Config{Model: "claude-sonnet-4", APIKey: "key"}).check()
		if err == nil || err.Error() != "max_tokens must be positive" {
			t.Fatalf("check() error = %v", err)
		}

		if err := (&Config{Model: "claude-sonnet-4", APIKey: "key", MaxTokens: 1024}).check(); err != nil {
			t.Fatalf("check() error = %v", err)
		}
	})

	t.Run("new and basic methods", func(t *testing.T) {
		disableParallelToolUse := true
		thinking := &anthropic.ThinkingConfigParamUnion{
			OfEnabled: &anthropic.ThinkingConfigEnabledParam{BudgetTokens: 1024},
		}

		m, err := New(context.Background(), &Config{
			APIKey:                 "key",
			BaseURL:                "https://example.com",
			Model:                  "claude-sonnet-4",
			MaxTokens:              1024,
			DisableParallelToolUse: &disableParallelToolUse,
			Thinking:               thinking,
		})
		if err != nil {
			t.Fatalf("New() error = %v", err)
		}
		if m.model != "claude-sonnet-4" || m.maxTokens != 1024 {
			t.Fatalf("model = %#v", m)
		}
		if m.thinking != thinking {
			t.Fatalf("thinking = %#v, want %#v", m.thinking, thinking)
		}
		if m.GetType() != implType {
			t.Fatalf("GetType() = %q, want %q", m.GetType(), implType)
		}
		if !m.IsCallbacksEnabled() {
			t.Fatalf("IsCallbacksEnabled() should return true")
		}
	})

	t.Run("new client direct mode", func(t *testing.T) {
		_, err := newClient(context.Background(), &Config{
			APIKey:  "key",
			BaseURL: "https://example.com",
		})
		if err != nil {
			t.Fatalf("newClient() error = %v", err)
		}
	})
}

func TestDirectAnthropicAuthSelection(t *testing.T) {
	t.Run("config auth exists", func(t *testing.T) {
		clearAnthropicAuthEnv(t)
		if !hasDirectAnthropicConfigAuth(&Config{APIKey: "api-key"}) {
			t.Fatalf("hasDirectAnthropicConfigAuth(APIKey) should be true")
		}
		if !hasDirectAnthropicConfigAuth(&Config{AuthToken: "auth-token"}) {
			t.Fatalf("hasDirectAnthropicConfigAuth(AuthToken) should be true")
		}
		if !hasDirectAnthropicConfigAuth(&Config{APIKey: "api-key", AuthToken: "auth-token"}) {
			t.Fatalf("hasDirectAnthropicConfigAuth(both) should be true")
		}
		if hasDirectAnthropicConfigAuth(&Config{}) {
			t.Fatalf("hasDirectAnthropicConfigAuth(empty) should be false")
		}
	})

	t.Run("env auth exists", func(t *testing.T) {
		clearAnthropicAuthEnv(t)
		t.Setenv("ANTHROPIC_API_KEY", "env-api-key")
		m, err := New(context.Background(), &Config{
			Model:     "claude-sonnet-4",
			MaxTokens: 1024,
		})
		if err != nil {
			t.Fatalf("New() error = %v", err)
		}
		if m == nil {
			t.Fatalf("New() returned nil model")
		}
	})

	t.Run("missing auth still allows creation", func(t *testing.T) {
		clearAnthropicAuthEnv(t)
		m, err := New(context.Background(), &Config{
			Model:     "claude-sonnet-4",
			MaxTokens: 1024,
		})
		if err != nil {
			t.Fatalf("New() error = %v", err)
		}
		if m == nil {
			t.Fatalf("New() returned nil model")
		}
	})

	t.Run("config auth shields env auth", func(t *testing.T) {
		clearAnthropicAuthEnv(t)
		t.Setenv("ANTHROPIC_AUTH_TOKEN", "env-auth-token")

		var gotAuthorization, gotAPIKey string
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			gotAuthorization = r.Header.Get("Authorization")
			gotAPIKey = r.Header.Get("X-Api-Key")
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
		}))
		defer srv.Close()

		m, err := New(context.Background(), &Config{
			BaseURL:   srv.URL,
			APIKey:    "config-api-key",
			Model:     "claude-sonnet-4",
			MaxTokens: 1024,
		})
		if err != nil {
			t.Fatalf("New() error = %v", err)
		}

		_, err = m.Generate(context.Background(), []*schema.AgenticMessage{
			schema.UserAgenticMessage("hello"),
		})
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}

		if gotAPIKey != "config-api-key" {
			t.Fatalf("X-Api-Key = %q, want %q", gotAPIKey, "config-api-key")
		}
		if gotAuthorization != "" {
			t.Fatalf("Authorization = %q, env credential should not leak into the request", gotAuthorization)
		}
	})
}

func clearAnthropicAuthEnv(t *testing.T) {
	t.Helper()
	// Use os.Unsetenv to truly remove the env vars, since the SDK uses
	// os.LookupEnv — an empty string is still "present".
	for _, key := range []string{"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "ANTHROPIC_BASE_URL"} {
		prev, existed := os.LookupEnv(key)
		os.Unsetenv(key)
		if existed {
			t.Cleanup(func() { os.Setenv(key, prev) })
		} else {
			t.Cleanup(func() { os.Unsetenv(key) })
		}
	}
}

func TestOptionAndUtilsHelpers(t *testing.T) {
	t.Run("impl specific options", func(t *testing.T) {
		serverTools := []*ServerToolConfig{
			{WebFetch20260309: &anthropic.WebFetchTool20260309Param{}},
		}
		m := &Model{model: "claude", maxTokens: 100}
		_, specOptions, err := m.getOptions([]model.Option{
			WithServerTools(serverTools),
			WithCustomHeaders(map[string]string{"x-trace-id": "trace-1"}),
			WithExtraFields(map[string]any{"foo": "bar"}),
		})
		if err != nil {
			t.Fatalf("getOptions() error = %v", err)
		}
		if len(specOptions.serverTools) != 1 || specOptions.serverTools[0].WebFetch20260309 == nil {
			t.Fatalf("specOptions.serverTools = %#v", specOptions.serverTools)
		}
		if specOptions.customHeaders["x-trace-id"] != "trace-1" {
			t.Fatalf("specOptions.customHeaders = %#v", specOptions.customHeaders)
		}
		if specOptions.extraFields["foo"] != "bar" {
			t.Fatalf("specOptions.extraFields = %#v", specOptions.extraFields)
		}
	})

	t.Run("env fallbacks", func(t *testing.T) {
		t.Setenv("AGENTIC_CLAUDE_PRIMARY", "")
		t.Setenv("AGENTIC_CLAUDE_SECONDARY", "fallback")
		if got := getEnvWithFallbacks("AGENTIC_CLAUDE_PRIMARY", "AGENTIC_CLAUDE_SECONDARY"); got != "fallback" {
			t.Fatalf("getEnvWithFallbacks() = %q", got)
		}
	})

	t.Run("panic error", func(t *testing.T) {
		err := newPanicErr("boom", []byte("stack"))
		if err == nil || err.Error() != "panic error: boom,\nstack: stack" {
			t.Fatalf("newPanicErr() = %v", err)
		}
	})

	t.Run("newClaudeOpt helpers", func(t *testing.T) {
		value := true
		if !newClaudeOpt(&value).Valid() || !newClaudeOpt(&value).Value {
			t.Fatalf("newClaudeOpt() should keep bool value")
		}
		if newClaudeOpt[bool](nil).Valid() {
			t.Fatalf("newClaudeOpt(nil) should be invalid")
		}
		if !newClaudeStrOpt("name").Valid() || newClaudeStrOpt("name").Value != "name" {
			t.Fatalf("newClaudeStrOpt() should keep string value")
		}
		if newClaudeStrOpt("").Valid() {
			t.Fatalf("newClaudeStrOpt(\"\") should be invalid")
		}
	})
}

func testToolInfo(name string) *schema.ToolInfo {
	return &schema.ToolInfo{
		Name:        name,
		Desc:        name + " desc",
		ParamsOneOf: schema.NewParamsOneOfByJSONSchema(&jsonschema.Schema{Type: "object"}),
	}
}

func TestCacheControl(t *testing.T) {
	t.Run("cacheControl nil does not set req.CacheControl", func(t *testing.T) {
		m := &Model{}
		if m.cacheControl != nil {
			t.Fatalf("cacheControl should be nil")
		}
	})

	t.Run("cacheControl non-nil is stored", func(t *testing.T) {
		ctrl := anthropic.NewCacheControlEphemeralParam()
		ctrl.TTL = anthropic.CacheControlEphemeralTTLTTL1h
		m := &Model{cacheControl: &ctrl}
		if m.cacheControl == nil {
			t.Fatalf("cacheControl should be non-nil")
		}
		if m.cacheControl.TTL != anthropic.CacheControlEphemeralTTLTTL1h {
			t.Fatalf("cacheControl.TTL = %q, want \"1h\"", m.cacheControl.TTL)
		}
	})
}

func TestSetContentBlockCacheControl(t *testing.T) {
	t.Run("set and read", func(t *testing.T) {
		block := schema.NewContentBlock(&schema.UserInputText{Text: "hello"})
		ctrl := anthropic.NewCacheControlEphemeralParam()
		ctrl.TTL = anthropic.CacheControlEphemeralTTLTTL1h
		block = SetContentBlockCacheControl(block, &ctrl)

		if !hasCacheControlOnContentBlock(block) {
			t.Fatalf("block should have cache control set")
		}
		got := getContentBlockCacheControl(block)
		if got == nil || got.TTL != anthropic.CacheControlEphemeralTTLTTL1h {
			t.Fatalf("getContentBlockCacheControl() = %#v", got)
		}
	})

	t.Run("nil ctrl does nothing", func(t *testing.T) {
		block := schema.NewContentBlock(&schema.UserInputText{Text: "hello"})
		block = SetContentBlockCacheControl(block, nil)

		if hasCacheControlOnContentBlock(block) {
			t.Fatalf("block should not have cache control set with nil ctrl")
		}
	})

	t.Run("nil block is safe", func(t *testing.T) {
		ctrl := anthropic.NewCacheControlEphemeralParam()
		SetContentBlockCacheControl(nil, &ctrl)
	})
}

func TestSetToolInfoCacheControl(t *testing.T) {
	t.Run("set and read", func(t *testing.T) {
		toolInfo := &schema.ToolInfo{Name: "fn", Desc: "desc"}
		ctrl := anthropic.NewCacheControlEphemeralParam()
		ctrl.TTL = anthropic.CacheControlEphemeralTTLTTL1h
		toolInfo = SetToolInfoCacheControl(toolInfo, &ctrl)

		if !hasCacheControlOnToolInfo(toolInfo) {
			t.Fatalf("tool should have cache control set")
		}
		got := getToolInfoCacheControl(toolInfo)
		if got == nil || got.TTL != anthropic.CacheControlEphemeralTTLTTL1h {
			t.Fatalf("getToolInfoCacheControl() = %#v", got)
		}
	})

	t.Run("nil ctrl does nothing", func(t *testing.T) {
		toolInfo := &schema.ToolInfo{Name: "fn", Desc: "desc"}
		toolInfo = SetToolInfoCacheControl(toolInfo, nil)

		if hasCacheControlOnToolInfo(toolInfo) {
			t.Fatalf("tool should not have cache control set with nil ctrl")
		}
	})

	t.Run("nil toolInfo is safe", func(t *testing.T) {
		ctrl := anthropic.NewCacheControlEphemeralParam()
		SetToolInfoCacheControl(nil, &ctrl)
	})

	t.Run("propagated to tool param in toFunctionTools", func(t *testing.T) {
		toolInfo := testToolInfo("fn")
		ctrl := anthropic.NewCacheControlEphemeralParam()
		ctrl.TTL = anthropic.CacheControlEphemeralTTLTTL5m
		toolInfo = SetToolInfoCacheControl(toolInfo, &ctrl)

		tools, err := toFunctionTools([]*schema.ToolInfo{toolInfo})
		if err != nil {
			t.Fatalf("toFunctionTools() error = %v", err)
		}
		if tools[0].OfTool.CacheControl.Type != "ephemeral" {
			t.Fatalf("tool param cache_control.type = %q, want \"ephemeral\"", tools[0].OfTool.CacheControl.Type)
		}
		if tools[0].OfTool.CacheControl.TTL != "5m" {
			t.Fatalf("tool param cache_control.ttl = %q, want \"5m\"", tools[0].OfTool.CacheControl.TTL)
		}
	})

	t.Run("tool without cache control has no cache_control in param", func(t *testing.T) {
		toolInfo := testToolInfo("fn")

		tools, err := toFunctionTools([]*schema.ToolInfo{toolInfo})
		if err != nil {
			t.Fatalf("toFunctionTools() error = %v", err)
		}
		if tools[0].OfTool.CacheControl.Type != "" {
			t.Fatalf("tool param should not have cache_control, got type=%q", tools[0].OfTool.CacheControl.Type)
		}
	})

	t.Run("tool without params one of keeps empty input schema", func(t *testing.T) {
		tools, err := toFunctionTools([]*schema.ToolInfo{{
			Name: "fn",
			Desc: "desc",
		}})
		if err != nil {
			t.Fatalf("toFunctionTools() error = %v", err)
		}
		if len(tools) != 1 || tools[0].OfTool == nil {
			t.Fatalf("toFunctionTools() = %#v", tools)
		}
		if tools[0].OfTool.InputSchema.Properties != nil {
			t.Fatalf("input schema properties = %#v, want nil", tools[0].OfTool.InputSchema.Properties)
		}
		if len(tools[0].OfTool.InputSchema.Required) != 0 {
			t.Fatalf("input schema required = %#v, want empty", tools[0].OfTool.InputSchema.Required)
		}
	})
}

func TestManualCacheControlInConvertors(t *testing.T) {
	t.Run("toSystemBlocks propagates manual cache control", func(t *testing.T) {
		block := schema.NewContentBlock(&schema.UserInputText{Text: "sys"})
		ctrl := anthropic.NewCacheControlEphemeralParam()
		ctrl.TTL = anthropic.CacheControlEphemeralTTLTTL1h
		block = SetContentBlockCacheControl(block, &ctrl)
		msg := &schema.AgenticMessage{
			Role:          schema.AgenticRoleTypeSystem,
			ContentBlocks: []*schema.ContentBlock{block},
		}

		blocks, err := toSystemBlocks(msg)
		if err != nil {
			t.Fatalf("toSystemBlocks() error = %v", err)
		}
		if blocks[0].CacheControl.Type != "ephemeral" {
			t.Fatalf("system block cache_control.type = %q, want \"ephemeral\"", blocks[0].CacheControl.Type)
		}
		if blocks[0].CacheControl.TTL != "1h" {
			t.Fatalf("system block cache_control.ttl = %q, want \"1h\"", blocks[0].CacheControl.TTL)
		}
	})

	t.Run("toUserMessageParam propagates manual cache control", func(t *testing.T) {
		block := schema.NewContentBlock(&schema.UserInputText{Text: "hello"})
		ctrl := anthropic.NewCacheControlEphemeralParam()
		ctrl.TTL = anthropic.CacheControlEphemeralTTLTTL5m
		block = SetContentBlockCacheControl(block, &ctrl)
		msg := &schema.AgenticMessage{
			Role:          schema.AgenticRoleTypeUser,
			ContentBlocks: []*schema.ContentBlock{block},
		}

		param, err := toUserMessageParam(msg)
		if err != nil {
			t.Fatalf("toUserMessageParam() error = %v", err)
		}
		if param.Content[0].OfText.CacheControl.Type != "ephemeral" {
			t.Fatalf("user block cache_control.type = %q, want \"ephemeral\"", param.Content[0].OfText.CacheControl.Type)
		}
		if param.Content[0].OfText.CacheControl.TTL != "5m" {
			t.Fatalf("user block cache_control.ttl = %q, want \"5m\"", param.Content[0].OfText.CacheControl.TTL)
		}
	})

	t.Run("toAssistantMessageParam propagates manual cache control", func(t *testing.T) {
		block := schema.NewContentBlock(&schema.AssistantGenText{Text: "hi"})
		ctrl := anthropic.NewCacheControlEphemeralParam()
		block = SetContentBlockCacheControl(block, &ctrl)
		msg := &schema.AgenticMessage{
			Role:          schema.AgenticRoleTypeAssistant,
			ContentBlocks: []*schema.ContentBlock{block},
		}

		param, err := toAssistantMessageParam(msg)
		if err != nil {
			t.Fatalf("toAssistantMessageParam() error = %v", err)
		}
		if param.Content[0].OfText.CacheControl.Type != "ephemeral" {
			t.Fatalf("assistant block cache_control.type = %q, want \"ephemeral\"", param.Content[0].OfText.CacheControl.Type)
		}
	})

	t.Run("blocks without manual cache control are not affected", func(t *testing.T) {
		msg := &schema.AgenticMessage{
			Role: schema.AgenticRoleTypeUser,
			ContentBlocks: []*schema.ContentBlock{
				schema.NewContentBlock(&schema.UserInputText{Text: "hello"}),
			},
		}

		param, err := toUserMessageParam(msg)
		if err != nil {
			t.Fatalf("toUserMessageParam() error = %v", err)
		}
		if param.Content[0].OfText.CacheControl.Type != "" {
			t.Fatalf("block should not have cache_control, got type=%q", param.Content[0].OfText.CacheControl.Type)
		}
	})
}

func TestStreamServerToolCall(t *testing.T) {
	cases := []struct {
		name      ServerToolName
		fragments []string
		want      *ServerToolCallArguments
	}{
		{
			name:      ServerToolNameWebSearch,
			fragments: []string{`{"query":"golang"}`},
			want:      &ServerToolCallArguments{WebSearch: &WebSearchArguments{Query: "golang"}},
		},
		{
			name:      ServerToolNameWebSearch,
			fragments: []string{"", `{"query`, `":"gola`, `ng"}`},
			want:      &ServerToolCallArguments{WebSearch: &WebSearchArguments{Query: "golang"}},
		},
		{
			name:      ServerToolNameWebFetch,
			fragments: []string{`{"url":"https://exa`, `mple.com"}`},
			want:      &ServerToolCallArguments{WebFetch: &WebFetchArguments{URL: "https://example.com"}},
		},
		{
			name:      ServerToolNameCodeExecution,
			fragments: []string{`{"code":"print(`, `1)"}`},
			want:      &ServerToolCallArguments{CodeExecution: &CodeExecutionArguments{Code: "print(1)"}},
		},
		{
			name:      ServerToolNameBashCodeExecution,
			fragments: []string{`{"command":`, `"ls -la"}`},
			want:      &ServerToolCallArguments{BashCodeExecution: &BashCodeExecutionArguments{Command: "ls -la"}},
		},
		{
			name:      ServerToolNameTextEditorCodeExecution,
			fragments: []string{`{"command":"view",`, `"path":"/tmp/a.txt"}`},
			want: &ServerToolCallArguments{TextEditorCodeExecution: &TextEditorCodeExecutionArguments{
				Command: "view",
				Path:    "/tmp/a.txt",
			}},
		},
		{
			name:      ServerToolNameToolSearchToolBm25,
			fragments: []string{`{"query":"find `, `tools"}`},
			want:      &ServerToolCallArguments{ToolSearchToolBm25: &ToolSearchToolBm25Arguments{Query: "find tools"}},
		},
		{
			name:      ServerToolNameToolSearchToolRegex,
			fragments: []string{`{"query":"find`, `.*"}`},
			want:      &ServerToolCallArguments{ToolSearchToolRegex: &ToolSearchToolRegexArguments{Query: "find.*"}},
		},
		{
			name:      ServerToolNameWebSearch,
			fragments: nil,
			want:      &ServerToolCallArguments{WebSearch: &WebSearchArguments{}},
		},
	}

	for _, tc := range cases {
		t.Run(fmt.Sprintf("%s/%d fragments", tc.name, len(tc.fragments)), func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = io.WriteString(w, serverToolCallSSE(t, tc.name, tc.fragments))
			}))
			defer srv.Close()

			m, err := New(context.Background(), &Config{
				BaseURL:   srv.URL,
				APIKey:    "test-key",
				Model:     "claude-sonnet-4",
				MaxTokens: 1024,
			})
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}

			sr, err := m.Stream(context.Background(), []*schema.AgenticMessage{schema.UserAgenticMessage("hello")})
			if err != nil {
				t.Fatalf("Stream() error = %v", err)
			}
			defer sr.Close()

			var chunks []*schema.AgenticMessage
			for {
				chunk, err := sr.Recv()
				if errors.Is(err, io.EOF) {
					break
				}
				if err != nil {
					t.Fatalf("Recv() error = %v", err)
				}
				chunks = append(chunks, chunk)
			}

			msg, err := schema.ConcatAgenticMessages(chunks)
			if err != nil {
				t.Fatalf("ConcatAgenticMessages() error = %v", err)
			}
			if len(msg.ContentBlocks) != 1 || msg.ContentBlocks[0].ServerToolCall == nil {
				t.Fatalf("content blocks = %#v", msg.ContentBlocks)
			}
			call := msg.ContentBlocks[0].ServerToolCall
			if call.CallID != "srvtoolu_1" || call.Name != string(tc.name) {
				t.Fatalf("server tool call = %#v", call)
			}
			if !reflect.DeepEqual(call.Arguments, tc.want) {
				t.Fatalf("arguments = %#v, want %#v", call.Arguments, tc.want)
			}
		})
	}
}

func serverToolCallSSE(t *testing.T, name ServerToolName, fragments []string) string {
	t.Helper()

	var sb strings.Builder
	writeEvent := func(typ string, data any) {
		raw, err := json.Marshal(data)
		if err != nil {
			t.Fatalf("json.Marshal() error = %v", err)
		}
		fmt.Fprintf(&sb, "event: %s\ndata: %s\n\n", typ, raw)
	}

	writeEvent("message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-4",
			"content": []any{}, "usage": map[string]any{"input_tokens": 1, "output_tokens": 1},
		},
	})
	writeEvent("content_block_start", map[string]any{
		"type":          "content_block_start",
		"index":         0,
		"content_block": map[string]any{"type": "server_tool_use", "id": "srvtoolu_1", "name": string(name), "input": map[string]any{}},
	})
	for _, f := range fragments {
		writeEvent("content_block_delta", map[string]any{
			"type":  "content_block_delta",
			"index": 0,
			"delta": map[string]any{"type": "input_json_delta", "partial_json": f},
		})
	}
	writeEvent("content_block_stop", map[string]any{"type": "content_block_stop", "index": 0})
	writeEvent("message_delta", map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": "tool_use"},
		"usage": map[string]any{"output_tokens": 5},
	})
	writeEvent("message_stop", map[string]any{"type": "message_stop"})
	return sb.String()
}

func TestGenerateReplaysServerToolBlocksUnchanged(t *testing.T) {
	const responseContent = `[
		{"type":"server_tool_use","id":"srvtoolu_ce","name":"code_execution","input":{"code":"search()"},"caller":{"type":"direct"}},
		{"type":"server_tool_use","id":"srvtoolu_ws","name":"web_search","input":{"query":"nginx"},"caller":{"type":"code_execution_20260120","tool_id":"srvtoolu_ce"}},
		{"type":"web_search_tool_result","tool_use_id":"srvtoolu_ws","caller":{"type":"code_execution_20260120","tool_id":"srvtoolu_ce"},"content":[
			{"type":"web_search_result","title":"nginx","url":"https://nginx.org","encrypted_content":"enc_1","page_age":null},
			{"type":"web_search_result","title":"docs","url":"https://nginx.org/docs","encrypted_content":"enc_2","page_age":"1 day"}
		]},
		{"type":"server_tool_use","id":"srvtoolu_wf","name":"web_fetch","input":{"url":"https://nginx.org"},"caller":{"type":"code_execution_20260120","tool_id":"srvtoolu_ce"}},
		{"type":"text","text":"done"}
	]`

	var bodies [][]byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("io.ReadAll() error = %v", err)
		}
		bodies = append(bodies, body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4","content":` + responseContent + `,"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer srv.Close()

	m, err := New(context.Background(), &Config{
		BaseURL:   srv.URL,
		APIKey:    "test-key",
		Model:     "claude-sonnet-4",
		MaxTokens: 1024,
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	first, err := m.Generate(context.Background(), []*schema.AgenticMessage{schema.UserAgenticMessage("search")})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}

	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(first); err != nil {
		t.Fatalf("gob encode error = %v", err)
	}
	decoded := &schema.AgenticMessage{}
	if err := gob.NewDecoder(&buf).Decode(decoded); err != nil {
		t.Fatalf("gob decode error = %v", err)
	}

	for name, history := range map[string]*schema.AgenticMessage{"in memory": first, "gob": decoded} {
		t.Run(name, func(t *testing.T) {
			bodies = nil
			_, err := m.Generate(context.Background(), []*schema.AgenticMessage{
				schema.UserAgenticMessage("search"),
				history,
				schema.UserAgenticMessage("again"),
			})
			if err != nil {
				t.Fatalf("Generate() error = %v", err)
			}

			var req struct {
				Messages []struct {
					Content []any `json:"content"`
				} `json:"messages"`
			}
			if err := json.Unmarshal(bodies[0], &req); err != nil {
				t.Fatalf("json.Unmarshal(request) error = %v", err)
			}
			var want []any
			if err := json.Unmarshal([]byte(responseContent), &want); err != nil {
				t.Fatalf("json.Unmarshal(response content) error = %v", err)
			}
			for _, block := range want {
				b := block.(map[string]any)
				if b["type"] != "web_search_tool_result" {
					continue
				}
				for _, item := range b["content"].([]any) {
					result := item.(map[string]any)
					if result["page_age"] == nil {
						delete(result, "page_age")
					}
				}
			}
			if got := req.Messages[1].Content; !reflect.DeepEqual(got, want) {
				t.Fatalf("assistant turn = %s, want %s", mustJSON(t, got), mustJSON(t, want))
			}
		})
	}
}
