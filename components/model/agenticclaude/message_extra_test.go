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
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/cloudwego/eino/schema"
)

func TestGetCacheCreationInputTokens(t *testing.T) {
	t.Run("nil message", func(t *testing.T) {
		tokens, ok := GetCacheCreationInputTokens(nil)
		if ok || tokens != 0 {
			t.Fatalf("expected (0, false), got (%d, %v)", tokens, ok)
		}
	})

	t.Run("nil Extra", func(t *testing.T) {
		msg := &schema.AgenticMessage{}
		tokens, ok := GetCacheCreationInputTokens(msg)
		if ok || tokens != 0 {
			t.Fatalf("expected (0, false), got (%d, %v)", tokens, ok)
		}
	})

	t.Run("key not present", func(t *testing.T) {
		msg := &schema.AgenticMessage{Extra: map[string]any{"other": 42}}
		tokens, ok := GetCacheCreationInputTokens(msg)
		if ok || tokens != 0 {
			t.Fatalf("expected (0, false), got (%d, %v)", tokens, ok)
		}
	})

	t.Run("key present with value", func(t *testing.T) {
		msg := &schema.AgenticMessage{Extra: map[string]any{
			keyOfCacheCreationInputTokens: 1234,
		}}
		tokens, ok := GetCacheCreationInputTokens(msg)
		if !ok || tokens != 1234 {
			t.Fatalf("expected (1234, true), got (%d, %v)", tokens, ok)
		}
	})
}

func TestCacheCreationInputTokens_SetInConvOutputMessage(t *testing.T) {
	t.Run("zero cache creation does not set Extra", func(t *testing.T) {
		var resp anthropic.Message
		if err := json.Unmarshal([]byte(`{
			"id": "msg_1",
			"type": "message",
			"role": "assistant",
			"content": [],
			"usage": {"input_tokens": 100, "output_tokens": 50, "cache_creation_input_tokens": 0}
		}`), &resp); err != nil {
			t.Fatal(err)
		}
		msg, err := toAgenticMessage(&resp)
		if err != nil {
			t.Fatal(err)
		}
		if msg.Extra != nil {
			t.Fatalf("expected nil Extra when cache_creation_input_tokens=0, got %v", msg.Extra)
		}
	})

	t.Run("nonzero cache creation sets Extra", func(t *testing.T) {
		var resp anthropic.Message
		if err := json.Unmarshal([]byte(`{
			"id": "msg_1",
			"type": "message",
			"role": "assistant",
			"content": [],
			"usage": {"input_tokens": 100, "output_tokens": 50, "cache_creation_input_tokens": 500}
		}`), &resp); err != nil {
			t.Fatal(err)
		}
		msg, err := toAgenticMessage(&resp)
		if err != nil {
			t.Fatal(err)
		}
		tokens, ok := GetCacheCreationInputTokens(msg)
		if !ok || tokens != 500 {
			t.Fatalf("expected (500, true), got (%d, %v)", tokens, ok)
		}
	})
}

func TestCacheCreationInputTokens_SetInStreamDelta(t *testing.T) {
	sc := newStreamConverter()

	t.Run("delta with zero cache creation", func(t *testing.T) {
		event := mustUnmarshalStreamEvent(t, `{
			"type": "message_delta",
			"delta": {"stop_reason": "end_turn"},
			"usage": {"output_tokens": 42, "cache_creation_input_tokens": 0}
		}`)
		msg, err := sc.toMessageStreamingChunk(event)
		if err != nil {
			t.Fatal(err)
		}
		if msg.Extra != nil {
			t.Fatalf("expected nil Extra for zero cache_creation, got %v", msg.Extra)
		}
	})

	t.Run("delta with nonzero cache creation", func(t *testing.T) {
		event := mustUnmarshalStreamEvent(t, `{
			"type": "message_delta",
			"delta": {"stop_reason": "end_turn"},
			"usage": {"output_tokens": 42, "cache_creation_input_tokens": 300}
		}`)
		msg, err := sc.toMessageStreamingChunk(event)
		if err != nil {
			t.Fatal(err)
		}
		tokens, ok := GetCacheCreationInputTokens(msg)
		if !ok || tokens != 300 {
			t.Fatalf("expected (300, true), got (%d, %v)", tokens, ok)
		}
	})
}

func TestCacheCreationInputTokens_PreservedAfterConcat(t *testing.T) {
	sc := newStreamConverter()
	event := mustUnmarshalStreamEvent(t, `{
		"type": "message_delta",
		"delta": {"stop_reason": "end_turn"},
		"usage": {"output_tokens": 42, "cache_creation_input_tokens": 300}
	}`)
	usageChunk, err := sc.toMessageStreamingChunk(event)
	if err != nil {
		t.Fatal(err)
	}
	textChunk := newAssistantStreamingChunk(schema.NewContentBlock(&schema.AssistantGenText{Text: "hello"}))

	msg, err := schema.ConcatAgenticMessages([]*schema.AgenticMessage{usageChunk, textChunk})
	if err != nil {
		t.Fatal(err)
	}
	tokens, ok := GetCacheCreationInputTokens(msg)
	if !ok || tokens != 300 {
		t.Fatalf("expected (300, true), got (%d, %v)", tokens, ok)
	}
}

func TestGetContainer(t *testing.T) {
	if c, ok := GetContainer(nil); ok || c != nil {
		t.Fatalf("expected (nil, false), got (%v, %v)", c, ok)
	}
	if c, ok := GetContainer(&schema.AgenticMessage{}); ok || c != nil {
		t.Fatalf("expected (nil, false), got (%v, %v)", c, ok)
	}
}

func TestContainer_SetInConvOutputMessage(t *testing.T) {
	t.Run("no container", func(t *testing.T) {
		var resp anthropic.Message
		if err := json.Unmarshal([]byte(`{"id":"msg_1","type":"message","role":"assistant","content":[],"container":null,"usage":{"input_tokens":1,"output_tokens":1}}`), &resp); err != nil {
			t.Fatal(err)
		}
		msg, err := toAgenticMessage(&resp)
		if err != nil {
			t.Fatal(err)
		}
		if c, ok := GetContainer(msg); ok {
			t.Fatalf("expected no container, got %+v", c)
		}
	})

	t.Run("with container", func(t *testing.T) {
		var resp anthropic.Message
		if err := json.Unmarshal([]byte(`{"id":"msg_1","type":"message","role":"assistant","content":[],"stop_reason":"pause_turn","container":{"id":"container_1","expires_at":"2026-01-02T03:04:05Z"},"usage":{"input_tokens":1,"output_tokens":1}}`), &resp); err != nil {
			t.Fatal(err)
		}
		msg, err := toAgenticMessage(&resp)
		if err != nil {
			t.Fatal(err)
		}
		c, ok := GetContainer(msg)
		want := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)
		if !ok || c.ID != "container_1" || !c.ExpiresAt.Equal(want) {
			t.Fatalf("expected container_1 expiring %v, got (%+v, %v)", want, c, ok)
		}
	})
}

func TestContainer_SetInStream(t *testing.T) {
	sc := newStreamConverter()
	start, err := sc.toMessageStreamingChunk(mustUnmarshalStreamEvent(t, `{
		"type": "message_start",
		"message": {"id":"msg_1","type":"message","role":"assistant","content":[],"container":{"id":"container_1","expires_at":"2026-01-02T03:04:05Z"},"usage":{"input_tokens":1,"output_tokens":0}}
	}`))
	if err != nil {
		t.Fatal(err)
	}
	text := newAssistantStreamingChunk(schema.NewContentBlock(&schema.AssistantGenText{Text: "hi"}))
	delta, err := sc.toMessageStreamingChunk(mustUnmarshalStreamEvent(t, `{
		"type": "message_delta",
		"delta": {"stop_reason": "pause_turn", "container": {"id":"container_2","expires_at":"2026-01-02T03:04:06Z"}},
		"usage": {"output_tokens": 1}
	}`))
	if err != nil {
		t.Fatal(err)
	}
	if c, ok := GetContainer(delta); !ok || c.ID != "container_2" {
		t.Fatalf("expected container_2 on delta, got (%+v, %v)", c, ok)
	}

	msg, err := schema.ConcatAgenticMessages([]*schema.AgenticMessage{start, text, delta})
	if err != nil {
		t.Fatal(err)
	}
	if c, ok := GetContainer(msg); !ok || c.ID != "container_2" {
		t.Fatalf("expected container_2 after concat, got (%+v, %v)", c, ok)
	}
}

func mustUnmarshalStreamEvent(t *testing.T, raw string) anthropic.MessageStreamEventUnion {
	t.Helper()
	var event anthropic.MessageStreamEventUnion
	if err := json.Unmarshal([]byte(raw), &event); err != nil {
		t.Fatalf("json.Unmarshal(event) error = %v", err)
	}
	return event
}

func TestGetCacheCreationByTTL(t *testing.T) {
	t.Run("nil message", func(t *testing.T) {
		got, ok := GetCacheCreationByTTL(nil)
		if ok || got != (CacheCreationByTTL{}) {
			t.Fatalf("expected (zero, false), got (%+v, %v)", got, ok)
		}
	})

	t.Run("key not present", func(t *testing.T) {
		got, ok := GetCacheCreationByTTL(&schema.AgenticMessage{Extra: map[string]any{"other": 42}})
		if ok || got != (CacheCreationByTTL{}) {
			t.Fatalf("expected (zero, false), got (%+v, %v)", got, ok)
		}
	})

	t.Run("only 1h present", func(t *testing.T) {
		got, ok := GetCacheCreationByTTL(&schema.AgenticMessage{Extra: map[string]any{
			keyOfCacheCreationEphemeral1hInputTokens: 700,
		}})
		want := CacheCreationByTTL{Ephemeral1hInputTokens: 700}
		if !ok || got != want {
			t.Fatalf("expected (%+v, true), got (%+v, %v)", want, got, ok)
		}
	})
}

func TestGetServerToolUsage(t *testing.T) {
	t.Run("nil message", func(t *testing.T) {
		got, ok := GetServerToolUsage(nil)
		if ok || got != (ServerToolUsage{}) {
			t.Fatalf("expected (zero, false), got (%+v, %v)", got, ok)
		}
	})

	t.Run("key not present", func(t *testing.T) {
		got, ok := GetServerToolUsage(&schema.AgenticMessage{Extra: map[string]any{"other": 42}})
		if ok || got != (ServerToolUsage{}) {
			t.Fatalf("expected (zero, false), got (%+v, %v)", got, ok)
		}
	})

	t.Run("only web search present", func(t *testing.T) {
		got, ok := GetServerToolUsage(&schema.AgenticMessage{Extra: map[string]any{
			keyOfServerToolUseWebSearchRequests: 3,
		}})
		want := ServerToolUsage{WebSearchRequests: 3}
		if !ok || got != want {
			t.Fatalf("expected (%+v, true), got (%+v, %v)", want, got, ok)
		}
	})
}

func TestUsageDetails_SetInConvOutputMessage(t *testing.T) {
	t.Run("zero usage details do not set Extra", func(t *testing.T) {
		var resp anthropic.Message
		if err := json.Unmarshal([]byte(`{
			"id": "msg_1",
			"type": "message",
			"role": "assistant",
			"content": [],
			"usage": {
				"input_tokens": 100, "output_tokens": 50,
				"cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
				"server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 0}
			}
		}`), &resp); err != nil {
			t.Fatal(err)
		}
		msg, err := toAgenticMessage(&resp)
		if err != nil {
			t.Fatal(err)
		}
		if msg.Extra != nil {
			t.Fatalf("expected nil Extra, got %v", msg.Extra)
		}
	})

	t.Run("nonzero usage details set Extra", func(t *testing.T) {
		var resp anthropic.Message
		if err := json.Unmarshal([]byte(`{
			"id": "msg_1",
			"type": "message",
			"role": "assistant",
			"content": [],
			"usage": {
				"input_tokens": 100, "output_tokens": 50, "cache_creation_input_tokens": 500,
				"cache_creation": {"ephemeral_5m_input_tokens": 200, "ephemeral_1h_input_tokens": 300},
				"server_tool_use": {"web_search_requests": 2, "web_fetch_requests": 1}
			}
		}`), &resp); err != nil {
			t.Fatal(err)
		}
		msg, err := toAgenticMessage(&resp)
		if err != nil {
			t.Fatal(err)
		}
		assertUsageDetails(t, msg,
			CacheCreationByTTL{Ephemeral5mInputTokens: 200, Ephemeral1hInputTokens: 300},
			ServerToolUsage{WebSearchRequests: 2, WebFetchRequests: 1})
	})
}

func TestUsageDetails_SetInStreamChunks(t *testing.T) {
	sc := newStreamConverter()

	t.Run("message_start carries TTL split", func(t *testing.T) {
		event := mustUnmarshalStreamEvent(t, `{
			"type": "message_start",
			"message": {
				"id": "msg_1", "type": "message", "role": "assistant", "content": [],
				"usage": {
					"input_tokens": 10, "output_tokens": 1, "cache_creation_input_tokens": 500,
					"cache_creation": {"ephemeral_5m_input_tokens": 200, "ephemeral_1h_input_tokens": 300}
				}
			}
		}`)
		msg, err := sc.toMessageStreamingChunk(event)
		if err != nil {
			t.Fatal(err)
		}
		got, ok := GetCacheCreationByTTL(msg)
		want := CacheCreationByTTL{Ephemeral5mInputTokens: 200, Ephemeral1hInputTokens: 300}
		if !ok || got != want {
			t.Fatalf("expected (%+v, true), got (%+v, %v)", want, got, ok)
		}
	})

	t.Run("message_delta carries server tool usage", func(t *testing.T) {
		event := mustUnmarshalStreamEvent(t, `{
			"type": "message_delta",
			"delta": {"stop_reason": "end_turn"},
			"usage": {"output_tokens": 42, "server_tool_use": {"web_search_requests": 2, "web_fetch_requests": 1}}
		}`)
		msg, err := sc.toMessageStreamingChunk(event)
		if err != nil {
			t.Fatal(err)
		}
		got, ok := GetServerToolUsage(msg)
		want := ServerToolUsage{WebSearchRequests: 2, WebFetchRequests: 1}
		if !ok || got != want {
			t.Fatalf("expected (%+v, true), got (%+v, %v)", want, got, ok)
		}
	})
}

func TestUsageDetails_PreservedAfterConcat(t *testing.T) {
	sc := newStreamConverter()
	startChunk, err := sc.toMessageStreamingChunk(mustUnmarshalStreamEvent(t, `{
		"type": "message_start",
		"message": {
			"id": "msg_1", "type": "message", "role": "assistant", "content": [],
			"usage": {
				"input_tokens": 10, "output_tokens": 1, "cache_creation_input_tokens": 500,
				"cache_creation": {"ephemeral_5m_input_tokens": 200, "ephemeral_1h_input_tokens": 300},
				"server_tool_use": {"web_search_requests": 1}
			}
		}
	}`))
	if err != nil {
		t.Fatal(err)
	}
	deltaChunk, err := sc.toMessageStreamingChunk(mustUnmarshalStreamEvent(t, `{
		"type": "message_delta",
		"delta": {"stop_reason": "end_turn"},
		"usage": {"output_tokens": 42, "cache_creation_input_tokens": 500, "server_tool_use": {"web_search_requests": 2, "web_fetch_requests": 1}}
	}`))
	if err != nil {
		t.Fatal(err)
	}
	textChunk := newAssistantStreamingChunk(schema.NewContentBlock(&schema.AssistantGenText{Text: "hello"}))

	msg, err := schema.ConcatAgenticMessages([]*schema.AgenticMessage{startChunk, textChunk, deltaChunk})
	if err != nil {
		t.Fatal(err)
	}
	assertUsageDetails(t, msg,
		CacheCreationByTTL{Ephemeral5mInputTokens: 200, Ephemeral1hInputTokens: 300},
		ServerToolUsage{WebSearchRequests: 2, WebFetchRequests: 1})
	if tokens, ok := GetCacheCreationInputTokens(msg); !ok || tokens != 500 {
		t.Fatalf("expected (500, true), got (%d, %v)", tokens, ok)
	}
}

func TestUsageDetails_GenerateAndStream(t *testing.T) {
	wantTTL := CacheCreationByTTL{Ephemeral5mInputTokens: 200, Ephemeral1hInputTokens: 300}
	wantServerToolUsage := ServerToolUsage{WebSearchRequests: 2, WebFetchRequests: 1}

	t.Run("Generate", func(t *testing.T) {
		m := newUsageTestModel(t, func(w http.ResponseWriter) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn",` +
				`"usage":{"input_tokens":1,"output_tokens":1,"cache_creation_input_tokens":500,` +
				`"cache_creation":{"ephemeral_5m_input_tokens":200,"ephemeral_1h_input_tokens":300},` +
				`"server_tool_use":{"web_search_requests":2,"web_fetch_requests":1}}}`))
		})

		msg, err := m.Generate(context.Background(), []*schema.AgenticMessage{schema.UserAgenticMessage("hello")})
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		assertUsageDetails(t, msg, wantTTL, wantServerToolUsage)
	})

	t.Run("Stream", func(t *testing.T) {
		m := newUsageTestModel(t, func(w http.ResponseWriter) {
			w.Header().Set("Content-Type", "text/event-stream")
			for _, e := range []struct{ name, data string }{
				{"message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4","content":[],` +
					`"usage":{"input_tokens":1,"output_tokens":1,"cache_creation_input_tokens":500,` +
					`"cache_creation":{"ephemeral_5m_input_tokens":200,"ephemeral_1h_input_tokens":300}}}}`},
				{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
				{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}`},
				{"content_block_stop", `{"type":"content_block_stop","index":0}`},
				{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},` +
					`"usage":{"output_tokens":5,"cache_creation_input_tokens":500,"server_tool_use":{"web_search_requests":2,"web_fetch_requests":1}}}`},
				{"message_stop", `{"type":"message_stop"}`},
			} {
				_, _ = w.Write([]byte("event: " + e.name + "\ndata: " + e.data + "\n\n"))
			}
		})

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
			t.Fatal(err)
		}
		assertUsageDetails(t, msg, wantTTL, wantServerToolUsage)
	})
}

func newUsageTestModel(t *testing.T, respond func(w http.ResponseWriter)) *Model {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		respond(w)
	}))
	t.Cleanup(srv.Close)

	m, err := New(context.Background(), &Config{
		BaseURL:   srv.URL,
		APIKey:    "test-api-key",
		Model:     "claude-sonnet-4",
		MaxTokens: 1024,
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	return m
}

func assertUsageDetails(t *testing.T, msg *schema.AgenticMessage, wantTTL CacheCreationByTTL, wantServerToolUsage ServerToolUsage) {
	t.Helper()
	ttl, ok := GetCacheCreationByTTL(msg)
	if !ok || ttl != wantTTL {
		t.Fatalf("GetCacheCreationByTTL() = (%+v, %v), want (%+v, true)", ttl, ok, wantTTL)
	}
	stu, ok := GetServerToolUsage(msg)
	if !ok || stu != wantServerToolUsage {
		t.Fatalf("GetServerToolUsage() = (%+v, %v), want (%+v, true)", stu, ok, wantServerToolUsage)
	}
}
