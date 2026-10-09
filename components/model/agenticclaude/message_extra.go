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
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/cloudwego/eino/compose"
	"github.com/cloudwego/eino/schema"
)

func init() {
	// A stream may report the container more than once; the latest report wins.
	compose.RegisterStreamChunkConcatFunc(func(chunks []*Container) (*Container, error) {
		var final *Container
		for _, c := range chunks {
			if c != nil {
				final = c
			}
		}
		return final, nil
	})
}

// Container is the code execution container a response ran in.
type Container struct {
	ID        string    `json:"id"`
	ExpiresAt time.Time `json:"expires_at"`
}

// GetCacheCreationInputTokens returns the Anthropic cache_creation_input_tokens
// count from the message. This is the number of input tokens written into a new
// prompt cache entry, billed at ~125% of the base input token rate.
//
// The cache-read side is available through the standard token usage path:
//
//	msg.ResponseMeta.TokenUsage.PromptTokenDetails.CachedTokens
//
// When streaming, Anthropic reports this once per request (on message_start or
// message_delta). schema.ConcatAgenticMessages merges Extra maps with a
// last-value-wins policy for int, so the final concatenated message carries
// the correct count without accumulation.
func GetCacheCreationInputTokens(msg *schema.AgenticMessage) (int, bool) {
	if msg == nil || msg.Extra == nil {
		return 0, false
	}
	v, ok := msg.Extra[keyOfCacheCreationInputTokens].(int)
	return v, ok
}

// CacheCreationByTTL splits cache_creation_input_tokens by cache TTL.
// 1h cache writes are billed at a higher rate than 5m writes.
type CacheCreationByTTL struct {
	Ephemeral5mInputTokens int
	Ephemeral1hInputTokens int
}

// GetCacheCreationByTTL returns Anthropic's usage.cache_creation breakdown from the message.
// ok is false when neither TTL reported any tokens.
func GetCacheCreationByTTL(msg *schema.AgenticMessage) (CacheCreationByTTL, bool) {
	if msg == nil || msg.Extra == nil {
		return CacheCreationByTTL{}, false
	}
	v5m, ok5m := msg.Extra[keyOfCacheCreationEphemeral5mInputTokens].(int)
	v1h, ok1h := msg.Extra[keyOfCacheCreationEphemeral1hInputTokens].(int)
	return CacheCreationByTTL{Ephemeral5mInputTokens: v5m, Ephemeral1hInputTokens: v1h}, ok5m || ok1h
}

// ServerToolUsage holds Anthropic's usage.server_tool_use request counts, which are billed per request.
type ServerToolUsage struct {
	WebSearchRequests int
	WebFetchRequests  int
}

// GetServerToolUsage returns Anthropic's usage.server_tool_use from the message.
// ok is false when no server tool requests were reported.
func GetServerToolUsage(msg *schema.AgenticMessage) (ServerToolUsage, bool) {
	if msg == nil || msg.Extra == nil {
		return ServerToolUsage{}, false
	}
	search, okSearch := msg.Extra[keyOfServerToolUseWebSearchRequests].(int)
	fetch, okFetch := msg.Extra[keyOfServerToolUseWebFetchRequests].(int)
	return ServerToolUsage{WebSearchRequests: search, WebFetchRequests: fetch}, okSearch || okFetch
}

// GetContainer returns the code execution container reported by the response,
// present when the turn used code execution (directly, via dynamic filtering or
// programmatic tool calling). Send its ID back with WithContainer to continue
// a turn paused inside it.
func GetContainer(msg *schema.AgenticMessage) (*Container, bool) {
	if msg == nil || msg.Extra == nil {
		return nil, false
	}
	v, ok := msg.Extra[keyOfContainer].(*Container)
	return v, ok && v != nil
}

func setContainer(msg *schema.AgenticMessage, c anthropic.Container) {
	if c.ID == "" {
		return
	}
	if msg.Extra == nil {
		msg.Extra = map[string]any{}
	}
	msg.Extra[keyOfContainer] = &Container{ID: c.ID, ExpiresAt: c.ExpiresAt}
}
