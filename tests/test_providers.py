#!/usr/bin/env python3
# ABOUTME: Tests for the OpenAI and Anthropic provider adapters.
# ABOUTME: Verifies request parameters and response parsing without calling real APIs.

from types import SimpleNamespace
from unittest.mock import MagicMock

from translator.config import ModelConfig
from translator.providers import AnthropicProvider, OpenAIProvider


def _openai_response(text):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def test_openai_provider_sends_no_sampling_parameters():
    """Reasoning models reject a non-default temperature, so none is sent."""
    client = MagicMock()
    client.chat.completions.create.return_value = _openai_response("Hola")

    text, usage, error = OpenAIProvider(client).translate_text(
        "Hello", "Spanish", "gpt-6.1-sol", "system prompt"
    )

    assert (text, error) == ("Hola", None)
    assert usage["total_tokens"] == 15
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "gpt-6.1-sol"
    assert "temperature" not in kwargs


class _FakeMessageStream:
    """Stand-in for the SDK's message stream: iterable events plus a final message."""

    def __init__(self, events=(), final_message=None):
        self.events = list(events)
        self.final_message = final_message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self.events)

    def get_final_message(self):
        return self.final_message


def _anthropic_client(stream):
    client = MagicMock()
    client.messages.stream.return_value = stream
    return client


def _final_message(blocks, stop_reason="end_turn"):
    return SimpleNamespace(
        content=blocks,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def test_anthropic_provider_reads_text_by_block_type():
    """Current Claude models put a thinking block before the text."""
    client = _anthropic_client(_FakeMessageStream(final_message=_final_message([
        SimpleNamespace(type="thinking", thinking=""),
        SimpleNamespace(type="text", text="Hola"),
    ])))

    text, usage, error = AnthropicProvider(client).translate_text(
        "Hello", "Spanish", "claude-opus-5-5", "system prompt"
    )

    assert (text, error) == ("Hola", None)
    assert usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def test_anthropic_provider_uses_model_output_limit():
    """max_tokens comes from the model config so thinking plus output fit."""
    client = _anthropic_client(_FakeMessageStream(final_message=_final_message([
        SimpleNamespace(type="text", text="Hola"),
    ])))

    AnthropicProvider(client).translate_text("Hello", "Spanish", "claude-opus-5-5", "system prompt")

    kwargs = client.messages.stream.call_args.kwargs
    assert kwargs["model"] == "claude-opus-5-5"
    assert kwargs["max_tokens"] == ModelConfig.get_output_tokens("claude-opus-5-5")


def test_anthropic_provider_reports_response_without_text():
    """A refusal or empty response is an error, not an empty translation."""
    client = _anthropic_client(_FakeMessageStream(final_message=_final_message(
        [SimpleNamespace(type="thinking", thinking="")], stop_reason="refusal"
    )))

    text, usage, error = AnthropicProvider(client).translate_text(
        "Hello", "Spanish", "claude-opus-5-5", "system prompt"
    )

    assert text is None
    assert "refusal" in error


def test_anthropic_provider_streaming_skips_thinking_deltas():
    """Only text deltas reach the translation and the token callback."""
    events = [
        SimpleNamespace(type="message_start", message=SimpleNamespace(usage=SimpleNamespace(input_tokens=10))),
        SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="thinking_delta", thinking="")),
        SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="signature_delta", signature="abc")),
        SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="text_delta", text="Ho")),
        SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="text_delta", text="la")),
        SimpleNamespace(type="message_delta", usage=SimpleNamespace(output_tokens=5)),
    ]
    client = _anthropic_client(_FakeMessageStream(events=events))
    seen_tokens = []

    text, usage, error = AnthropicProvider(client).translate_text(
        "Hello", "Spanish", "claude-opus-5-5", "system prompt",
        stream=True, token_callback=seen_tokens.append,
    )

    assert (text, error) == ("Hola", None)
    assert usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    assert seen_tokens == [1, 1]
