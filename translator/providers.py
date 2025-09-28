#!/usr/bin/env python3
# ABOUTME: Multi-provider abstraction for AI translation services.
# ABOUTME: Supports OpenAI and Anthropic models with unified interface.

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import openai
import anthropic

from translator.config import ModelConfig


class AIProvider(ABC):
    """Abstract base class for AI providers."""

    @abstractmethod
    def translate_text(
        self,
        text: str,
        target_language: str,
        model: str,
        system_prompt: str,
        stream: bool = False,
        cancellation_handler=None,
        token_callback=None
    ) -> Tuple[Optional[str], Dict, Optional[str]]:
        """Translate text using the provider's API."""
        pass

    @abstractmethod
    def is_supported_model(self, model: str) -> bool:
        """Check if the model is supported by this provider."""
        pass


class OpenAIProvider(AIProvider):
    """OpenAI API provider implementation."""

    def __init__(self, client: openai.OpenAI):
        self.client = client

    def translate_text(
        self,
        text: str,
        target_language: str,
        model: str,
        system_prompt: str,
        stream: bool = False,
        cancellation_handler=None,
        token_callback=None
    ) -> Tuple[Optional[str], Dict, Optional[str]]:
        """Translate text using OpenAI API."""
        try:
            # Extract actual model name (remove provider prefix if present)
            actual_model = model.split(":", 1)[-1] if ":" in model else model

            # Build API parameters
            params = {
                "model": actual_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Translate this text to {target_language}:\n\n{text}"}
                ],
                "stream": stream
            }

            # Add model-specific parameters
            if actual_model != "o3":
                params["temperature"] = 0.7

            if stream:
                return self._handle_streaming_response(params, cancellation_handler, token_callback)
            else:
                return self._handle_non_streaming_response(params)

        except Exception as e:
            return None, {}, str(e)

    def _handle_streaming_response(self, params, cancellation_handler, token_callback):
        """Handle streaming OpenAI response."""
        try:
            response = self.client.chat.completions.create(**params)
            translated_text = ""

            for chunk in response:
                if cancellation_handler and cancellation_handler.is_cancellation_requested():
                    break

                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    translated_text += content
                    if token_callback:
                        token_callback(1)  # Approximate token count

            # Get final usage stats
            usage = getattr(response, 'usage', {})
            usage_dict = {
                'prompt_tokens': getattr(usage, 'prompt_tokens', 0),
                'completion_tokens': getattr(usage, 'completion_tokens', 0),
                'total_tokens': getattr(usage, 'total_tokens', 0)
            }

            return translated_text, usage_dict, None

        except Exception as e:
            return None, {}, str(e)

    def _handle_non_streaming_response(self, params):
        """Handle non-streaming OpenAI response."""
        try:
            response = self.client.chat.completions.create(**params)
            translated_text = response.choices[0].message.content

            usage_dict = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }

            return translated_text, usage_dict, None

        except Exception as e:
            return None, {}, str(e)

    def is_supported_model(self, model: str) -> bool:
        """Check if model is supported by OpenAI."""
        return ModelConfig.is_openai_model(model)


class AnthropicProvider(AIProvider):
    """Anthropic Claude API provider implementation."""

    def __init__(self, client: anthropic.Anthropic):
        self.client = client

    def translate_text(
        self,
        text: str,
        target_language: str,
        model: str,
        system_prompt: str,
        stream: bool = False,
        cancellation_handler=None,
        token_callback=None
    ) -> Tuple[Optional[str], Dict, Optional[str]]:
        """Translate text using Anthropic Claude API."""
        try:
            # Extract actual model name (remove provider prefix if present)
            actual_model = model.split(":", 1)[-1] if ":" in model else model

            # Build API parameters for Anthropic
            params = {
                "model": actual_model,
                "max_tokens": 4096,  # Claude's output limit
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": f"Translate this text to {target_language}:\n\n{text}"}
                ]
            }

            if stream:
                return self._handle_streaming_response(params, cancellation_handler, token_callback)
            else:
                return self._handle_non_streaming_response(params)

        except Exception as e:
            return None, {}, str(e)

    def _handle_streaming_response(self, params, cancellation_handler, token_callback):
        """Handle streaming Anthropic response."""
        try:
            translated_text = ""
            total_input_tokens = 0
            total_output_tokens = 0

            with self.client.messages.stream(**params) as stream:
                for chunk in stream:
                    if cancellation_handler and cancellation_handler.is_cancellation_requested():
                        break

                    if chunk.type == 'content_block_delta':
                        content = chunk.delta.text
                        translated_text += content
                        if token_callback:
                            token_callback(1)  # Approximate token count
                    elif chunk.type == 'message_start':
                        total_input_tokens = chunk.message.usage.input_tokens
                    elif chunk.type == 'message_delta':
                        total_output_tokens = chunk.delta.usage.output_tokens

            usage_dict = {
                'prompt_tokens': total_input_tokens,
                'completion_tokens': total_output_tokens,
                'total_tokens': total_input_tokens + total_output_tokens
            }

            return translated_text, usage_dict, None

        except Exception as e:
            return None, {}, str(e)

    def _handle_non_streaming_response(self, params):
        """Handle non-streaming Anthropic response."""
        try:
            response = self.client.messages.create(**params)
            translated_text = response.content[0].text

            usage_dict = {
                'prompt_tokens': response.usage.input_tokens,
                'completion_tokens': response.usage.output_tokens,
                'total_tokens': response.usage.input_tokens + response.usage.output_tokens
            }

            return translated_text, usage_dict, None

        except Exception as e:
            return None, {}, str(e)

    def is_supported_model(self, model: str) -> bool:
        """Check if model is supported by Anthropic."""
        return ModelConfig.is_anthropic_model(model)


class ProviderFactory:
    """Factory for creating appropriate AI providers."""

    @staticmethod
    def create_provider(model: str, openai_client=None, anthropic_client=None) -> AIProvider:
        """Create the appropriate provider for the given model.

        Args:
            model: Model name (supports prefixes like "openai:gpt-4" or "anthropic:claude-3")
            openai_client: OpenAI client instance
            anthropic_client: Anthropic client instance

        Returns:
            Appropriate provider instance

        Raises:
            ValueError: If model is not supported or required client is missing
        """
        # Parse model prefix if present (e.g., "openai:gpt-4" -> "openai", "gpt-4")
        if ":" in model:
            provider_prefix, model_name = model.split(":", 1)
            provider_prefix = provider_prefix.lower()
        else:
            # Fall back to existing detection logic for backward compatibility
            provider_prefix = None
            model_name = model

        # Determine provider based on prefix or model detection
        if provider_prefix == "openai" or (provider_prefix is None and ModelConfig.is_openai_model(model_name)):
            if openai_client is None:
                raise ValueError("OpenAI client required for OpenAI models")
            return OpenAIProvider(openai_client)

        elif provider_prefix == "anthropic" or (provider_prefix is None and ModelConfig.is_anthropic_model(model_name)):
            if anthropic_client is None:
                raise ValueError("Anthropic client required for Anthropic models")
            return AnthropicProvider(anthropic_client)

        else:
            raise ValueError(f"Unsupported model: {model}")