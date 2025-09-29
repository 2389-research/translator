#!/usr/bin/env python3
# ABOUTME: Configuration for OpenAI models including token limits and pricing.
# ABOUTME: Used for estimating costs and checking model capabilities.

from typing import Dict, Any, List


class ModelConfig:
    """Configuration for OpenAI models including token limits and pricing."""

    # Model configuration: max_tokens and cost per 1k tokens (input/output)
    MODELS: Dict[str, Dict[str, Any]] = {
        # Anthropic Claude 4 Series
        "claude-sonnet-4-5-20250929": {
            "provider": "anthropic",
            "max_tokens": 200000,  # 200K context window
            "output_tokens": 8192,  # Up to 8K output
            "input_cost": 0.004,  # $4.00 / 1M tokens (estimated)
            "output_cost": 0.024,  # $24.00 / 1M tokens (estimated)
            "capabilities": ["multimodal", "computer_use", "function_calling"]
        },
        "claude-sonnet-4-20250514": {
            "provider": "anthropic",
            "max_tokens": 200000,  # 200K context window
            "output_tokens": 8192,  # Up to 8K output
            "input_cost": 0.004,  # $4.00 / 1M tokens (estimated)
            "output_cost": 0.024,  # $24.00 / 1M tokens (estimated)
            "capabilities": ["multimodal", "computer_use", "function_calling"]
        },
        "claude-opus-4-1-20250805": {
            "provider": "anthropic",
            "max_tokens": 200000,  # 200K context window
            "output_tokens": 4096,  # Up to 4K output
            "input_cost": 0.015,  # $15.00 / 1M tokens
            "output_cost": 0.075,  # $75.00 / 1M tokens
            "capabilities": ["multimodal", "function_calling"]
        },
        "claude-opus-4-20250514": {
            "provider": "anthropic",
            "max_tokens": 200000,  # 200K context window
            "output_tokens": 4096,  # Up to 4K output
            "input_cost": 0.015,  # $15.00 / 1M tokens
            "output_cost": 0.075,  # $75.00 / 1M tokens
            "capabilities": ["multimodal", "function_calling"]
        },
        # Anthropic Claude 3 Series
        "claude-3-opus-latest": {
            "provider": "anthropic",
            "max_tokens": 200000,  # 200K context window
            "output_tokens": 4096,  # Up to 4K output
            "input_cost": 0.015,  # $15.00 / 1M tokens
            "output_cost": 0.075,  # $75.00 / 1M tokens
            "capabilities": ["multimodal", "function_calling"]
        },
        "claude-3-opus-20240229": {
            "provider": "anthropic",
            "max_tokens": 200000,  # 200K context window
            "output_tokens": 4096,  # Up to 4K output
            "input_cost": 0.015,  # $15.00 / 1M tokens
            "output_cost": 0.075,  # $75.00 / 1M tokens
            "capabilities": ["multimodal", "function_calling"]
        },
        "claude-3-5-sonnet-latest": {
            "provider": "anthropic",
            "max_tokens": 200000,
            "output_tokens": 8192,
            "input_cost": 0.003,  # $3.00 / 1M tokens
            "output_cost": 0.015,  # $15.00 / 1M tokens
            "capabilities": ["multimodal", "computer_use", "function_calling"]
        },
        "claude-3-5-sonnet-20241022": {
            "provider": "anthropic",
            "max_tokens": 200000,
            "output_tokens": 8192,
            "input_cost": 0.003,  # $3.00 / 1M tokens
            "output_cost": 0.015,  # $15.00 / 1M tokens
            "capabilities": ["multimodal", "computer_use", "function_calling"]
        },
        "claude-3-haiku-20240307": {
            "provider": "anthropic",
            "max_tokens": 200000,
            "output_tokens": 4096,
            "input_cost": 0.00025,  # $0.25 / 1M tokens
            "output_cost": 0.00125,  # $1.25 / 1M tokens
            "capabilities": ["multimodal", "function_calling"]
        },
        # GPT-5 Series (Released August 2025)
        "gpt-5": {
            "provider": "openai",
            "max_tokens": 272000,  # 272K input tokens
            "output_tokens": 128000,  # 128K output tokens (including reasoning)
            "input_cost": 0.00125,  # $1.25 / 1M tokens
            "output_cost": 0.01,  # $10.00 / 1M tokens
            "capabilities": ["multimodal", "reasoning_levels", "function_calling", "structured_outputs"]
        },
        "gpt-5-mini": {
            "provider": "openai",
            "max_tokens": 272000,
            "output_tokens": 128000,
            "input_cost": 0.0005,  # Estimated - competitive with gpt-4o-mini
            "output_cost": 0.002,  # Estimated
            "capabilities": ["multimodal", "reasoning_levels", "function_calling"]
        },
        "gpt-5-nano": {
            "provider": "openai",
            "max_tokens": 272000,
            "output_tokens": 128000,
            "input_cost": 0.0002,  # Estimated - ultra-competitive
            "output_cost": 0.0008,  # Estimated
            "capabilities": ["function_calling", "structured_outputs"]
        },

        # o3/o4 Reasoning Series (2025)
        "o3": {
            "provider": "openai",
            "max_tokens": 128000,
            "input_cost": 0.0011,  # $1.10 / 1M tokens (2025 pricing)
            "output_cost": 0.0044,  # $4.40 / 1M tokens
            "capabilities": ["reasoning_effort", "function_calling", "structured_outputs"]
        },
        "o3-mini": {
            "provider": "openai",
            "max_tokens": 64000,
            "input_cost": 0.00015,  # ~90% cheaper than o3
            "output_cost": 0.0006,
            "capabilities": ["reasoning_effort", "function_calling", "structured_outputs"]
        },
        "o4-mini": {
            "provider": "openai",
            "max_tokens": 200000,
            "input_cost": 0.00015,  # $0.15 / 1M tokens
            "output_cost": 0.0006,  # $0.60 / 1M tokens
            "capabilities": ["reasoning_effort", "function_calling", "structured_outputs"]
        },

        # Legacy o-series
        "o1-pro": {
            "provider": "openai",
            "max_tokens": 200000,
            "input_cost": 0.15,  # $150.00 / 1M tokens
            "output_cost": 0.60,  # $600.00 / 1M tokens
        },

        # GPT-4 Series
        "gpt-4o": {
            "provider": "openai",
            "max_tokens": 128000,
            "input_cost": 0.005,  # $5.00 / 1M tokens
            "output_cost": 0.02,  # $20.00 / 1M tokens
        },
        "gpt-4o-mini": {
            "provider": "openai",
            "max_tokens": 128000,
            "input_cost": 0.00015,  # $0.15 / 1M tokens
            "output_cost": 0.0006,  # $0.60 / 1M tokens
        },
        "gpt-4-turbo": {
            "provider": "openai",
            "max_tokens": 128000,
            "input_cost": 0.01,  # $10.00 / 1M tokens
            "output_cost": 0.03,  # $30.00 / 1M tokens
        },
        "gpt-4": {
            "provider": "openai",
            "max_tokens": 8192,
            "input_cost": 0.03,  # $30.00 / 1M tokens
            "output_cost": 0.06,  # $60.00 / 1M tokens
        },
        "gpt-4-vision-preview": {
            "provider": "openai",
            "max_tokens": 128000,
            "input_cost": 0.01,  # $10.00 / 1M tokens
            "output_cost": 0.03,  # $30.00 / 1M tokens
        },
        "gpt-4.1": {
            "provider": "openai",
            "max_tokens": 1048576,
            "input_cost": 0.002,  # $2.00 / 1M tokens
            "output_cost": 0.008,  # $8.00 / 1M tokens
        },
        "gpt-4.1-mini": {
            "provider": "openai",
            "max_tokens": 1048576,
            "input_cost": 0.0004,  # $0.40 / 1M tokens
            "output_cost": 0.0016,  # $1.60 / 1M tokens
        },
        "gpt-4.1-nano": {
            "provider": "openai",
            "max_tokens": 1048576,
            "input_cost": 0.0001,  # $0.10 / 1M tokens
            "output_cost": 0.0004,  # $0.40 / 1M tokens
        },

        # Legacy GPT-3.5
        "gpt-3.5-turbo": {
            "provider": "openai",
            "max_tokens": 16385,
            "input_cost": 0.0005,  # $0.50 / 1M tokens
            "output_cost": 0.0015,  # $1.50 / 1M tokens
        },
    }

    @classmethod
    def get_model_info(cls, model: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        return cls.MODELS.get(
            model, {"max_tokens": 4000, "input_cost": 0.0, "output_cost": 0.0}
        )

    @classmethod
    def get_max_tokens(cls, model: str) -> int:
        """Get the maximum token limit for a model."""
        return cls.get_model_info(model).get("max_tokens", 4000)

    @classmethod
    def get_input_cost(cls, model: str) -> float:
        """Get the input cost per 1k tokens for a model."""
        return cls.get_model_info(model).get("input_cost", 0.0)

    @classmethod
    def get_output_cost(cls, model: str) -> float:
        """Get the output cost per 1k tokens for a model."""
        return cls.get_model_info(model).get("output_cost", 0.0)

    @classmethod
    def list_all_models(cls) -> Dict[str, Dict[str, Any]]:
        """Get all available models and their configurations."""
        return cls.MODELS

    @classmethod
    def get_available_openai_models(cls) -> List[str]:
        """Get list of available models from OpenAI API.

        Returns:
            List of model IDs available from OpenAI API, or empty list if API call fails.
        """
        try:
            import openai
            client = openai.OpenAI()
            models_response = client.models.list()
            # Filter for models that are likely to be chat completion models
            chat_models = []
            for model in models_response.data:
                model_id = model.id
                # Include GPT models, o-series, and other chat models
                if any(prefix in model_id for prefix in ['gpt-', 'o1-', 'o3-', 'o4-', 'chatgpt']):
                    chat_models.append(model_id)
            return sorted(chat_models)
        except Exception:
            # If API call fails, return empty list
            return []

    @classmethod
    def get_enhanced_model_list(cls) -> List[str]:
        """Get combined list of configured models and available API models.

        Returns:
            Deduplicated list of all available models, prioritizing local configuration.
        """
        local_models = set(cls.MODELS.keys())
        api_models = set(cls.get_available_openai_models())

        # Combine and sort, with local models first
        all_models = list(local_models) + [m for m in sorted(api_models) if m not in local_models]
        return all_models

    @classmethod
    def get_provider(cls, model: str) -> str:
        """Get the provider for a specific model.

        Args:
            model: Model name

        Returns:
            Provider name ('openai', 'anthropic', or 'unknown')
        """
        model_info = cls.get_model_info(model)
        return model_info.get("provider", "unknown")

    @classmethod
    def get_models_by_provider(cls, provider: str) -> List[str]:
        """Get all models for a specific provider.

        Args:
            provider: Provider name ('openai' or 'anthropic')

        Returns:
            List of model names for the specified provider
        """
        return [
            model for model, config in cls.MODELS.items()
            if config.get("provider") == provider
        ]

    @classmethod
    def is_anthropic_model(cls, model: str) -> bool:
        """Check if a model is from Anthropic."""
        return cls.get_provider(model) == "anthropic"

    @classmethod
    def is_openai_model(cls, model: str) -> bool:
        """Check if a model is from OpenAI."""
        return cls.get_provider(model) == "openai"
