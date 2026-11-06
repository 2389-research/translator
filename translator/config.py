#!/usr/bin/env python3
# ABOUTME: Configuration for OpenAI and Anthropic models including token limits and pricing.
# ABOUTME: Used for estimating costs and checking model capabilities.

from typing import Dict, Any, List


class ModelConfig:
    """Configuration for OpenAI and Anthropic models including token limits and pricing."""

    # The model used when neither -m nor DEFAULT_MODEL is given
    DEFAULT_MODEL = "gpt-6.1-sol"

    # Model configuration: max_tokens is the context window, output_tokens the
    # per-response limit, and costs are per 1k tokens (input/output)
    MODELS: Dict[str, Dict[str, Any]] = {
        # Anthropic Claude 5 series: 1M context, 128K output, adaptive thinking always on
        "claude-fable-5-1": {
            "provider": "anthropic",
            "max_tokens": 1000000,  # 1M context window
            "output_tokens": 128000,  # 128K output, shared with thinking
            "input_cost": 0.01,  # $10.00 / 1M tokens
            "output_cost": 0.05,  # $50.00 / 1M tokens
            "capabilities": ["multimodal", "function_calling", "adaptive_thinking"]
        },
        "claude-opus-5-5": {
            "provider": "anthropic",
            "max_tokens": 1000000,
            "output_tokens": 128000,
            "input_cost": 0.004,  # $4.00 / 1M tokens
            "output_cost": 0.02,  # $20.00 / 1M tokens
            "capabilities": ["multimodal", "function_calling", "adaptive_thinking"]
        },
        "claude-sonnet-5-5": {
            "provider": "anthropic",
            "max_tokens": 1000000,
            "output_tokens": 128000,
            "input_cost": 0.002,  # $2.00 / 1M tokens
            "output_cost": 0.01,  # $10.00 / 1M tokens
            "capabilities": ["multimodal", "function_calling", "adaptive_thinking"]
        },
        "claude-haiku-5-5": {
            "provider": "anthropic",
            "max_tokens": 1000000,
            "output_tokens": 128000,
            "input_cost": 0.0001,  # $0.10 / 1M tokens for prompts up to 100K ($0.50 beyond)
            "output_cost": 0.0005,  # $0.50 / 1M tokens for prompts up to 100K ($2.50 beyond)
            "capabilities": ["multimodal", "function_calling", "adaptive_thinking"]
        },
        # OpenAI GPT-6 series: 1.05M context (922K input), 128K output, reasoning effort
        "gpt-6-astra": {
            "provider": "openai",
            "max_tokens": 1050000,  # 1.05M context window, 922K of it input
            "output_tokens": 128000,
            "input_cost": 0.01,  # $10.00 / 1M tokens (doubles above 272K input tokens)
            "output_cost": 0.05,  # $50.00 / 1M tokens (1.5x above 272K input tokens)
            "capabilities": ["reasoning_effort"]
        },
        "gpt-6.1-sol": {
            "provider": "openai",
            "max_tokens": 1050000,
            "output_tokens": 128000,
            "input_cost": 0.002,  # $2.00 / 1M tokens (doubles above 272K input tokens)
            "output_cost": 0.01,  # $10.00 / 1M tokens (1.5x above 272K input tokens)
            "capabilities": ["reasoning_effort"]
        },
        "gpt-6.1-terra": {
            "provider": "openai",
            "max_tokens": 1050000,
            "output_tokens": 128000,
            "input_cost": 0.001,  # $1.00 / 1M tokens (doubles above 272K input tokens)
            "output_cost": 0.005,  # $5.00 / 1M tokens (1.5x above 272K input tokens)
            "capabilities": ["reasoning_effort"]
        },
        "gpt-6-luna": {
            "provider": "openai",
            "max_tokens": 1050000,
            "output_tokens": 128000,
            "input_cost": 0.0001,  # $0.10 / 1M tokens (doubles above 272K input tokens)
            "output_cost": 0.0005,  # $0.50 / 1M tokens (1.5x above 272K input tokens)
            "capabilities": ["reasoning_effort"]
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
    def get_output_tokens(cls, model: str) -> int:
        """Get the per-response output token limit for a model."""
        return cls.get_model_info(model).get("output_tokens", 4096)

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
