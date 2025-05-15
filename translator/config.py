#!/usr/bin/env python3
# ABOUTME: Configuration for OpenAI models including token limits and pricing.
# ABOUTME: Used for estimating costs and checking model capabilities.

from typing import Dict, Any


class ModelConfig:
    """Configuration for OpenAI models including token limits and pricing."""

    # Model configuration: max_tokens and cost per 1k tokens (input/output)
    MODELS: Dict[str, Dict[str, Any]] = {
        "o3": {
            "max_tokens": 128000,
            "input_cost": 0.01,  # $10.00 / 1M tokens
            "output_cost": 0.04,  # $40.00 / 1M tokens
        },
        "gpt-4o": {
            "max_tokens": 128000,
            "input_cost": 0.005,  # $5.00 / 1M tokens
            "output_cost": 0.02,  # $20.00 / 1M tokens
        },
        "gpt-4o-mini": {
            "max_tokens": 128000,
            "input_cost": 0.00015,  # $0.15 / 1M tokens
            "output_cost": 0.0006,  # $0.60 / 1M tokens
        },
        "gpt-4-turbo": {
            "max_tokens": 128000,
            "input_cost": 0.01,  # $10.00 / 1M tokens
            "output_cost": 0.03,  # $30.00 / 1M tokens
        },
        "gpt-4": {
            "max_tokens": 8192,
            "input_cost": 0.03,  # $30.00 / 1M tokens
            "output_cost": 0.06,  # $60.00 / 1M tokens
        },
        "gpt-3.5-turbo": {
            "max_tokens": 16385,
            "input_cost": 0.0005,  # $0.50 / 1M tokens
            "output_cost": 0.0015,  # $1.50 / 1M tokens
        },
        "gpt-4-vision-preview": {
            "max_tokens": 128000,
            "input_cost": 0.01,  # $10.00 / 1M tokens
            "output_cost": 0.03,  # $30.00 / 1M tokens
        },
        "gpt-4.1": {
            "max_tokens": 1048576,
            "input_cost": 0.002,  # $2.00 / 1M tokens
            "output_cost": 0.008,  # $8.00 / 1M tokens
        },
        "gpt-4.1-mini": {
            "max_tokens": 1048576,
            "input_cost": 0.0004,  # $0.40 / 1M tokens
            "output_cost": 0.0016,  # $1.60 / 1M tokens
        },
        "gpt-4.1-nano": {
            "max_tokens": 1048576,
            "input_cost": 0.0001,  # $0.10 / 1M tokens
            "output_cost": 0.0004,  # $0.40 / 1M tokens
        },
        "o1-pro": {
            "max_tokens": 200000,
            "input_cost": 0.15,  # $150.00 / 1M tokens
            "output_cost": 0.60,  # $600.00 / 1M tokens
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
