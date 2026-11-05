#!/usr/bin/env python3
# ABOUTME: Tests for the model configuration module.
# ABOUTME: Verifies the supported model lineup, token limits, and pricing lookups.

from translator.config import ModelConfig


def test_models_dict_structure():
    """Test the structure of the MODELS dictionary."""
    assert hasattr(ModelConfig, "MODELS")
    assert isinstance(ModelConfig.MODELS, dict)

    assert "gpt-6.1-sol" in ModelConfig.MODELS
    model_info = ModelConfig.MODELS["gpt-6.1-sol"]
    assert "max_tokens" in model_info
    assert "input_cost" in model_info
    assert "output_cost" in model_info

    assert isinstance(model_info["max_tokens"], int)
    assert isinstance(model_info["input_cost"], float)
    assert isinstance(model_info["output_cost"], float)


def test_openai_lineup():
    """The OpenAI lineup is GPT-6 Astra and GPT-6.1 Sol."""
    assert ModelConfig.get_models_by_provider("openai") == ["gpt-6-astra", "gpt-6.1-sol"]


def test_anthropic_lineup():
    """The Anthropic lineup is the current Claude 5 generation."""
    assert ModelConfig.get_models_by_provider("anthropic") == [
        "claude-fable-5-1",
        "claude-opus-5-5",
        "claude-sonnet-5-5",
        "claude-haiku-5-5",
    ]


def test_default_model_is_configured():
    """The built-in default model has pricing and limits on file."""
    assert ModelConfig.DEFAULT_MODEL == "gpt-6.1-sol"
    assert ModelConfig.DEFAULT_MODEL in ModelConfig.MODELS


def test_get_model_info_with_known_model():
    """Test retrieving configuration for a known model."""
    model_info = ModelConfig.get_model_info("gpt-6.1-sol")

    assert isinstance(model_info, dict)
    assert "max_tokens" in model_info
    assert "input_cost" in model_info
    assert "output_cost" in model_info


def test_get_model_info_with_unknown_model():
    """Test retrieving configuration for an unknown model."""
    model_info = ModelConfig.get_model_info("non-existent-model")

    assert isinstance(model_info, dict)
    assert model_info["max_tokens"] == 4000
    assert model_info["input_cost"] == 0.0
    assert model_info["output_cost"] == 0.0


def test_get_max_tokens_with_known_model():
    """Test retrieving max tokens for a known model."""
    assert ModelConfig.get_max_tokens("gpt-6.1-sol") == ModelConfig.MODELS["gpt-6.1-sol"]["max_tokens"]


def test_get_max_tokens_with_unknown_model():
    """Test retrieving max tokens for an unknown model."""
    assert ModelConfig.get_max_tokens("non-existent-model") == 4000


def test_get_output_tokens():
    """Output limits come from the config, with the old 4K default for unknown models."""
    assert ModelConfig.get_output_tokens("claude-opus-5-5") == 128000
    assert ModelConfig.get_output_tokens("gpt-6-astra") == 128000
    assert ModelConfig.get_output_tokens("non-existent-model") == 4096


def test_get_input_cost_with_known_model():
    """Test retrieving input cost for a known model."""
    assert ModelConfig.get_input_cost("gpt-6.1-sol") == ModelConfig.MODELS["gpt-6.1-sol"]["input_cost"]


def test_get_input_cost_with_unknown_model():
    """Test retrieving input cost for an unknown model."""
    assert ModelConfig.get_input_cost("non-existent-model") == 0.0


def test_get_output_cost_with_known_model():
    """Test retrieving output cost for a known model."""
    assert ModelConfig.get_output_cost("gpt-6.1-sol") == ModelConfig.MODELS["gpt-6.1-sol"]["output_cost"]


def test_get_output_cost_with_unknown_model():
    """Test retrieving output cost for an unknown model."""
    assert ModelConfig.get_output_cost("non-existent-model") == 0.0


def test_get_provider():
    """Provider lookup works for both vendors and reports unknown models."""
    assert ModelConfig.get_provider("gpt-6-astra") == "openai"
    assert ModelConfig.get_provider("claude-opus-5-5") == "anthropic"
    assert ModelConfig.get_provider("non-existent-model") == "unknown"
    assert ModelConfig.is_openai_model("gpt-6.1-sol")
    assert ModelConfig.is_anthropic_model("claude-haiku-5-5")


def test_flagships_cost_more_than_workhorses():
    """Pricing follows each vendor's tiering."""
    assert ModelConfig.get_input_cost("gpt-6-astra") > ModelConfig.get_input_cost("gpt-6.1-sol")
    assert ModelConfig.get_output_cost("gpt-6-astra") > ModelConfig.get_output_cost("gpt-6.1-sol")

    tiers = ["claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
    for pricier, cheaper in zip(tiers, tiers[1:]):
        assert ModelConfig.get_input_cost(pricier) > ModelConfig.get_input_cost(cheaper)
        assert ModelConfig.get_output_cost(pricier) > ModelConfig.get_output_cost(cheaper)


def test_list_all_models():
    """Test listing all available models."""
    all_models = ModelConfig.list_all_models()

    assert all_models == ModelConfig.MODELS
    assert isinstance(all_models, dict)
    for model in ["gpt-6-astra", "gpt-6.1-sol", "claude-opus-5-5"]:
        assert model in all_models


def test_model_info_value_types():
    """Test that model info values have the correct types."""
    for model_name, model_info in ModelConfig.MODELS.items():
        assert isinstance(model_info, dict)

        assert isinstance(model_info["max_tokens"], int)
        assert model_info["max_tokens"] > 0

        assert isinstance(model_info["output_tokens"], int)
        assert model_info["output_tokens"] > 0

        assert isinstance(model_info["input_cost"], float)
        assert model_info["input_cost"] >= 0

        assert isinstance(model_info["output_cost"], float)
        assert model_info["output_cost"] >= 0

        assert model_info["provider"] in ("openai", "anthropic")


def test_pricing_consistency():
    """Output tokens never cost less than input tokens."""
    for model_name, model_info in ModelConfig.MODELS.items():
        assert model_info["output_cost"] >= model_info["input_cost"]
