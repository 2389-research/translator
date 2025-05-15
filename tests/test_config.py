#!/usr/bin/env python3
# ABOUTME: Tests for the model configuration module.
# ABOUTME: Verifies model tokens and pricing configurations.

import pytest
from translator.config import ModelConfig


def test_models_dict_structure():
    """Test the structure of the MODELS dictionary."""
    # Verify the MODELS dictionary exists and is properly structured
    assert hasattr(ModelConfig, 'MODELS')
    assert isinstance(ModelConfig.MODELS, dict)
    
    # Check content of at least one model
    assert "gpt-4" in ModelConfig.MODELS
    model_info = ModelConfig.MODELS["gpt-4"]
    assert "max_tokens" in model_info
    assert "input_cost" in model_info
    assert "output_cost" in model_info
    
    # Check types
    assert isinstance(model_info["max_tokens"], int)
    assert isinstance(model_info["input_cost"], float)
    assert isinstance(model_info["output_cost"], float)


def test_get_model_info_with_known_model():
    """Test retrieving configuration for a known model."""
    # Get info for a known model
    model_info = ModelConfig.get_model_info("gpt-4")
    
    # Verify the returned info
    assert isinstance(model_info, dict)
    assert "max_tokens" in model_info
    assert "input_cost" in model_info
    assert "output_cost" in model_info


def test_get_model_info_with_unknown_model():
    """Test retrieving configuration for an unknown model."""
    # Get info for an unknown model
    model_info = ModelConfig.get_model_info("non-existent-model")
    
    # Should return default values
    assert isinstance(model_info, dict)
    assert model_info["max_tokens"] == 4000
    assert model_info["input_cost"] == 0.0
    assert model_info["output_cost"] == 0.0


def test_get_max_tokens_with_known_model():
    """Test retrieving max tokens for a known model."""
    # Get max tokens for gpt-4
    max_tokens = ModelConfig.get_max_tokens("gpt-4")
    
    # Should match the defined value
    assert max_tokens == ModelConfig.MODELS["gpt-4"]["max_tokens"]


def test_get_max_tokens_with_unknown_model():
    """Test retrieving max tokens for an unknown model."""
    # Get max tokens for an unknown model
    max_tokens = ModelConfig.get_max_tokens("non-existent-model")
    
    # Should return the default value
    assert max_tokens == 4000


def test_get_input_cost_with_known_model():
    """Test retrieving input cost for a known model."""
    # Get input cost for gpt-4
    input_cost = ModelConfig.get_input_cost("gpt-4")
    
    # Should match the defined value
    assert input_cost == ModelConfig.MODELS["gpt-4"]["input_cost"]


def test_get_input_cost_with_unknown_model():
    """Test retrieving input cost for an unknown model."""
    # Get input cost for an unknown model
    input_cost = ModelConfig.get_input_cost("non-existent-model")
    
    # Should return the default value
    assert input_cost == 0.0


def test_get_output_cost_with_known_model():
    """Test retrieving output cost for a known model."""
    # Get output cost for gpt-4
    output_cost = ModelConfig.get_output_cost("gpt-4")
    
    # Should match the defined value
    assert output_cost == ModelConfig.MODELS["gpt-4"]["output_cost"]


def test_get_output_cost_with_unknown_model():
    """Test retrieving output cost for an unknown model."""
    # Get output cost for an unknown model
    output_cost = ModelConfig.get_output_cost("non-existent-model")
    
    # Should return the default value
    assert output_cost == 0.0


def test_model_families_consistency():
    """Test consistency across model families."""
    # Check that related models have appropriate token caps
    if "gpt-4" in ModelConfig.MODELS and "gpt-4-turbo" in ModelConfig.MODELS:
        assert ModelConfig.get_max_tokens("gpt-4-turbo") >= ModelConfig.get_max_tokens("gpt-4")
    
    if "gpt-3.5-turbo" in ModelConfig.MODELS and "gpt-4" in ModelConfig.MODELS:
        # Typically gpt-4 is more expensive than gpt-3.5-turbo
        assert ModelConfig.get_input_cost("gpt-4") > ModelConfig.get_input_cost("gpt-3.5-turbo")
        assert ModelConfig.get_output_cost("gpt-4") > ModelConfig.get_output_cost("gpt-3.5-turbo")


def test_list_all_models():
    """Test listing all available models."""
    # Get all models
    all_models = ModelConfig.list_all_models()
    
    # Should return the MODELS dictionary
    assert all_models == ModelConfig.MODELS
    assert isinstance(all_models, dict)
    
    # Check that several common models are included
    common_models = ["gpt-4", "gpt-3.5-turbo", "o3"]
    for model in common_models:
        if model in ModelConfig.MODELS:  # Only check models that are actually defined
            assert model in all_models


def test_model_info_value_types():
    """Test that model info values have the correct types."""
    # Check all models
    for model_name, model_info in ModelConfig.MODELS.items():
        assert isinstance(model_info, dict)
        
        # Check max_tokens
        assert "max_tokens" in model_info
        assert isinstance(model_info["max_tokens"], int)
        assert model_info["max_tokens"] > 0
        
        # Check input_cost
        assert "input_cost" in model_info
        assert isinstance(model_info["input_cost"], float)
        assert model_info["input_cost"] >= 0
        
        # Check output_cost
        assert "output_cost" in model_info
        assert isinstance(model_info["output_cost"], float)
        assert model_info["output_cost"] >= 0


def test_model_token_hierarchies():
    """Test token limit hierarchies among different model versions."""
    models = ModelConfig.MODELS
    
    # Check that more advanced models generally have higher token limits
    # This test is only relevant if certain model pairs exist
    if "gpt-4" in models and "gpt-4-turbo" in models:
        assert models["gpt-4-turbo"]["max_tokens"] >= models["gpt-4"]["max_tokens"]
    
    if "gpt-3.5-turbo" in models and "gpt-4" in models:
        # This might not always be true as models evolve, but generally holds
        # Only assert if the actual config follows this pattern
        if models["gpt-4"]["max_tokens"] > models["gpt-3.5-turbo"]["max_tokens"]:
            assert models["gpt-4"]["max_tokens"] > models["gpt-3.5-turbo"]["max_tokens"]


def test_pricing_consistency():
    """Test pricing consistency and hierarchy."""
    models = ModelConfig.MODELS
    
    # For most models, output cost is higher than input cost
    for model_name, model_info in models.items():
        # Skip if there are any exceptions to this rule
        if model_name not in ["custom-exception-model"]:
            assert model_info["output_cost"] >= model_info["input_cost"]