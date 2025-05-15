#!/usr/bin/env python3
# ABOUTME: Tests for the cost estimator module.
# ABOUTME: Verifies cost estimation and calculation functionality.

from unittest.mock import patch
from translator.cost import CostEstimator
from translator.config import ModelConfig


def test_estimate_cost_basic():
    """Test basic cost estimation."""
    # Call the method with basic parameters
    token_count = 1000
    model = "gpt-4"

    # Get estimated cost
    cost, cost_str = CostEstimator.estimate_cost(token_count, model)

    # Verify results
    assert isinstance(cost, float)
    assert cost > 0
    assert isinstance(cost_str, str)
    assert "$" in cost_str


def test_estimate_cost_with_different_models():
    """Test cost estimation with different models."""
    token_count = 1000

    # Test with multiple models to ensure they give different estimates
    cost_gpt4, _ = CostEstimator.estimate_cost(token_count, "gpt-4")
    cost_gpt35, _ = CostEstimator.estimate_cost(token_count, "gpt-3.5-turbo")
    cost_o3, _ = CostEstimator.estimate_cost(token_count, "o3")

    # Different models should have different costs
    assert cost_gpt4 != cost_gpt35
    assert cost_o3 != cost_gpt4
    assert cost_o3 != cost_gpt35


def test_estimate_cost_with_editing():
    """Test cost estimation with editing enabled."""
    token_count = 1000
    model = "gpt-4"

    # Get cost with editing enabled/disabled
    cost_with_edit, _ = CostEstimator.estimate_cost(token_count, model, with_edit=True)
    cost_without_edit, _ = CostEstimator.estimate_cost(
        token_count, model, with_edit=False
    )

    # Cost with editing should be higher
    assert cost_with_edit > cost_without_edit


def test_estimate_cost_with_critique():
    """Test cost estimation with critique enabled."""
    token_count = 1000
    model = "gpt-4"

    # Get cost with critique enabled/disabled
    cost_with_critique, _ = CostEstimator.estimate_cost(
        token_count, model, with_critique=True
    )
    cost_without_critique, _ = CostEstimator.estimate_cost(
        token_count, model, with_critique=False
    )

    # Cost with critique should be higher
    assert cost_with_critique > cost_without_critique


def test_estimate_cost_with_multiple_critique_loops():
    """Test cost estimation with multiple critique loops."""
    token_count = 1000
    model = "gpt-4"

    # Get cost with different numbers of critique loops
    cost_1_loop, _ = CostEstimator.estimate_cost(
        token_count, model, with_critique=True, critique_loops=1
    )
    cost_2_loops, _ = CostEstimator.estimate_cost(
        token_count, model, with_critique=True, critique_loops=2
    )
    cost_3_loops, _ = CostEstimator.estimate_cost(
        token_count, model, with_critique=True, critique_loops=3
    )

    # More critique loops should result in higher cost
    assert cost_2_loops > cost_1_loop
    assert cost_3_loops > cost_2_loops


def test_estimate_cost_zero_loops():
    """Test cost estimation with zero critique loops."""
    token_count = 1000
    model = "gpt-4"

    # With zero critique loops, should be the same as with_critique=False
    cost_with_zero_loops, _ = CostEstimator.estimate_cost(
        token_count, model, with_critique=True, critique_loops=0
    )
    cost_without_critique, _ = CostEstimator.estimate_cost(
        token_count, model, with_critique=False
    )

    # Should be the same cost
    assert cost_with_zero_loops == cost_without_critique


def test_estimate_cost_with_all_options():
    """Test cost estimation with all options enabled."""
    token_count = 1000
    model = "gpt-4"

    # Get cost with all options
    cost, cost_str = CostEstimator.estimate_cost(
        token_count, model, with_edit=True, with_critique=True, critique_loops=2
    )

    # Verify the formatted string contains the cost
    if cost < 0.01:
        assert cost_str == "Less than $0.01"
    else:
        assert f"${cost:.2f}" in cost_str


def test_estimate_cost_with_mock_costs():
    """Test cost estimation with mocked model costs."""
    token_count = 1000
    model = "test-model"

    # Mock the model costs
    with patch.object(ModelConfig, "get_input_cost", return_value=0.01):
        with patch.object(ModelConfig, "get_output_cost", return_value=0.02):
            # Calculate cost with mock values for translation only (with_edit=False, with_critique=False)
            cost, _ = CostEstimator.estimate_cost(
                token_count, model, with_edit=False, with_critique=False
            )

            # For translation only:
            # Input: 1000 tokens + system prompt (200) at $0.01 per 1k = $0.012
            # Output: 1000 tokens at $0.02 per 1k = $0.02
            # Total: $0.032
            assert cost == 0.032


def test_estimate_cost_small_amount_formatting():
    """Test formatting of very small cost estimates."""
    token_count = 10  # Very small token count
    model = "gpt-3.5-turbo"  # Cheapest model

    # Mock to ensure we get a very small cost
    with patch.object(ModelConfig, "get_input_cost", return_value=0.0001):
        with patch.object(ModelConfig, "get_output_cost", return_value=0.0001):
            # Calculate cost with mocked values
            _, cost_str = CostEstimator.estimate_cost(token_count, model)

            # Very small costs should show as "Less than $0.01"
            assert cost_str == "Less than $0.01"


def test_calculate_actual_cost():
    """Test calculating actual cost from usage data."""
    # Create usage data
    usage = {
        "prompt_tokens": 1000,
        "completion_tokens": 500,
        "total_tokens": 1500,  # This is ignored by the calculation
    }
    model = "gpt-4"

    # Calculate actual cost
    cost, cost_str = CostEstimator.calculate_actual_cost(usage, model)

    # Verify results
    assert isinstance(cost, float)
    assert cost > 0
    assert isinstance(cost_str, str)


def test_calculate_actual_cost_with_different_models():
    """Test cost calculation with different models."""
    usage = {"prompt_tokens": 1000, "completion_tokens": 500}

    # Calculate costs for different models
    cost_gpt4, _ = CostEstimator.calculate_actual_cost(usage, "gpt-4")
    cost_gpt35, _ = CostEstimator.calculate_actual_cost(usage, "gpt-3.5-turbo")
    cost_o3, _ = CostEstimator.calculate_actual_cost(usage, "o3")

    # Different models should have different costs
    assert cost_gpt4 != cost_gpt35
    assert cost_o3 != cost_gpt4
    assert cost_o3 != cost_gpt35


def test_calculate_actual_cost_with_mock_costs():
    """Test cost calculation with mocked model costs."""
    usage = {"prompt_tokens": 1000, "completion_tokens": 500}
    model = "test-model"

    # Mock the model costs
    with patch.object(ModelConfig, "get_input_cost", return_value=0.01):
        with patch.object(ModelConfig, "get_output_cost", return_value=0.02):
            # Calculate cost with mocked values
            cost, _ = CostEstimator.calculate_actual_cost(usage, model)

            # Input: 1000 tokens at $0.01 per 1k = $0.01
            # Output: 500 tokens at $0.02 per 1k = $0.01
            # Total: $0.02
            assert cost == 0.02


def test_calculate_actual_cost_small_amount_formatting():
    """Test formatting of very small actual costs."""
    usage = {"prompt_tokens": 10, "completion_tokens": 5}
    model = "gpt-3.5-turbo"  # Cheapest model

    # Mock to ensure we get a very small cost
    with patch.object(ModelConfig, "get_input_cost", return_value=0.0001):
        with patch.object(ModelConfig, "get_output_cost", return_value=0.0001):
            # Calculate cost with mocked values
            _, cost_str = CostEstimator.calculate_actual_cost(usage, model)

            # Very small costs should show as "Less than $0.01"
            assert cost_str == "Less than $0.01"


def test_calculate_actual_cost_decimal_precision():
    """Test decimal precision in cost calculation."""
    usage = {"prompt_tokens": 1000, "completion_tokens": 500}
    model = "test-model"

    # Mock the model costs to get a non-rounded result
    with patch.object(ModelConfig, "get_input_cost", return_value=0.00123):
        with patch.object(ModelConfig, "get_output_cost", return_value=0.00456):
            # Calculate cost with mocked values
            cost, cost_str = CostEstimator.calculate_actual_cost(usage, model)

            # Input: 1000 tokens at $0.00123 per 1k = $0.00123
            # Output: 500 tokens at $0.00456 per 1k = $0.00228
            # Total: $0.00351

            # Verify the cost calculation precision
            assert round(cost, 5) == 0.00351

            # Verify string formatting to 4 decimal places
            if cost >= 0.01:
                assert f"${cost:.4f}" in cost_str
