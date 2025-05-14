#!/usr/bin/env python3
# ABOUTME: Cost estimation and calculation for OpenAI API usage.
# ABOUTME: Provides functions to estimate and calculate actual costs.

from typing import Dict, Tuple

from translator.config import ModelConfig


class CostEstimator:
    """Cost estimation and calculation for OpenAI API usage."""

    @staticmethod
    def estimate_cost(token_count: int, model: str, with_edit: bool = True, 
                     with_critique: bool = False, critique_loops: int = 1) -> Tuple[float, str]:
        """Estimate the cost of translation based on token count and model.
        
        Args:
            token_count: The number of tokens in the content
            model: The model name to use for translation
            with_edit: Whether to include editing step in estimate
            with_critique: Whether to include critique step in estimate
            critique_loops: Number of critique-revision loops to perform
            
        Returns:
            Tuple containing:
                - Estimated cost as a float
                - Formatted cost string
        """
        # Get cost per 1k tokens
        input_cost = ModelConfig.get_input_cost(model)
        output_cost = ModelConfig.get_output_cost(model)
        
        # Estimate input/output tokens
        input_tokens = token_count
        # Translation typically produces similar token count to original
        output_tokens = token_count
        
        # Start with translation cost
        cost = (
            (input_tokens / 1000) * input_cost +
            (output_tokens / 1000) * output_cost
        )
        
        # If editing is enabled, add its cost
        if with_edit:
            # For the edit, we input both original and translated text
            edit_input_tokens = token_count * 2
            # Output is similar to the translation
            edit_output_tokens = token_count
            
            # Add editing cost
            cost += (
                (edit_input_tokens / 1000) * input_cost +
                (edit_output_tokens / 1000) * output_cost
            )
        
        # If critique is enabled, add its cost (both critique generation and application)
        if with_critique and critique_loops > 0:
            for _ in range(critique_loops):
                # Each critique loop includes:
                
                # 1. For the critique generation, we input both original and translated text
                critique_input_tokens = token_count * 2
                # Critique output is typically longer than the translation (detailed feedback)
                critique_output_tokens = token_count * 1.5
                
                # 2. For applying critique feedback, we input original, translation, and critique
                feedback_input_tokens = token_count * 3.5  # original + translation + critique feedback
                # Output is similar to the translation
                feedback_output_tokens = token_count
                
                # Add critique generation and application costs for this loop
                cost += (
                    (critique_input_tokens / 1000) * input_cost +
                    (critique_output_tokens / 1000) * output_cost +
                    (feedback_input_tokens / 1000) * input_cost +
                    (feedback_output_tokens / 1000) * output_cost
                )
        
        # Format approximate price
        if cost < 0.01:
            cost_str = f"Less than $0.01"
        else:
            cost_str = f"Approximately ${cost:.2f}"
        
        return (cost, cost_str)

    @staticmethod
    def calculate_actual_cost(usage: Dict[str, int], model: str) -> Tuple[float, str]:
        """Calculate the actual cost based on token usage.
        
        Args:
            usage: Dictionary with 'prompt_tokens' and 'completion_tokens' keys
            model: The model name used for translation
            
        Returns:
            Tuple containing:
                - Actual cost as a float
                - Formatted cost string
        """
        # Get cost per 1k tokens
        input_cost = ModelConfig.get_input_cost(model)
        output_cost = ModelConfig.get_output_cost(model)
        
        # Calculate cost
        prompt_cost = (usage["prompt_tokens"] / 1000) * input_cost
        completion_cost = (usage["completion_tokens"] / 1000) * output_cost
        total_cost = prompt_cost + completion_cost
        
        # Format cost string
        if total_cost < 0.01:
            cost_str = f"Less than $0.01"
        else:
            cost_str = f"${total_cost:.4f}"
        
        return (total_cost, cost_str)