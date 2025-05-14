#!/usr/bin/env python3
# ABOUTME: Token counting utilities for estimating OpenAI API usage.
# ABOUTME: Provides functions to count tokens and check token limits.

from typing import Tuple

import tiktoken

from translator.config import ModelConfig


class TokenCounter:
    """Token counting utilities for OpenAI API usage."""

    @staticmethod
    def count_tokens(text: str, model: str) -> int:
        """Count the number of tokens in a text string for a specific model.
        
        Args:
            text: The text to count tokens for
            model: The model name to use for counting
            
        Returns:
            The number of tokens in the text
        """
        try:
            # Use gpt-4 encoder for o3 model
            model_name = model
            if model == "o3":
                model_name = "gpt-4"
                
            encoding = tiktoken.encoding_for_model(model_name)
            return len(encoding.encode(text))
        except Exception:
            # Fallback to cl100k_base if model-specific encoding not found
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))

    @classmethod
    def check_token_limits(cls, content: str, model: str) -> Tuple[bool, int]:
        """Check if content is within token limits for the model.
        
        Args:
            content: The text content to check
            model: The model name to check against
            
        Returns:
            Tuple containing:
                - Boolean indicating if the content is within limits
                - The token count
        """
        token_count = cls.count_tokens(content, model)
        
        # Get max tokens for the model
        max_tokens = ModelConfig.get_max_tokens(model)
        
        # We need room for system prompt, translation, and response
        # Estimate total tokens needed as 2.5x the input content
        estimated_total = token_count * 2.5
        
        return (estimated_total <= max_tokens, token_count)