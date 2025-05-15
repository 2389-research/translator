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
    def check_token_limits(cls, content: str, model: str, with_edit: bool = True, 
                          with_critique: bool = True, critique_loops: int = 4) -> Tuple[bool, int]:
        """Check if content is within token limits for the model.
        
        Args:
            content: The text content to check
            model: The model name to check against
            with_edit: Whether editing will be performed
            with_critique: Whether critique will be performed
            critique_loops: Number of critique loops planned
            
        Returns:
            Tuple containing:
                - Boolean indicating if the content is within limits
                - The token count
        """
        token_count = cls.count_tokens(content, model)
        
        # Get max tokens for the model
        max_tokens = ModelConfig.get_max_tokens(model)
        
        # Base token usage for translation
        # Translation system prompt (~200 tokens) + content + output content
        estimated_total = 200 + token_count + token_count  # ~2x content size for translation step
        
        # Add editing tokens if enabled
        if with_edit:
            # Editing needs original + translated content + system prompt + output
            edit_tokens = 200 + (token_count * 2) + token_count
            estimated_total += edit_tokens
        
        # Add critique tokens if enabled
        if with_critique and critique_loops > 0:
            for _ in range(critique_loops):
                # Each critique cycle has two steps:
                
                # 1. Critique generation (original + translated + system prompt + output critique)
                critique_tokens = 200 + (token_count * 2) + int(token_count * 1.5)
                
                # 2. Critique application (original + translated + critique + system prompt + output)
                feedback_tokens = 200 + (token_count * 3.5) + token_count
                
                estimated_total += critique_tokens + feedback_tokens
        
        return (estimated_total <= max_tokens, token_count)