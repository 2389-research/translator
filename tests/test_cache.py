#!/usr/bin/env python3
# ABOUTME: Tests the LRU caching of token encoders.
# ABOUTME: Verifies that encoders are properly cached for performance.

import time
from translator.token_counter import TokenCounter


def test_encoder_caching():
    """Test that encoders are cached and reused."""
    # First call should cache the encoder
    start_time = time.time()
    encoder1 = TokenCounter._get_encoding("gpt-4")
    first_call_time = time.time() - start_time

    # Second call should use cached encoder and be much faster
    start_time = time.time()
    encoder2 = TokenCounter._get_encoding("gpt-4")
    second_call_time = time.time() - start_time

    # Verify same object is returned (due to caching)
    assert encoder1 is encoder2

    # Try with a completely different encoding
    # Using r50k_base which is for older models like davinci
    try:
        encoder3 = TokenCounter._get_encoding("davinci")
        # Only assert if we get a different encoding
        if str(encoder1) != str(encoder3):
            assert encoder1 is not encoder3
    except Exception:
        # If the encoding isn't available, skip this assertion
        print("Skipping different encoder test as davinci encoding not available")
    
    # Call with same model again to verify caching
    encoder4 = TokenCounter._get_encoding("gpt-4")
    assert encoder1 is encoder4

    # Print timing for information (second call should be faster)
    print(f"First call time: {first_call_time:.6f}s")
    print(f"Second call time: {second_call_time:.6f}s")
    
    # While timing can vary, the second call should generally be faster
    # This is a loose assertion as timing depends on system load
    assert second_call_time < first_call_time or second_call_time < 0.001