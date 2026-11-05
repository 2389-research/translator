# Gotchas

Short notes for people and agents working in this repo. Add your own; don't rewrite others'.

- **Model names vs IDs.** Harper refers to OpenAI models by marketing name ("Astra", "Sol"). The API IDs are `gpt-6-astra` and `gpt-6.1-sol`; check developers.openai.com/api/docs/models before guessing. The configured lineup lives in `translator/config.py`.
- **Token counts are estimates for current models.** tiktoken has no encoding for GPT-6 or Claude 5 models, so `TokenCounter` falls back to `cl100k_base`. Good enough for cost estimates, not exact.
- **OpenAI streaming reports 0 tokens.** The stream object has no `usage` unless the request sends `stream_options={"include_usage": True}`, and that final usage chunk has empty `choices`. Known, unfixed as of 2026-11-04.
- **Pace small tasks fast.** Harper sizes process to the diff: for a change under ~200 lines, write the failing test within a few tool calls and research the rest while implementing.
