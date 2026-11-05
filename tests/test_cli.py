#!/usr/bin/env python3
# ABOUTME: Tests for command-line model resolution.
# ABOUTME: Verifies the -m flag, DEFAULT_MODEL, and built-in default precedence.

from translator.cli import TranslatorCLI
from translator.config import ModelConfig


def test_resolve_model_prefers_cli_flag(monkeypatch):
    monkeypatch.setenv("DEFAULT_MODEL", "claude-haiku-5-5")
    assert TranslatorCLI.resolve_model("gpt-6-astra") == "gpt-6-astra"


def test_resolve_model_falls_back_to_env_then_builtin(monkeypatch):
    monkeypatch.setenv("DEFAULT_MODEL", "claude-haiku-5-5")
    assert TranslatorCLI.resolve_model(None) == "claude-haiku-5-5"

    monkeypatch.delenv("DEFAULT_MODEL")
    assert TranslatorCLI.resolve_model(None) == ModelConfig.DEFAULT_MODEL
