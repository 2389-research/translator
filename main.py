#!/usr/bin/env python3
# ABOUTME: Command-line interface for translating text files to different languages.
# ABOUTME: Uses OpenAI's API to perform translations while maintaining formatting.

from translator.cli import TranslatorCLI


def main() -> None:
    """Main entry point for the translator CLI."""
    TranslatorCLI.run()


if __name__ == "__main__":
    main()