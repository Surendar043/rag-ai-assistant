"""LLM interface wrapper.

Supports swapping between different providers (OpenAI, local, mock) by
implementing the same `complete` interface. Falls back to a deterministic
mock when no API key/provider is configured so the project is runnable offline.
"""

import os
from typing import Optional


class LLM:
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature
        self.client = self._init_client()

    def _init_client(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                from openai import OpenAI
                return OpenAI(api_key=api_key)
            except ImportError:
                pass
        return None

    def complete(self, prompt: str) -> str:
        """Generate a completion for the given prompt."""
        if self.client is not None:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
                return response.choices[0].message.content.strip()
            except Exception:
                pass
        # Deterministic offline fallback: echo the most relevant context sentence
        return self._mock_complete(prompt)

    def _mock_complete(self, prompt: str) -> str:
        """Offline mock: extract the first context block as the answer."""
        lines = prompt.split("\n\n")
        context_blocks = [l for l in lines if l.startswith("[Source")]
        if context_blocks:
            # Return the first context block's text as a simple extractive answer
            first = context_blocks[0]
            text = first.split("]", 1)[1].strip() if "]" in first else first
            return f"Based on the provided context: {text}"
        return "I don't have enough information to answer that."
