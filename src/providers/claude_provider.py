"""
Claude Provider
===============
Claude API — misma interfaz que GeminiProvider.
Permite comparar outputs Claude vs Gemini con el mismo agente.
"""
import os
import logging
from typing import List, Generator

import anthropic

from src.providers.base import BaseProvider, Message, AgentResponse

logger = logging.getLogger("claude-provider")

SYSTEM_PROMPT = """
You are a senior financial analyst AI assistant with expertise in
financial statement analysis, KPI interpretation, risk assessment,
revenue forecasting, and FP&A automation.

Always be precise with numbers, flag risks clearly, and provide
actionable recommendations. Respond in the user's language.
"""


class ClaudeProvider(BaseProvider):

    def __init__(self, model: str = "claude-sonnet-4-6"):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY no configurada en .env")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model
        logger.info(f"ClaudeProvider inicializado — modelo: {model}")

    def chat(self, messages: List[Message], tools: List[dict] = None) -> AgentResponse:
        anthropic_messages = [
            {"role": m.role, "content": m.content} for m in messages
        ]
        kwargs = {
            "model": self.model_name,
            "max_tokens": 2048,
            "system": SYSTEM_PROMPT,
            "messages": anthropic_messages,
        }
        if tools:
            kwargs["tools"] = tools

        response = self.client.messages.create(**kwargs)
        text = next((b.text for b in response.content if hasattr(b, "text")), "")
        tool_calls = [
            {"name": b.name, "args": b.input}
            for b in response.content
            if b.type == "tool_use"
        ] or None

        return AgentResponse(
            content=text,
            provider="claude",
            model=self.model_name,
            tool_calls=tool_calls,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
        )

    def stream_chat(self, messages: List[Message]) -> Generator[str, None, None]:
        anthropic_messages = [
            {"role": m.role, "content": m.content} for m in messages
        ]
        with self.client.messages.stream(
            model=self.model_name,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=anthropic_messages,
        ) as stream:
            for text in stream.text_stream:
                yield text
