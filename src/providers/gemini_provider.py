"""
Gemini Provider
===============
Gemini API directo — sin LangChain, sin wrappers.
Demuestra:
  - google-generativeai SDK
  - Function calling nativo de Gemini
  - System instructions
  - Safety settings
  - Streaming
  - Grounding con Google Search

Modelos gratuitos (abril 2026):
  - gemini-1.5-flash  → 15 req/min, 1M tokens/día
  - gemini-2.0-flash  → 15 req/min
"""
import os
import json
import logging
from typing import List, Generator

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from src.providers.base import BaseProvider, Message, AgentResponse

logger = logging.getLogger("gemini-provider")

# ── Safety settings — permisivos para análisis financiero ────────────────────
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

SYSTEM_INSTRUCTION = """
You are a senior financial analyst AI assistant with expertise in:
- Financial statement analysis (P&L, Balance Sheet, Cash Flow)
- KPI interpretation and benchmarking
- Risk assessment and early warning signals
- Revenue forecasting and variance analysis
- FP&A automation

Always:
- Be precise with numbers — state units (USD, %, x) explicitly
- Flag risks clearly with severity (HIGH/MEDIUM/LOW)
- Provide actionable recommendations, not just observations
- Cite the specific data point that drives each conclusion

Respond in the same language as the user's question.
"""


class GeminiProvider(BaseProvider):

    def __init__(self, model: str = "gemini-2.0-flash"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY no configurada en .env")
        genai.configure(api_key=api_key)
        self.model_name = model
        self.model = genai.GenerativeModel(
            model_name=model,
            system_instruction=SYSTEM_INSTRUCTION,
            safety_settings=SAFETY_SETTINGS,
        )
        logger.info(f"GeminiProvider inicializado — modelo: {model}")

    def _convert_messages(self, messages: List[Message]) -> List[dict]:
        """Convierte formato interno al formato de Gemini."""
        converted = []
        for m in messages:
            role = "model" if m.role == "assistant" else "user"
            converted.append({"role": role, "parts": [m.content]})
        return converted

    def chat(self, messages: List[Message], tools: List[dict] = None) -> AgentResponse:
        """Chat con Gemini — soporta function calling si se pasan tools."""
        gemini_messages = self._convert_messages(messages)

        # Si hay tools, usar function calling nativo de Gemini
        if tools:
            model_with_tools = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=SYSTEM_INSTRUCTION,
                safety_settings=SAFETY_SETTINGS,
                tools=tools,
            )
            chat_session = model_with_tools.start_chat(history=gemini_messages[:-1])
            response = chat_session.send_message(gemini_messages[-1]["parts"][0])
        else:
            chat_session = self.model.start_chat(history=gemini_messages[:-1])
            response = chat_session.send_message(gemini_messages[-1]["parts"][0])

        # Procesar function calls si las hay
        tool_calls = []
        for part in response.parts:
            if hasattr(part, "function_call") and part.function_call:
                tool_calls.append({
                    "name": part.function_call.name,
                    "args": dict(part.function_call.args),
                })

        text = response.text if hasattr(response, "text") else ""

        return AgentResponse(
            content=text,
            provider="gemini",
            model=self.model_name,
            tool_calls=tool_calls if tool_calls else None,
            tokens_used=response.usage_metadata.total_token_count
            if hasattr(response, "usage_metadata") else None,
        )

    def stream_chat(self, messages: List[Message]) -> Generator[str, None, None]:
        """Streaming — yield texto chunk a chunk."""
        gemini_messages = self._convert_messages(messages)
        chat_session = self.model.start_chat(history=gemini_messages[:-1])
        response = chat_session.send_message(
            gemini_messages[-1]["parts"][0],
            stream=True,
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text

    def count_tokens(self, text: str) -> int:
        """Cuenta tokens antes de enviar — útil para cost estimation."""
        result = self.model.count_tokens(text)
        return result.total_tokens

    def generate_with_search_grounding(self, query: str) -> AgentResponse:
        """
        Gemini con Google Search grounding — respuestas basadas en datos reales.
        Útil para: precios de mercado, noticias financieras, tipos de cambio.
        Requiere: gemini-1.5-pro o gemini-2.0-flash con grounding habilitado.
        """
        model_with_search = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=SYSTEM_INSTRUCTION,
            tools="google_search_retrieval",  # Grounding nativo de Gemini
        )
        response = model_with_search.generate_content(query)
        return AgentResponse(
            content=response.text,
            provider="gemini-grounded",
            model="gemini-2.0-flash",
        )
