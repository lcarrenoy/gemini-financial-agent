"""
Base provider — interfaz común para Gemini y Claude.
El agente no sabe con qué LLM está hablando.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Message:
    role: str   # "user" | "assistant"
    content: str


@dataclass
class AgentResponse:
    content: str
    provider: str
    model: str
    tool_calls: List[dict] = None
    tokens_used: Optional[int] = None


class BaseProvider(ABC):

    @abstractmethod
    def chat(self, messages: List[Message], tools: List[dict] = None) -> AgentResponse:
        """Envía mensajes y retorna respuesta del LLM."""
        pass

    @abstractmethod
    def stream_chat(self, messages: List[Message]):
        """Versión streaming — yield chunks de texto."""
        pass
