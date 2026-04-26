"""
Financial Agent
===============
Agente financiero multi-turn con memoria de conversación.
Provider-agnostic: funciona con Gemini o Claude sin cambiar el agente.

Uso:
  # Con Gemini (gratis)
  agent = FinancialAgent(provider="gemini")

  # Con Claude
  agent = FinancialAgent(provider="claude")

  # Comparar ambos con la misma pregunta
  python src/agents/financial_agent.py --compare
"""
import os
import logging
from typing import List, Literal
from datetime import datetime
from dotenv import load_dotenv

from src.providers.base import Message, AgentResponse
from src.providers.gemini_provider import GeminiProvider
from src.providers.claude_provider import ClaudeProvider

load_dotenv()
logger = logging.getLogger("financial-agent")


class FinancialAgent:

    def __init__(self, provider: Literal["gemini", "claude"] = "gemini"):
        self.provider_name = provider
        self.history: List[Message] = []
        self.session_start = datetime.now()

        if provider == "gemini":
            self.provider = GeminiProvider(
                model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
            )
        elif provider == "claude":
            self.provider = ClaudeProvider(
                model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
            )
        else:
            raise ValueError(f"Provider desconocido: {provider}. Usa 'gemini' o 'claude'")

        logger.info(f"FinancialAgent iniciado — provider: {provider}")

    def chat(self, user_message: str) -> AgentResponse:
        """Envía mensaje y mantiene historial de conversación."""
        self.history.append(Message(role="user", content=user_message))
        response = self.provider.chat(self.history)
        self.history.append(Message(role="assistant", content=response.content))
        return response

    def stream(self, user_message: str):
        """Versión streaming — imprime en tiempo real."""
        self.history.append(Message(role="user", content=user_message))
        full_response = ""
        print(f"\n[{self.provider_name.upper()}] ", end="", flush=True)
        for chunk in self.provider.stream_chat(self.history):
            print(chunk, end="", flush=True)
            full_response += chunk
        print()  # newline final
        self.history.append(Message(role="assistant", content=full_response))

    def reset(self):
        """Limpia historial — nueva sesión."""
        self.history = []
        self.session_start = datetime.now()

    def get_history_summary(self) -> dict:
        return {
            "provider": self.provider_name,
            "session_start": self.session_start.isoformat(),
            "turns": len([m for m in self.history if m.role == "user"]),
            "total_messages": len(self.history),
        }


def compare_providers(question: str):
    """
    Compara respuestas de Gemini vs Claude para la misma pregunta.
    Diferenciador de portafolio: demuestra provider-agnostic architecture.
    """
    print(f"\n{'='*70}")
    print(f"PREGUNTA: {question}")
    print(f"{'='*70}")

    results = {}
    for provider_name in ["gemini", "claude"]:
        try:
            agent = FinancialAgent(provider=provider_name)
            response = agent.chat(question)
            results[provider_name] = {
                "response": response.content,
                "model": response.model,
                "tokens": response.tokens_used,
            }
            print(f"\n── {provider_name.upper()} ({response.model}) ──")
            print(response.content[:500] + "..." if len(response.content) > 500 else response.content)
            if response.tokens_used:
                print(f"   [tokens: {response.tokens_used}]")
        except Exception as e:
            print(f"\n── {provider_name.upper()} ERROR: {e}")
            results[provider_name] = {"error": str(e)}

    return results


# ── Demo interactiva ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if "--compare" in sys.argv:
        compare_providers(
            "Una empresa tiene revenue de $8.5M, margen neto 14%, "
            "current ratio 2.5 y D/E de 0.5. ¿Cuál es tu evaluación del perfil financiero?"
        )
        sys.exit(0)

    # Chat interactivo con Gemini
    print("\n🤖 Gemini Financial Agent — Chat Interactivo")
    print("   Escribe 'exit' para salir | 'reset' para nueva sesión")
    print("   Modelo: gemini-2.0-flash (GRATIS)\n")

    agent = FinancialAgent(provider="gemini")

    # Contexto inicial de demo
    demo_questions = [
        "Analiza estos datos: revenue $4.2M (+12% YoY), EBITDA margin 22.7%, churn 2.1%/mes, CAC $1,250, LTV $18,400. ¿Cuál es el estado de la empresa?",
        "¿Cuál es el LTV/CAC ratio y qué significa para el negocio?",
        "Si el churn sube a 4%, ¿cómo impacta el LTV?",
    ]

    print("Demo con preguntas predefinidas (o escribe la tuya):\n")
    for i, q in enumerate(demo_questions, 1):
        print(f"[Demo {i}] {q[:80]}...")

    print("\n" + "-"*50)

    while True:
        try:
            user_input = input("\nTú: ").strip()
            if not user_input:
                continue
            if user_input.lower() == "exit":
                print(f"\nSesión finalizada. {agent.get_history_summary()}")
                break
            if user_input.lower() == "reset":
                agent.reset()
                print("✅ Sesión reiniciada.")
                continue

            agent.stream(user_input)

        except KeyboardInterrupt:
            print("\n\nInterrumpido.")
            break
        except Exception as e:
            print(f"Error: {e}")
