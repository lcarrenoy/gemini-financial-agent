# Gemini Financial Agent

> Financial analysis agent built directly on **Gemini API** (google-generativeai SDK) with **provider-agnostic architecture** — swap Gemini for Claude with one env variable.

**Stack:** Python · Gemini API · Claude API · Streamlit · google-generativeai

**Key differentiator:** Same agent, any LLM provider. Demonstrates multi-provider architecture — no vendor lock-in.

---

## Architecture

```
FinancialAgent
    │
    ├── provider="gemini"  →  GeminiProvider  →  gemini-2.0-flash (FREE)
    └── provider="claude"  →  ClaudeProvider  →  claude-sonnet-4-6
```

Both providers implement the same `BaseProvider` interface — the agent doesn't know which LLM it's using.

## Gemini API features demonstrated

| Feature | Implementation |
|---------|---------------|
| Direct SDK | `google-generativeai` — no LangChain |
| Function calling | Native Gemini tool declarations |
| System instructions | Persistent across the session |
| Safety settings | Configurable per use case |
| Streaming | `stream=True` — real-time output |
| Search grounding | `google_search_retrieval` tool |
| Token counting | `model.count_tokens()` |
| Multi-turn memory | Managed conversation history |

## How to Run

```bash
# 1. Get your FREE Gemini API key
# → https://aistudio.google.com/app/apikey

# 2. Install
uv sync

# 3. Configure
cp .env.example .env
# Add your GEMINI_API_KEY

# 4. Chat interactivo
uv run python src/agents/financial_agent.py

# 5. Comparar Gemini vs Claude
uv run python src/agents/financial_agent.py --compare
```

## Structure

```
gemini-financial-agent/
├── src/
│   ├── providers/
│   │   ├── base.py              # Interfaz común (Message, AgentResponse)
│   │   ├── gemini_provider.py   # Gemini API directo + function calling
│   │   └── claude_provider.py  # Claude API — misma interfaz
│   └── agents/
│       └── financial_agent.py  # Agente multi-turn + compare_providers()
├── .env.example
└── pyproject.toml
```

## Roadmap

- [ ] Function calling con tools financieras (reutilizar mcp-financial-tools)
- [ ] Streamlit dashboard para comparar Gemini vs Claude en tiempo real
- [ ] LangFuse para tracing multi-provider
- [ ] Gemini Files API para analizar PDFs de estados financieros

---

*Part of [Luis Carreño's Portfolio](https://github.com/lcarrenoy) · AI Engineer · Gemini API · Multi-provider Architecture*
