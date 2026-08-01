# Advanced GenAI Customer Service Bot

A single, modular, production-quality customer service bot, built incrementally
across independent internship milestones on top of one shared codebase.

**Status: Milestone 1 (Sentiment Analysis) complete. LLM Provider Abstraction
(architectural enhancement, not a milestone) complete.**

Each milestone (Sentiment Analysis, Medical RAG, Dynamic Knowledge Base,
Research Paper Assistant, Multimodal AI, Multilingual Support, Conversation
Memory) is developed, tested, committed, and documented independently, only
once its own PRD is provided.

## LLM Provider Abstraction

This is a foundational/architectural improvement (not an internship
milestone): the app can now switch between LLM providers through a single
`.env` setting, with no code changes anywhere.

### Supported providers

| Provider | Status | Default model |
|---|---|---|
| **Groq** | ✅ Fully working | `llama-3.1-8b-instant` |
| **OpenAI** | ✅ Fully working | `gpt-4o-mini` |
| **Gemini** | 🔶 Future-ready placeholder (see note below) | `gemini-1.5-flash` |

> **Gemini note:** the provider class is fully wired into the same interface
> as Groq/OpenAI and can be selected today, but `google-generativeai` isn't
> installed as a hard dependency yet since this path hasn't been verified
> against the live API. If actually invoked without the package installed,
> it raises a clear, friendly error rather than crashing. To activate for
> real: `pip install google-generativeai` and uncomment it in
> `requirements.txt`.

### Environment configuration

```bash
# Choose one: openai | groq | gemini
LLM_PROVIDER=groq

# Only the selected provider's key needs to be set
GROQ_API_KEY=
OPENAI_API_KEY=
GEMINI_API_KEY=

# Optional, applies to whichever provider is selected
LLM_MODEL=            # blank = use that provider's own default model
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=800
LLM_TIMEOUT_SECONDS=30
```

### How to switch providers

Change `LLM_PROVIDER` in `.env` (and make sure that provider's API key is
set), then restart the app. Nothing else needs to change — `app.py` and
every milestone module call `utils/llm_client.py`'s `get_chat_completion()`
exactly as before; none of them know which provider is active.

### Architecture

```
utils/
├── llm_client.py                # Unified client every module calls (unchanged public API)
└── providers/
    ├── base_provider.py           # Abstract LLMProvider interface + shared exceptions
    ├── openai_provider.py           # OpenAI implementation
    ├── groq_provider.py              # Groq implementation
    ├── gemini_provider.py             # Gemini implementation (placeholder)
    └── __init__.py                     # Provider registry (build_provider)
```

Adding a new provider later means adding one file in `utils/providers/` and
one branch in `build_provider()` — no other file needs to change.

### Error handling

Missing API key, invalid `LLM_PROVIDER` value, network failure, rate limits,
and timeouts are all caught and mapped to friendly messages; the app never
crashes on a bad or missing configuration. Startup configuration (selected
provider + whether it's valid) is logged automatically when the app starts.

## Milestone 1: Sentiment Analysis

### Feature overview

- **Real-time sentiment analysis** — every user message is classified as
  Positive, Negative, or Neutral (with a confidence score) before the
  assistant generates a reply.
- **Adaptive tone** — the assistant's reply tone adapts to the detected
  sentiment: friendly/encouraging (positive), supportive/empathetic
  (negative), professional/informative (neutral).
- **Dual backend** — primary: HuggingFace transformers
  (`cardiffnlp/twitter-roberta-base-sentiment-latest`); automatic fallback:
  a dependency-free rule-based lexicon scorer, used transparently if the
  transformer model can't be loaded (no network access to download it, or
  `transformers`/`torch` aren't installed). Every result records which
  backend actually produced it.
- **Conversation storage** — every analyzed message (content, sentiment
  label, confidence, timestamp) is saved to SQLite, added additively to the
  existing `messages` table via a schema migration.
- **Analytics dashboard** (sidebar → Analytics) — total conversations,
  positive/negative/neutral counts, a pie chart, and a bar chart.
- **About Module page** (sidebar → About Module) — documents this
  milestone's model, fallback behavior, and adaptive-tone rules in-app.
- **Sidebar**: Chat, Analytics, About Module (plus the existing "coming
  soon" placeholders for pending milestones).

### Installation

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# edit .env: set LLM_PROVIDER (openai | groq | gemini) and that
# provider's API key. SENTIMENT_MODEL_NAME / SENTIMENT_USE_TRANSFORMER
# are optional, sensible defaults are already set.
```

### Usage

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

- **Chat** — type a message; its sentiment + confidence appear under your
  message and in the right-hand panel, and the assistant's reply tone
  adapts accordingly.
- **Analytics** — view aggregate sentiment stats across all conversations.
- **About Module** — read how this milestone works.

### Screenshots

_Placeholder — add screenshots of the Chat page (with sentiment badges +
side panel) and the Analytics dashboard here before sharing externally._

### Future improvements

- Escalation flagging for strongly negative messages (route to a human queue)
- Per-session sentiment trend over time (line chart) on the Analytics page
- Configurable confidence threshold surfaced in the UI (currently env-only)
- Multi-label / mixed-sentiment detection for longer messages

## Project structure

```
advanced-genai-chatbot/
├── app.py                        # Streamlit entry point: navigation + chat + analytics + about
├── config.py                      # Centralized, env-driven configuration
├── requirements.txt
├── .env.example
├── components/                     # Shared, reusable Streamlit UI building blocks
│   ├── navigation.py                 # Page registry (single source of truth)
│   ├── layout.py                       # Page header + sidebar
│   ├── chat.py                          # Chat history/input rendering
│   ├── placeholder.py                    # "Coming soon" page for pending milestones
│   ├── sentiment_panel.py                 # Current sentiment + confidence side panel
│   ├── dashboard.py                        # Sentiment analytics (pie + bar chart)
│   └── about.py                             # Generic "About Module" page renderer
├── modules/
│   ├── sentiment/                  # Milestone 1: implemented
│   │   ├── analyzer.py                # SentimentAnalyzer (transformer + lexicon fallback)
│   │   ├── lexicon.py                  # Rule-based fallback scorer
│   │   └── response_style.py            # Sentiment -> tone instructions
│   ├── medical/                     # Not yet implemented
│   ├── knowledge_base/               # Not yet implemented
│   ├── research/                      # Not yet implemented
│   ├── multimodal/                     # Not yet implemented
│   └── multilingual/                    # Not yet implemented
├── utils/
│   ├── logger.py                    # Shared logging setup
│   ├── storage.py                     # SQLite data access (messages + sentiment columns)
│   ├── llm_client.py                    # Unified, provider-agnostic chat completion client
│   └── providers/                         # LLM provider abstraction
│       ├── base_provider.py                 # Abstract interface + shared exceptions
│       ├── openai_provider.py                 # OpenAI implementation
│       ├── groq_provider.py                     # Groq implementation
│       └── gemini_provider.py                     # Gemini implementation (placeholder)
├── data/, database/, uploads/, vector_store/, assets/, docs/
└── tests/
    ├── test_config.py
    ├── test_storage.py
    ├── test_llm_client.py
    ├── test_llm_providers.py           # LLM Provider Abstraction
    ├── test_navigation.py
    └── test_sentiment.py                 # Milestone 1
```

## Running tests

```bash
pytest tests/ -v
```

50 tests currently pass: config defaults, the storage layer (including
sentiment columns and summary queries), the LLM provider abstraction
(provider registry, per-provider error handling, configuration validation,
provider switching), the navigation registry, and the sentiment analyzer.

## Dependencies

`requirements.txt`: Streamlit, `python-dotenv`, `pytest` (foundation);
`openai` + `groq` (LLM Provider Abstraction — install whichever you'll
actually use, or both); `transformers` + `torch` (Milestone 1 sentiment
model); `pandas` + `plotly` (Milestone 1 analytics dashboard).
`google-generativeai` (Gemini) and remaining milestone-specific
dependencies (LangChain, sentence-transformers, FAISS, Pillow, OpenCV) stay
commented out until actually needed.

## Development workflow

1. Provide the PRD for one milestone.
2. Its logic is implemented inside its own `modules/<name>/` package,
   wired into `app.py`'s `render_page()` and marked `implemented=True` in
   `components/navigation.py`.
3. Its own tests are added under `tests/`.
4. Its own section is added to this README, and a daily report is written
   under `docs/`.
5. Committed to GitHub independently (e.g. `Milestone 2 - Medical RAG`),
   keeping git history clean and each milestone reviewable on its own.
6. No milestone's implementation happens without its PRD being supplied first.

## Known limitations

- Chat replies require a real API key for whichever `LLM_PROVIDER` is
  selected — without one, the UI shows a clear configuration notice instead
  of crashing (sentiment analysis still works either way).
- Gemini is a future-ready placeholder: selecting it without installing
  `google-generativeai` raises a clear error rather than working live (see
  the LLM Provider Abstraction section above).
- The sidebar's "no API key" warning message (in `components/layout.py`)
  still mentions `OPENAI_API_KEY` specifically rather than naming whichever
  provider is actually selected — left as-is since this task's scope was
  limited to `utils/`, `config.py`, `.env.example`, `README.md`, and
  `requirements.txt`; worth a small follow-up fix in `components/`.
- The transformer sentiment backend needs network access to download model
  weights on first run; in restricted-network environments it transparently
  uses the rule-based lexicon fallback (verified: this dev sandbox has no
  HuggingFace Hub access, and the fallback path was exercised directly).
- No authentication/authorization yet (fine for internal/demo use; add
  before any public deployment).
- No rate limiting on the chat endpoint yet.
