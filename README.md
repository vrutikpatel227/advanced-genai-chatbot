# Advanced GenAI Customer Service Bot

A single, modular, production-quality customer service bot, built incrementally
across independent internship milestones on top of one shared codebase.

**Status: Project foundation complete. No feature milestones implemented yet.**

Each milestone (Sentiment Analysis, Medical RAG, Dynamic Knowledge Base,
Research Paper Assistant, Multimodal AI, Multilingual Support, Conversation
Memory, Analytics Dashboard) will be developed, tested, committed, and
documented independently, only once its own PRD is provided. This README
describes the foundation only.

## What's in the foundation

- **Streamlit application shell** — `app.py`, with navigation-driven page
  routing and a base chat interface (plain LLM Q&A, no milestone-specific
  intelligence layered in yet).
- **Configuration management** — `config.py`, fully env-driven via `.env`,
  no hardcoded values.
- **Logging system** — `utils/logger.py`, console + rotating file handler,
  shared by every module.
- **SQLite database layer** — `utils/storage.py`, a minimal, milestone-agnostic
  `messages` table (session_id, role, content, timestamp). Milestone-specific
  columns (e.g. a sentiment score) are added later, additively, only when
  that milestone is implemented — not baked in ahead of time.
- **Shared UI components** — `components/`, reusable Streamlit building
  blocks (page header, sidebar, chat history/input, placeholder page) so
  every milestone's UI stays consistent instead of each writing raw
  Streamlit calls.
- **Navigation system** — `components/navigation.py` is the single source of
  truth for which pages exist and whether they're implemented yet; the
  sidebar and `app.py` both read from it.
- **Common utilities** — `utils/llm_client.py`, an OpenAI-compatible chat
  completion wrapper with explicit error handling, reused by the base chat
  page and by every future milestone.
- **Placeholder modules** — `modules/{sentiment,medical,knowledge_base,
  research,multimodal,multilingual}/`, empty, importable packages ready for
  each milestone's real implementation.

## Project structure

```
advanced-genai-chatbot/
├── app.py                     # Streamlit entry point: navigation + base chat page
├── config.py                   # Centralized, env-driven configuration
├── requirements.txt
├── .env.example
├── components/                  # Shared, reusable Streamlit UI building blocks
│   ├── navigation.py              # Page registry (single source of truth)
│   ├── layout.py                    # Page header + sidebar
│   ├── chat.py                       # Chat history/input rendering
│   └── placeholder.py                 # "Coming soon" page for pending milestones
├── modules/                      # One empty placeholder package per milestone
│   ├── sentiment/
│   ├── medical/
│   ├── knowledge_base/
│   ├── research/
│   ├── multimodal/
│   └── multilingual/
├── utils/
│   ├── logger.py                  # Shared logging setup
│   ├── storage.py                   # SQLite data access (messages only)
│   └── llm_client.py                  # OpenAI-compatible chat completion wrapper
├── data/, database/, uploads/, vector_store/, assets/, docs/
└── tests/
    ├── test_config.py
    ├── test_storage.py
    ├── test_llm_client.py
    └── test_navigation.py
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and set OPENAI_API_KEY (and OPENAI_BASE_URL if using a
# different OpenAI-compatible provider)
```

## Running the app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501). You'll see
the chat page (fully functional, given an API key) and a "coming soon"
placeholder for every pending milestone in the sidebar.

## Running tests

```bash
pytest tests/ -v
```

15 tests currently cover config defaults, the storage layer, LLM client
error handling, and the navigation registry.

## Dependencies

`requirements.txt` currently installs only what the foundation uses:
Streamlit (UI), `python-dotenv` (config), `openai` (chat), `pytest` (tests).
Milestone-specific dependencies (LangChain, transformers/torch,
sentence-transformers, FAISS, Pandas, Plotly, Pillow, OpenCV) are listed
in the file, commented out, to be uncommented as each milestone is
implemented — this keeps the installed footprint always matching what's
actually built, per the "no premature implementation" workflow below.

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

- The base chat page requires a real `OPENAI_API_KEY` — without one, the UI
  shows a clear configuration notice instead of crashing.
- No authentication/authorization yet (fine for internal/demo use; add
  before any public deployment).
- No rate limiting on the chat endpoint yet.
