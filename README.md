# Advanced GenAI Customer Service Bot

A single, modular, production-quality customer service bot, built incrementally
across independent internship milestones on top of one shared codebase.

**Status: Milestone 1 (Sentiment Analysis) complete. Milestone 2 (Medical
Knowledge Assistant / RAG) complete. LLM Provider Abstraction (architectural
enhancement, not a milestone) complete.**

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

## Milestone 2: Medical Knowledge Assistant (RAG)

### Feature overview

- **Retrieval-Augmented Generation** — every medical question is answered
  using retrieved passages from the MedQuAD medical knowledge base, never
  from the LLM's own unguided knowledge. The system prompt explicitly
  instructs the model to answer only from retrieved context and to say so
  honestly when the context is insufficient.
- **Automatic dataset loading** — the MedQuAD dataset downloads
  automatically on first run (from the official GitHub repository, a
  single-request zip download) and is cached locally; no manual dataset
  setup required. Corrupted individual files are skipped with a logged
  warning rather than aborting the whole load.
- **Reusable processing pipeline** — text cleaning + chunking (via
  LangChain's `RecursiveCharacterTextSplitter`) with metadata (question,
  topic, source, URL) preserved on every chunk.
- **Embeddings with automatic fallback** — primary: Sentence Transformers
  (`sentence-transformers/all-MiniLM-L6-v2`); automatic fallback: a local
  TF-IDF vectorizer, used transparently if the transformer model can't be
  downloaded (mirrors Milestone 1's sentiment analyzer resilience pattern,
  and satisfies the PRD's "embedding failure" error-handling requirement).
- **FAISS vector search** — builds automatically on first run, loads from
  disk on subsequent runs, and **skips regeneration entirely** if the
  underlying dataset hasn't changed (detected via a content fingerprint).
- **Uses the existing configurable LLM provider** — no direct Groq/OpenAI
  SDK calls; the same `utils/llm_client.py` from the LLM Provider
  Abstraction is reused, so this milestone works with whichever provider
  is selected in `.env`.
- **Source attribution** — every answer displays which knowledge-base
  entries (source dataset, topic, relevance score) it was grounded in.
- **Conversation storage** — every medical Q&A exchange (question, answer,
  sources, timestamp) is saved to a new, separate SQLite table
  (`medical_queries`), reusing the existing database layer without
  touching the base `messages` table.
- **Educational disclaimer** — shown on every page load and after every
  answer: this assistant does not provide professional medical advice.
- **Sidebar**: Chat, Analytics, Medical Knowledge Assistant, About Module
  (plus the existing "coming soon" placeholders for pending milestones).

### Installation

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# edit .env: set LLM_PROVIDER + that provider's API key (as before).
# No medical-specific setup is required -- see "Dataset setup" below.
```

### Dataset setup

No manual setup needed. On first visit to the Medical Knowledge Assistant
page, the app automatically:
1. Downloads the [MedQuAD dataset](https://github.com/abachaa/MedQuAD)
   (~16 MB zip, single request) to `data/medical/medquad_raw/`.
2. Parses and caches it to `data/medical/medquad_processed.json`.
3. Builds a FAISS index in `vector_store/medical/`.

All of this is skipped on subsequent runs (cached), and none of these
generated files are committed to git (see `.gitignore`).

**Configuration (optional, all in `.env`):**
```bash
MEDICAL_DATASET_DIR=                  # default: data/medical/medquad_raw
MEDICAL_MAX_SOURCE_FILES=400          # 0 = process the full ~47k-pair dataset
MEDICAL_CHUNK_SIZE=800
MEDICAL_CHUNK_OVERLAP=120
MEDICAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
MEDICAL_TOP_K=4
MEDICAL_MIN_SIMILARITY=0.05
```
`MEDICAL_MAX_SOURCE_FILES` is a dev-speed knob, not a hard limitation:
raise it (or set to `0`) to index the complete dataset -- the default of
400 files (sampled round-robin across every source category, ~1,700+ QA
pairs) keeps first-run indexing under a minute.

### How RAG works

```
Question
  -> embed (Sentence Transformers, or TF-IDF if that's unavailable)
  -> search FAISS index for the most similar chunks
  -> take the top-K (default 4) above a minimum similarity
  -> build a prompt that includes only those retrieved passages
  -> send to the configured LLM provider (Groq/OpenAI/Gemini)
  -> display the answer + which sources it came from
```

If no chunk meets the similarity threshold, the LLM is told explicitly
that no relevant context was found, so it can say so rather than guessing.

### Running the application

```bash
streamlit run app.py
```

Open the sidebar → **🏥 Medical Knowledge Assistant**, wait for the
knowledge base to load (first run only), then ask a question like *"What
are the symptoms of diabetes?"*. The answer, its sources, and the
educational disclaimer all display on the page.

### Future improvements

- Let users adjust Top-K / similarity threshold from the UI
- Show a confidence indicator based on retrieval score, not just LLM output
- Support incremental dataset updates without a full rebuild
- Surface which embedding backend is active in the sidebar (not just the page)

## Project structure

```
advanced-genai-chatbot/
├── app.py                        # Streamlit entry point: navigation + chat + analytics + about + medical
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
│   ├── medical/                     # Milestone 2: implemented
│   │   ├── config.py                   # Self-contained, env-driven config for this milestone
│   │   ├── loader.py                    # MedQuAD download + XML parsing + caching
│   │   ├── preprocess.py                 # Text cleaning + chunking (LangChain)
│   │   ├── embeddings.py                  # Sentence Transformers + TF-IDF fallback
│   │   ├── vector_store.py                 # FAISS index build/save/load
│   │   ├── retriever.py                      # Top-K similarity search
│   │   ├── prompts.py                          # Grounded-answer prompt templates
│   │   ├── rag_pipeline.py                       # End-to-end orchestration
│   │   └── medical_chat.py                        # Streamlit page
│   ├── knowledge_base/               # Not yet implemented
│   ├── research/                      # Not yet implemented
│   ├── multimodal/                     # Not yet implemented
│   └── multilingual/                    # Not yet implemented
├── utils/
│   ├── logger.py                    # Shared logging setup
│   ├── storage.py                     # SQLite data access (messages, sentiment, medical_queries)
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
    ├── test_sentiment.py                 # Milestone 1
    └── test_medical.py                    # Milestone 2
```

## Running tests

```bash
pytest tests/ -v
```

79 tests currently pass: config defaults, the storage layer (sentiment
columns, medical_queries table, summary queries), the LLM provider
abstraction, the navigation registry, the sentiment analyzer, and the
medical RAG pipeline (dataset parsing, chunking, embeddings with fallback,
vector store build/cache/search, retrieval filtering, prompt construction,
and end-to-end answer generation with mocked LLM calls). Medical tests use
small synthetic data -- none require downloading the full dataset or
network access.

## Dependencies

`requirements.txt`: Streamlit, `python-dotenv`, `pytest` (foundation);
`openai` + `groq` (LLM Provider Abstraction); `transformers` + `torch`
(Milestone 1 sentiment model); `pandas` + `plotly` (Milestone 1 analytics
dashboard); `sentence-transformers` + `faiss-cpu` + `langchain` +
`langchain-text-splitters` + `scikit-learn` (Milestone 2 RAG pipeline;
scikit-learn powers the TF-IDF fallback embedder).
`google-generativeai` (Gemini) and remaining milestone-specific
dependencies (Pillow, OpenCV) stay commented out until actually needed.

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
- The medical embedding backend needs network access to download the
  Sentence Transformers model on first run; in restricted-network
  environments it transparently uses a local TF-IDF fallback (verified:
  this dev sandbox has no HuggingFace Hub access, and the fallback path
  was exercised directly -- retrieval quality is somewhat lower than with
  real sentence embeddings, but stays functional and grounded).
- By default, only 400 source files (~1,700+ QA pairs, sampled across every
  source category) are indexed for fast first-run startup; set
  `MEDICAL_MAX_SOURCE_FILES=0` to index the complete ~47,000-pair dataset
  (slower first run, better coverage).
- The medical assistant does not yet flag when a question falls far
  outside the knowledge base's coverage beyond the standard "no relevant
  passages found" notice.
