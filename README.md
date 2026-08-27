# Advanced GenAI Customer Service Bot

A single, modular, production-quality customer service bot, built incrementally
across independent internship milestones on top of one shared codebase.

**Status: Milestone 1 (Sentiment Analysis) complete. Milestone 2 (Medical
Knowledge Assistant / RAG) complete. Milestone 3 (Dynamic Knowledge Base)
complete. Milestone 4 (Research Assistant) complete. Milestone 5
(Multimodal AI) complete. LLM Provider Abstraction (architectural
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
| **Gemini** | ✅ Fully working | `gemini-3.6-flash
` |

> **Gemini note:** uses Google's current `google-genai` SDK (the actively
> maintained successor to the older, now-legacy `google-generativeai`
> package — this project never uses that legacy package). Supports both
> text and vision requests. See "Bug Fix: Gemini Provider" below for the
> history of why an earlier version of this provider didn't work.

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
    ├── gemini_provider.py             # Gemini implementation (production, text + vision)
    └── __init__.py                     # Provider registry (build_provider)
```

Adding a new provider later means adding one file in `utils/providers/` and
one branch in `build_provider()` — no other file needs to change.

### Error handling

Missing API key, invalid `LLM_PROVIDER` value, network failure, rate limits,
and timeouts are all caught and mapped to friendly messages; the app never
crashes on a bad or missing configuration. Startup configuration (selected
provider + whether it's valid) is logged automatically when the app starts.

### Vision support (added in Milestone 5)

The provider interface also supports image input: every `LLMProvider` has
`supports_vision(model)` (capability check) and `generate_with_image()`
(the actual vision call), both with safe defaults (`False` / a friendly
`VisionNotSupportedError`) so providers that don't implement vision never
crash the app — they just report "not supported." OpenAI (`gpt-4o`/`gpt-4o-mini`
family), Groq (Llama-Vision models), and Gemini (`1.5`+/`2`+ family) all
have working implementations. See the Milestone 5 section below for how
this is used.

### Bug Fix: Gemini Provider (post-Milestone 5)

**Root cause**: the original Gemini provider was a placeholder that
imported `google.generativeai` — Google's older, now-legacy Gemini SDK —
which was never added as an installed dependency (only left commented out
in `requirements.txt`). Every call hit the `ImportError` branch and raised
the "future-ready placeholder" message, regardless of configuration.
Installing that legacy package wouldn't have been the right fix either:
Google has since released `google-genai`, a newer, actively maintained,
unified SDK that supersedes it.

**Fix**: replaced the placeholder with a full production implementation
using `google-genai` (`genai.Client(...).models.generate_content(...)`),
supporting both text and vision requests through the same interface every
other provider already uses — no new methods, no interface changes.
Includes real error mapping (auth, rate limit, timeout, connection errors,
each translated from the SDK's actual exception types) and retry handling
via the SDK's built-in `HttpRetryOptions`. `requirements.txt` now lists
`google-genai` as an active dependency (not `google-generativeai`, which
this project does not use).

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

## Milestone 3: Dynamic Knowledge Base

### Feature overview

- **Upload your own documents** — PDF, TXT, and Markdown (`.md`) are
  supported today; the format registry in `parser.py` is designed so a
  new format is one function + one registry entry, not a rewrite.
- **Incremental indexing** — new documents are embedded and added
  directly to the existing FAISS index without rebuilding anything else.
  A separate, explicit **Rebuild Index** button exists for a full
  from-scratch rebuild when you actually want one (e.g. after changing
  chunk size).
- **Immediate searchability** — every newly indexed document becomes
  searchable right away; try it in the page's own search box.
- **Duplicate detection** — every upload is hashed (SHA-256); re-uploading
  the same content (even under a different filename) is detected and
  skipped with a clear message, never silently re-indexed.
- **Reuses Milestone 2's RAG infrastructure directly** — `clean_text()`
  and `SentenceTransformerEmbedder` are imported from `modules/medical/`,
  not reimplemented. The only genuinely new embedding code is a stateless
  hashing-vectorizer fallback (`HashingEmbedder`), used instead of
  Milestone 2's TF-IDF fallback because TF-IDF must be fit on a fixed
  corpus up front — incompatible with "new documents show up
  incrementally," which the hashing approach handles natively.
- **Metadata tracking** — every document's filename, type, content hash,
  chunk count, and status live in a new, separate `knowledge_documents`
  SQLite table (additive; doesn't touch `messages` or `medical_queries`).
- **Update Index vs. Rebuild Index** — "Update" only reprocesses documents
  left in a non-indexed state (e.g. from an earlier failure); "Rebuild"
  regenerates the entire index from all stored document text.
- **Sidebar**: Chat, Analytics, About Module, Medical Knowledge Assistant,
  Dynamic Knowledge Base (plus remaining "coming soon" placeholders).

### Supported file formats

| Format | Extension | Notes |
|---|---|---|
| PDF | `.pdf` | Text extracted per page via `pypdf`; unreadable individual pages are skipped with a warning rather than failing the whole file |
| Plain text | `.txt` | Decoded as UTF-8, falling back to Latin-1 if needed |
| Markdown | `.md` | Extracted as plain text; markdown syntax is preserved (not stripped) since it carries useful structure |

Adding a new format later: write one `_extract_<format>()` function in
`modules/knowledge_base/parser.py`, register it in `_EXTRACTORS`, and add
its extension to `SUPPORTED_EXTENSIONS` in `config.py` — nothing else
needs to change.

### Upload workflow

```
Upload file
  -> validate (format, size, non-empty)
  -> hash content (SHA-256) -> check for duplicates -> stop here if found
  -> extract text (per-format extractor)
  -> clean + chunk (LangChain, same approach as Milestone 2)
  -> embed new chunks only (Sentence Transformers, or hashing fallback)
  -> add to the FAISS index incrementally (existing vectors untouched)
  -> save extracted text to disk (enables future "Rebuild Index")
  -> record document metadata in SQLite
```

### Incremental indexing

Unlike Milestone 2's medical knowledge base (which rebuilds-or-skips
based on whether its *entire* source dataset changed), the Knowledge
Base is designed for continuous growth: `KnowledgeVectorStore.add_chunks()`
embeds only the new document's chunks and appends them to the existing
FAISS `IndexIDMap`, so uploading document #50 doesn't touch the vectors
for documents #1–49 at all.

### Knowledge management

- **Knowledge Statistics**: total documents, total chunks, last update
  time, and vector database status, all reused directly from the new
  `knowledge_documents` SQLite table.
- **Document List**: filename, type, upload date, chunk count, and
  status for every indexed document.
- **Update Index**: reprocesses only documents not currently marked
  `"indexed"` — a no-op (with a clear message) when everything is
  already up to date.
- **Rebuild Index**: fully regenerates the index from every stored
  document's saved text — useful after a config change (e.g. chunk
  size) or if you want to switch from the hashing fallback to real
  Sentence Transformers embeddings once network access is available.

### Running the application

```bash
streamlit run app.py
```

Open the sidebar → **📚 Dynamic Knowledge Base**, upload a PDF/TXT/MD
file, watch it appear in the document list and statistics, then use the
search box to confirm it's immediately retrievable.

### Future improvements

- Per-document delete/re-index (currently additive-only; removing a
  document requires a full rebuild after removing its stored text)
- Chunk-level preview before confirming an upload
- Batch/multi-file upload in a single action
- Support for `.docx` and `.csv` (straightforward given the existing
  format-registry design)

## Milestone 4: Research Assistant

### Feature overview

- **Upload research papers (PDF)** and ask grounded, cited questions about
  them, or generate a structured six-section summary.
- **Maximally reuses Milestone 2 & 3's infrastructure** — per the PRD's
  explicit "avoid duplicate implementations" instruction: PDF text
  extraction, chunking, and the embedding pipeline are all imported
  directly from `modules/knowledge_base/`, not reimplemented. The vector
  store (`ResearchVectorStore`) **subclasses**
  `KnowledgeVectorStore` from Milestone 3, inheriting incremental add /
  save / load / search unchanged.
- **The one genuinely new capability**: per-paper deletion. Milestone 3's
  knowledge base is additive-only; the Research Assistant needed
  "deleting a paper removes its vectors without affecting other indexed
  papers," so `ResearchVectorStore` adds a single new method
  (`delete_document()`) using FAISS's native `remove_ids()` — verified
  directly that deleting one paper leaves every other paper's vectors
  and search results completely untouched.
- **Grounded question answering** — every answer is generated strictly
  from retrieved paper excerpts; the system prompt explicitly instructs
  the LLM to refuse rather than guess when the evidence is insufficient.
- **Citations, never fabricated** — every citation shown is built
  directly from an actual retrieved passage (filename, excerpt text,
  similarity score) — there's no path that invents a citation.
- **Structured paper summaries** — Executive Summary, Research Objective,
  Methodology, Key Findings, Conclusion, Future Work, generated from the
  paper's own indexed content (capped to a configurable context size for
  very long papers).
- **Paper management** — list, delete, and re-index papers, with live
  chunk counts and status.
- **Reuses the existing configurable LLM provider** — no direct
  Groq/OpenAI/Gemini SDK calls anywhere in this module.
- **Sidebar**: Chat, Analytics, About Module, Medical Knowledge Assistant,
  Dynamic Knowledge Base, Research Assistant (plus remaining "coming
  soon" placeholders).

### Upload workflow

Identical shape to Milestone 3's (validate → hash → dedupe check →
extract → chunk → embed new chunks only → index incrementally → persist
metadata), but PDF-only and using the `research_papers` SQLite table
instead of `knowledge_documents`.

### Question answering

```
Question -> retrieve top-K relevant passages (embeds query, searches
FAISS) -> build a prompt containing only those passages -> existing
configurable LLM provider -> answer + citations
```

If no chunk meets the similarity threshold, the LLM is told explicitly
that no relevant context was found, so it says so rather than guessing —
the same grounding pattern as Milestone 2's medical assistant.

### Research summarization

Selecting a paper and clicking "Generate Summary" retrieves **all** of
that paper's indexed chunks (not just the top-K similarity matches used
for Q&A) via `get_chunks_for_document()`, concatenates them up to a
configurable character budget (`RESEARCH_MAX_SUMMARY_CONTEXT_CHARS`,
default 16,000), and asks the LLM to produce the six required sections
strictly from that content.

### Citation support

Every Q&A answer and every retrieved passage displays: the source
paper's filename, the actual retrieved excerpt, and its similarity
score — all three always come from a real `RetrievedPassage`, never
synthesized.

### Paper management

- **List**: filename, upload date, chunk count, status.
- **Delete** (🗑️): removes the paper's vectors (via `remove_ids()`),
  its stored extracted text, and its metadata row — verified directly
  that this never touches any other paper's vectors or search results.
- **Re-index** (🔄): re-chunks and re-embeds one paper from its stored
  text (e.g. after a config change), without needing the original PDF
  again.

### Running the application

```bash
streamlit run app.py
```

Open the sidebar → **📄 Research Assistant**, upload a PDF, then try the
Question Answering, Paper Summary, and Paper Management sections.

### Future improvements

- Multi-paper comparative Q&A ("how do these two papers' methodologies differ?")
- Section-aware chunking (detect Abstract/Methods/Results boundaries)
- Export summaries as PDF/Markdown
- Support for arXiv URL ingestion, not just local file upload

## Milestone 5: Multimodal AI

### Feature overview

- **Upload an image** (PNG/JPG/JPEG), preview it, and ask questions about
  it using a vision-capable LLM.
- **Vision capability detection** — before analyzing, the app checks
  whether the *currently selected provider and model* actually support
  image input. If not, it shows a clear message ("The currently selected
  model does not support image analysis. Please select a vision-capable
  model.") instead of failing confusingly or crashing.
- **Genuinely extends the LLM Provider Abstraction** — this was the one
  piece of real infrastructure work this milestone needed: `LLMProvider`
  gained two new methods, `supports_vision(model)` and
  `generate_with_image()`, both with safe no-op defaults so every
  existing provider keeps working unchanged. OpenAI and Groq have real
  vision implementations (both use the same `image_url` content-part
  format, since Groq's API is OpenAI-compatible). Gemini's vision
  implementation was completed in a later bug-fix pass — see "Bug Fix:
  Gemini Provider" earlier in this README.
- **Suggested questions** — one-click buttons for "Describe this image,"
  "What objects are visible?," "Explain what is happening," and
  "Summarize this image."
- **Conversation history** — every image filename, question, and AI
  response is saved (SQLite), viewable in a "Previous Analyses" section
  for the current session.
- **Reuses the existing configurable LLM provider** — no direct
  Groq/OpenAI/Gemini SDK calls anywhere in this module; provider
  selection works exactly as it does for text chat.
- **Sidebar**: Chat, Analytics, About Module, Medical Knowledge Assistant,
  Dynamic Knowledge Base, Research Assistant, Multimodal AI (plus
  remaining "coming soon" placeholders).

### Supported image formats

PNG, JPG, JPEG. Validation uses Pillow (already installed transitively
via Streamlit) to confirm the uploaded bytes are an actual, openable
image — not just trusting the file extension; a renamed non-image file
is caught and rejected with a friendly message.

### Vision model requirements

Whether image analysis works depends entirely on which `LLM_PROVIDER` /
`LLM_MODEL` you've configured:

| Provider | Vision-capable models |
|---|---|
| OpenAI | `gpt-4o`, `gpt-4o-mini` (the default), `gpt-4-turbo`, and dated snapshots of these |
| Groq | Llama-Vision models (e.g. `llama-3.2-11b-vision-preview`) — **not** the default `llama-3.1-8b-instant`, which is text-only |
| Gemini | `gemini-1.5`+ / `gemini-2`+ family (the default `gemini-1.5-flash` qualifies) |

If you're on Groq with its default text model, the Multimodal AI page will
show the "please select a vision-capable model" notice until you set
`LLM_MODEL` to a vision model in `.env`.

### Image analysis workflow

```
Upload image
  -> validate (format, size, genuine-image check via Pillow)
  -> preview
  -> vision-capability check (current provider + model)
  -> user asks a question
  -> existing configurable LLM provider, called with image + prompt
  -> AI response displayed + saved to conversation history
```

### Running the application

```bash
streamlit run app.py
```

Open the sidebar → **🖼️ Multimodal AI**, upload a PNG/JPG/JPEG, then ask
a question (or click one of the suggested questions) and click Analyze.

### Future improvements

- Multi-image comparison in a single request
- Persist uploaded image files themselves (currently only filename +
  Q&A text are stored, not the image bytes — an intentional
  storage-efficiency choice matching the PRD's stated storage
  requirements, but worth reconsidering if "view past images" becomes a need)
- Bounding-box/object-detection style structured output, not just free text
- Let users switch models directly from this page (currently `.env`-only)

## Project structure

```
advanced-genai-chatbot/
├── app.py                        # Streamlit entry point: navigation + chat + analytics + about + medical + knowledge base
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
│   ├── knowledge_base/               # Milestone 3: implemented
│   │   ├── config.py                    # Self-contained, env-driven config for this milestone
│   │   ├── parser.py                     # File validation + text extraction (PDF/TXT/MD)
│   │   ├── chunker.py                     # Chunking (reuses medical/preprocess.clean_text)
│   │   ├── embeddings.py                   # Reuses SentenceTransformerEmbedder + new hashing fallback
│   │   ├── vector_store.py                  # Incremental FAISS add/rebuild/search
│   │   ├── metadata.py                       # Document metadata + duplicate detection
│   │   ├── updater.py                         # "Update Index" (pending documents only)
│   │   ├── manager.py                          # Upload/search/stats orchestration
│   │   └── knowledge_chat.py                    # Streamlit page
│   ├── research/                      # Milestone 4: implemented
│   │   ├── config.py                     # Reuses KnowledgeBaseConfig type (composition, not duplication)
│   │   ├── parser.py                      # PDF-only validation; reuses knowledge_base's extract_text
│   │   ├── chunker.py                      # Reuses chunk_document/KnowledgeChunk from knowledge_base
│   │   ├── embeddings.py                    # Reuses KnowledgeEmbeddingGenerator directly
│   │   ├── vector_store.py                   # Subclasses KnowledgeVectorStore; adds delete_document()
│   │   ├── retriever.py                       # Similarity-filtered retrieval + retrieve_from_paper()
│   │   ├── citation.py                         # Builds citation objects from retrieved passages
│   │   ├── summarizer.py                        # Six-section structured paper summaries
│   │   ├── manager.py                            # Upload/retrieve/summarize/delete/reindex orchestration
│   │   ├── research_pipeline.py                   # Grounded question-answering
│   │   └── research_chat.py                        # Streamlit page
│   ├── multimodal/                     # Milestone 5: implemented
│   │   ├── config.py                     # Self-contained, env-driven config for this milestone
│   │   ├── image_loader.py                # Validation via Pillow (genuine-image check, not just extension)
│   │   ├── vision_client.py                # Thin wrapper around llm_client's vision functions
│   │   ├── manager.py                       # Validate/analyze/history orchestration
│   │   └── multimodal_chat.py                # Streamlit page
│   └── multilingual/                    # Not yet implemented
├── utils/
│   ├── logger.py                    # Shared logging setup
│   ├── storage.py                     # SQLite data access (messages, sentiment, medical_queries, knowledge_documents, research_papers, multimodal_conversations)
│   ├── llm_client.py                    # Unified, provider-agnostic chat completion client (+ vision functions)
│   └── providers/                         # LLM provider abstraction
│       ├── base_provider.py                 # Abstract interface + shared exceptions + vision interface
│       ├── openai_provider.py                 # OpenAI implementation (text + vision)
│       ├── groq_provider.py                     # Groq implementation (text + vision)
│       └── gemini_provider.py                     # Gemini implementation (production, text + vision)
├── data/, database/, uploads/, vector_store/, assets/, docs/
└── tests/
    ├── test_config.py
    ├── test_storage.py
    ├── test_llm_client.py
    ├── test_llm_providers.py           # LLM Provider Abstraction (+ vision capability tests)
    ├── test_navigation.py
    ├── test_sentiment.py                 # Milestone 1
    ├── test_medical.py                    # Milestone 2
    ├── test_knowledge_base.py               # Milestone 3
    ├── test_research.py                       # Milestone 4
    ├── test_multimodal.py                       # Milestone 5
    └── test_gemini_provider.py                     # Bug Fix: Gemini Provider
```

## Running tests

```bash
pytest tests/ -v
```

215 tests currently pass: config defaults, the storage layer (sentiment
columns, medical_queries table, knowledge_documents table,
research_papers table, multimodal_conversations table, summary
queries), the LLM provider abstraction (including per-provider vision
capability detection: OpenAI's `gpt-4o` family, Groq's Llama-Vision
models, Gemini's `1.5`+ family, and confirming text-only default
models like Groq's `llama-3.1-8b-instant` correctly report no vision
support), the navigation registry, the sentiment analyzer, the medical
RAG pipeline (dataset parsing, chunking, embeddings with fallback,
vector store build/cache/search, retrieval filtering, prompt
construction, end-to-end answer generation with mocked LLM calls), the
dynamic knowledge base (file validation, text extraction for
PDF/TXT/MD, chunking, the hashing fallback embedder, incremental
vector-store add/persist/rebuild, duplicate detection, and the full
upload/search/update/rebuild manager workflow), the research assistant
(PDF-only validation, chunking/embedding/retrieval reuse, **per-paper
deletion leaving other papers' vectors untouched**, re-indexing,
structured summarization, and grounded citation-backed question
answering with mocked LLM calls), and the multimodal assistant (image
validation via Pillow -- including rejecting a renamed non-image file
with a valid extension, not just trusting it -- vision-not-supported
and missing-API-key error paths, and per-session conversation history
isolation), and the production Gemini provider (request construction
for both text and vision calls verified against the real `google-genai`
SDK types, plus error mapping for auth/rate-limit/timeout/connection
failures -- all via mocked SDK responses, since this dev sandbox has no
network access to Google's API endpoint). All tests use small synthetic
data (including reportlab-generated test PDFs and Pillow-generated test
images) -- none require downloading the full medical dataset or network access.

## Dependencies

`requirements.txt`: Streamlit, `python-dotenv`, `pytest` (foundation);
`openai` + `groq` + `google-genai` (LLM Provider Abstraction — all three
providers are fully working; `google-genai` is Google's current SDK, not
the legacy `google-generativeai` package); `transformers` + `torch`
(Milestone 1 sentiment model); `pandas` + `plotly` (Milestone 1 analytics
dashboard); `sentence-transformers` + `faiss-cpu` + `langchain` +
`langchain-text-splitters` + `scikit-learn` (Milestone 2 RAG pipeline;
scikit-learn also powers Milestone 3's incremental-friendly hashing
fallback embedder); `pypdf` (Milestone 3 PDF text extraction -- the only
new dependency that milestone needed, since embeddings/vector
search/chunking all reuse Milestone 2's already-installed libraries).
**Milestone 4 (Research Assistant) needed zero new dependencies** --
its PDF parsing, chunking, and embedding pipeline all import directly
from `modules/knowledge_base`, and its one genuinely new capability
(per-paper deletion) uses a FAISS method (`remove_ids()`) already
available in the already-installed `faiss-cpu` package. `pillow`
(Milestone 5 image validation -- was already installed transitively via
Streamlit; pinned explicitly now that `modules/multimodal` imports it
directly). Vision support itself needed **no new SDK** -- OpenAI, Groq,
and Gemini's vision calls all use their already-installed clients, since
image input is just a different content shape on the same underlying
completions endpoint.
`opencv-python-headless` (reserved, not required by Milestone 5's actual
scope) stays commented out until actually needed.

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
- The Dynamic Knowledge Base is additive-only: there's no per-document
  delete/remove operation yet. Removing a document today requires
  manually deleting its stored text file and running "Rebuild Index".
- Like Milestone 2, the knowledge base's Sentence Transformers backend
  needs network access on first run; it falls back to a stateless hashing
  vectorizer otherwise (verified directly — searches remain functional,
  though semantic quality is lower than real embeddings, which is the
  expected trade-off for a dependency-free fallback).
- The "Update Index" button only catches documents left in a non-`indexed`
  status (e.g. a partial failure); since `process_upload()` already
  indexes successfully-processed documents immediately, there's normally
  nothing for it to do — this is by design, not a bug.
- The Research Assistant only accepts PDF (per the PRD); unlike the
  Dynamic Knowledge Base, TXT/Markdown papers aren't supported.
- Like Milestones 2 and 3, the research assistant's Sentence Transformers
  backend needs network access on first run; it falls back to the same
  stateless hashing embedder otherwise (verified directly, including that
  a deleted paper's vectors are correctly excluded from search results
  under the fallback backend too).
- Paper summarization caps context at `RESEARCH_MAX_SUMMARY_CONTEXT_CHARS`
  (default 16,000 characters) and `RESEARCH_MAX_SUMMARY_CHUNKS` (default
  40); extremely long papers may have later sections excluded from the
  summary. Both are configurable via `.env`.
- Re-indexing a paper requires its extracted text to still be stored on
  disk (saved automatically at upload time); if that file was manually
  deleted outside the app, re-index will fail with a clear error rather
  than silently doing nothing.
- Image analysis depends entirely on your configured provider/model
  supporting vision — Groq's *default* model does not (its Llama-Vision
  models do, but you must set `LLM_MODEL` explicitly); the page detects
  this and shows a clear message rather than a confusing API error.
- Uploaded images themselves aren't persisted to disk — only the
  filename, question, and AI response are stored, matching the PRD's
  stated conversation-storage requirements exactly. This means past
  images can't be re-displayed from history, only their filename/Q&A text.
- Conversation history in the Multimodal AI page is per-session (cleared
  when the browser session ends), consistent with how session state
  works elsewhere in the app (e.g. the base Chat page).



## Submission Status

### Completed Milestones
- ✅ Milestone 1 — Sentiment Analysis
- ✅ Milestone 2 — Medical Q&A / RAG
- ✅ Milestone 3 — Dynamic Knowledge Base
- ✅ Milestone 4 — Research Assistant
- ✅ Milestone 5 — Multimodal AI

### Not Included in This Submission
- ⏭️ Milestone 6 — Multilingual AI

Milestone 6 is intentionally not included in this submission.
The current submission scope is Milestones 1–5, with all completed
features fully integrated and tested.