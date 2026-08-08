# Requirements Documentation

This document explains every dependency in `requirements.txt` — what it's
for, and which milestone introduced it. For the actual installable file,
see [`requirements.txt`](./requirements.txt).

```bash
pip install -r requirements.txt
```

---

## Foundation

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | `>=1.38,<2.0` | Web UI framework — the entire app's frontend |
| `python-dotenv` | `>=1.0,<2.0` | Loads configuration from `.env` (no hardcoded secrets/paths) |
| `pytest` | `>=8.2,<9.0` | Test runner for the whole project |

## LLM Provider Abstraction (architectural enhancement)

| Package | Version | Purpose |
|---|---|---|
| `openai` | `>=1.40,<2.0` | SDK for the OpenAI provider option |
| `groq` | `>=0.9,<1.0` | SDK for the Groq provider option |
| `google-genai` | `>=1.0,<3.0` | SDK for the Gemini provider option |

Only the SDK for whichever provider you select via `LLM_PROVIDER` in
`.env` actually needs to work at runtime — all three are installed so you
can switch freely with zero code changes.

> **Note on `google-genai` vs `google-generativeai`:** Google publishes
> two Python packages for Gemini. `google-generativeai` is the older,
> now-legacy SDK. `google-genai` is the current, actively maintained,
> unified SDK. This project uses **`google-genai` only** — do not install
> `google-generativeai`, the Gemini provider does not use it.

## Milestone 1 — Sentiment Analysis

| Package | Version | Purpose |
|---|---|---|
| `transformers` | `>=4.42,<5.0` | HuggingFace library; loads the sentiment classification model |
| `torch` | `>=2.3,<3.0` | Backend the transformer model runs on (CPU) |
| `pandas` | `>=2.2,<3.0` | Data handling for the Analytics dashboard |
| `plotly` | `>=5.22,<6.0` | Pie/bar charts on the Analytics dashboard |

If `transformers`/`torch` can't download the model (no network access to
HuggingFace Hub), the app automatically falls back to a dependency-free
rule-based sentiment scorer — it never breaks the app.

## Milestone 2 — Medical Knowledge Assistant (RAG)

| Package | Version | Purpose |
|---|---|---|
| `sentence-transformers` | `>=3.0,<4.0` | Generates embeddings for medical text (semantic search) |
| `faiss-cpu` | `>=1.8,<2.0` | Vector similarity search index |
| `langchain` | `>=0.2,<0.3` | Framework used for the RAG pipeline structure |
| `langchain-text-splitters` | `>=0.2,<0.3` | Splits long medical answers into overlapping chunks |
| `scikit-learn` | `>=1.5,<2.0` | Powers the TF-IDF fallback embedder (used if Sentence Transformers can't load) |

## Milestone 3 — Dynamic Knowledge Base

| Package | Version | Purpose |
|---|---|---|
| `pypdf` | `>=4.2,<5.0` | Extracts text from uploaded PDF documents |

This milestone deliberately reuses `sentence-transformers`, `faiss-cpu`,
`langchain-text-splitters`, and `scikit-learn` from Milestone 2 rather
than adding duplicate dependencies — `pypdf` was the only genuinely new
package needed (for PDF parsing, which Milestone 2 never required).

## Milestone 4 — Research Assistant

**No new dependencies.** This milestone fully reuses Milestone 2/3's
already-installed stack (`sentence-transformers`, `faiss-cpu`, `pypdf`,
`langchain-text-splitters`, `scikit-learn`) via direct imports from
`modules/knowledge_base` — parsing, chunking, and embeddings are not
reimplemented. Its one genuinely new capability, per-paper deletion,
uses FAISS's `remove_ids()` method, already available in the
already-installed `faiss-cpu` package.

## Milestone 5 — Multimodal AI

| Package | Version | Purpose |
|---|---|---|
| `pillow` | `>=10.4,<11.0` | Validates uploaded images (confirms they're genuine, openable images) |

`pillow` was already installed transitively via Streamlit itself; it's
pinned explicitly here now that `modules/multimodal/image_loader.py`
imports it directly. Vision support itself needed **no new SDK** — image
input to OpenAI/Groq/Gemini all use their already-installed clients,
since an image is just a different content shape on the same underlying
completions endpoint, not a separate API.

## Bug Fix — Gemini Provider

The Gemini provider was originally a placeholder (see the "Bug Fix:
Gemini Provider" section in `README.md` for the root-cause analysis).
Fixing it activated `google-genai` as a real dependency — see the "LLM
Provider Abstraction" section above. No other packages were needed.

## Not yet active (commented out in `requirements.txt`)

| Package | For | Notes |
|---|---|---|
| `opencv-python-headless` | Reserved | Not required by any implemented milestone's actual scope; listed in the Master PRD's long-term tech stack for possible future, more advanced image processing |

Uncomment this line only when a milestone that actually needs it is implemented.

---

## Version pinning philosophy

Every dependency uses a `>=X,<Y` range (not an exact pin) — this allows
patch/minor updates (bug fixes, security patches) while blocking major
version bumps that could introduce breaking changes. If you need exact
reproducibility (e.g. for a production deployment), consider generating a
fully-pinned lockfile with `pip freeze > requirements.lock.txt` from a
known-working environment.
