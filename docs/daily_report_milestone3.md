# Daily Report — Milestone 3: Dynamic Knowledge Base

**Objective**: Let the chatbot continuously expand its knowledge by
uploading documents (PDF/TXT/Markdown) that are incrementally indexed
into the existing FAISS vector infrastructure, without rebuilding
anything else and without regressing the Foundation, Milestone 1,
Milestone 2, or the LLM Provider Abstraction.

**Completed Tasks**:
- Built `modules/knowledge_base/`: `config.py` (self-contained, env-driven
  — root `config.py` was out of this PRD's scope, same as Milestone 2),
  `parser.py` (file validation + PDF/TXT/MD text extraction via a small
  extensible registry), `chunker.py` (chunking, reusing `clean_text()`
  from `modules.medical.preprocess` rather than duplicating it),
  `embeddings.py` (reuses `SentenceTransformerEmbedder` directly from
  Milestone 2; adds a new stateless `HashingEmbedder` fallback since
  Milestone 2's TF-IDF fallback requires a fixed, pre-fit corpus and
  can't support true incremental indexing), `vector_store.py`
  (incremental FAISS `IndexIDMap` add/search, plus an explicit full
  rebuild path — genuinely new logic, since Milestone 2's vector store
  only does rebuild-or-skip), `metadata.py` (document tracking +
  duplicate detection, wrapping new `utils/storage.py` functions),
  `updater.py` ("Update Index": reprocesses only non-indexed documents),
  `manager.py` (upload/search/stats/index-control orchestration),
  `knowledge_chat.py` (Streamlit page: upload, stats, search, document
  list, index controls)
- Extended `utils/storage.py` with a new, separate `knowledge_documents`
  table (additive migration; `messages` and `medical_queries` untouched)
- Wired the page into `app.py`'s router and marked it `implemented=True`
  in `components/navigation.py`
- Added 29 new tests (`tests/test_knowledge_base.py`) plus 6 more in
  `tests/test_storage.py` for the new table — all using small synthetic
  documents, no network access required
- Ran the full suite: **114/114 tests passing**, confirming the
  Foundation, Milestone 1, Milestone 2, and the LLM Provider Abstraction
  all still work with zero regressions

**Testing performed** (live, not just unit tests):
- Uploaded real TXT, Markdown, and PDF (generated with `reportlab`, text
  extracted via `pypdf`) files end-to-end through the actual manager
- Verified duplicate detection: re-uploading identical content under a
  different filename was correctly rejected with a clear message
- Verified error handling: empty file, unsupported format (`.png`),
  corrupted PDF, and a valid-but-blank PDF (no extractable text) — all
  produced friendly errors, no crashes
- Verified true incremental indexing: uploaded 3 documents one at a time
  and confirmed each became immediately searchable without disturbing the
  others' vectors (`chunk_count` grew by exactly 1 chunk per document,
  not via a full rebuild)
- Verified "Update Index" correctly no-ops when nothing is pending, and
  "Rebuild Index" correctly regenerates the whole index while keeping
  everything searchable afterward
- Full app smoke test: boots, HTTP 200, no runtime errors

**Testing performed on the user's own machine (real deployment verification):**
- Ran the full suite on the deployed machine: `114 passed`
- Live-tested the actual UI with a real document: uploaded a real PDF
  ("StuWallet PRD.pdf", 126.7 KB) -> 34 chunks indexed, embedding backend
  correctly showing `sentence-transformer` (the real model, since this
  machine has HuggingFace Hub access -- an upgrade over the dev sandbox's
  hashing fallback path)
- Re-uploaded the identical PDF and confirmed duplicate detection caught
  it: "has already been indexed"
- Searched the indexed PDF ("stuwallet") and got back 4 ranked results
  with distinct relevance scores (0.39/0.30/0.20/0.16), confirming real
  semantic search rather than keyword matching
- Caught a second real deployment issue, same class as Milestone 2's:
  the deployed `requirements.txt` was missing the entire "Milestone 3"
  section (`pypdf`) -- worked locally only because `pypdf` happened to
  already be installed from earlier manual testing. Fixed by replacing
  the file with the correct, complete version and confirming the
  Milestone 3 section (lines 28-31) is present.

**Challenges**:
- Milestone 2's TF-IDF fallback embedder is fit once on a static corpus
  — realized early on that this fundamentally doesn't fit "documents
  must become searchable incrementally," since a pre-fit vectorizer
  can't represent vocabulary from documents it hasn't seen yet.
- A stale/incomplete `requirements.txt` on the deployment machine (missing
  the whole Milestone 3 section) wasn't caught by local testing since
  `pypdf` was already installed there from earlier manual verification.

**Solutions**:
- Used a `HashingVectorizer`-based fallback instead: stateless by
  design, so it embeds brand-new text correctly with zero refitting —
  this was new code because it solves a genuinely different problem
  than Milestone 2's fallback, not a duplication of it. Reused everything
  else from Milestone 2 that *was* directly reusable (the Sentence
  Transformers backend, the text-cleaning function).
- Replaced the deployed `requirements.txt` with the correct, complete
  version and re-verified the Milestone 3 section is present. Also added
  `REQUIREMENTS.md`, a human-readable dependency doc (per-package purpose
  and which milestone introduced it), to make this kind of drift easier
  to spot by inspection going forward.

**Files Created**: `modules/knowledge_base/config.py`, `parser.py`,
`chunker.py`, `embeddings.py`, `vector_store.py`, `metadata.py`,
`updater.py`, `manager.py`, `knowledge_chat.py`,
`tests/test_knowledge_base.py`, `docs/daily_report_milestone3.md`.

**Files Modified**: `modules/knowledge_base/__init__.py`, `app.py`,
`components/navigation.py`, `utils/storage.py`, `requirements.txt`,
`.gitignore`, `README.md`, `SUMMARY.md`, `tests/test_navigation.py`,
`tests/test_storage.py`.

**Git Commit Summary**: `Milestone 3 - Dynamic Knowledge Base`

**Next Day Plan**: Await the Milestone 4 PRD (Research Assistant) before
any further implementation, per the current development workflow.
