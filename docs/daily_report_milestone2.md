# Daily Report — Milestone 2: Medical Knowledge Assistant (RAG)

**Objective**: Implement a Retrieval-Augmented Generation medical assistant
using the MedQuAD dataset, grounded entirely in retrieved context (never
the LLM's own unguided knowledge), integrated into the existing chatbot
foundation without regressing Milestone 1 or the LLM Provider Abstraction.

**Completed Tasks**:
- Built `modules/medical/`: `config.py` (self-contained, env-driven --
  root `config.py` was out of this PRD's folder scope), `loader.py`
  (MedQuAD download + XML parsing + caching), `preprocess.py` (cleaning +
  chunking via LangChain), `embeddings.py` (Sentence Transformers primary,
  TF-IDF fallback), `vector_store.py` (FAISS build/save/load/skip-if-cached),
  `retriever.py` (top-K similarity search), `prompts.py` (grounded-answer
  prompt templates), `rag_pipeline.py` (end-to-end orchestration reusing
  the existing `utils/llm_client.py`), `medical_chat.py` (Streamlit page)
- Extended `utils/storage.py` with a new, separate `medical_queries` table
  (additive migration; base `messages` table untouched)
- Wired the page into `app.py`'s router and marked it `implemented=True`
  in `components/navigation.py`
- Verified the real MedQuAD dataset downloads automatically (single-request
  zip from GitHub, ~16 MB) and parses correctly against its actual XML schema
- Discovered and fixed a real dataset quirk during testing: the
  alphabetically-first folder (ADAM) has blanked-out `<Answer/>` tags for
  copyright reasons, so a naive "first N files" cap landed entirely inside
  it; fixed by sampling round-robin across every source folder instead
- Added 26 new tests (`tests/test_medical.py`) plus 5 more in
  `tests/test_storage.py` for the new table -- all using small synthetic
  data, no network/dataset download required in CI
- Ran the full suite: **79/79 tests passing**, confirming Milestone 1 and
  the LLM Provider Abstraction still work with zero regressions

**Testing performed** (live, not just unit tests):
- Downloaded and parsed the real MedQuAD dataset end-to-end (1,767 QA
  pairs across 9 source categories in ~10s with the default file cap)
- Built the FAISS index (5,658 chunks) and confirmed cache-skip works
  (0.3s on unchanged rebuild vs. full build)
- Ran real retrieval queries ("What are the symptoms of diabetes?", "What
  causes asthma?", "How is high blood pressure treated?") and confirmed
  semantically relevant results even via the TF-IDF fallback
- Verified the full pipeline with a mocked LLM call: correct grounded
  prompt construction, correct source attribution and deduplication
- Verified error handling: empty question (ValueError), pipeline not
  initialized (RuntimeError), missing LLM API key
  (LLMConfigurationError propagates correctly), corrupted XML file
  (skipped with a warning, load continues)
- Full app smoke test: boots, HTTP 200, no runtime errors

**Testing performed on the user's own machine (real deployment verification):**
- Installed dependencies and ran the full suite: `79 passed`
- Caught a real deployment issue: `requirements.txt` on the deployed
  machine was a stale copy missing the Milestone 2 dependency section
  (`sentence-transformers`, `faiss-cpu`, `langchain`, `scikit-learn` were
  still commented out under "Future milestone dependencies" instead of
  active) -- confirmed by the `pip install -r requirements.txt` log not
  mentioning any of those packages. Fixed by replacing the file's
  contents with the correct, up-to-date version and re-verifying `79
  passed` afterward.
- Live-tested the actual UI end-to-end with a real Groq API key: Chat
  page (sentiment badge + confidence + adaptive tone all working),
  Medical Knowledge Assistant (real question "What causes asthma?" ->
  grounded answer citing retrieved passages, "Knowledge base ready: 5,658
  indexed passages, embedding backend: sentence-transformer" -- confirming
  the real Sentence Transformers model downloaded successfully in this
  environment, an upgrade over the dev sandbox's TF-IDF fallback path)
- Noted one benign `FutureWarning` (a `sentence-transformers` method
  rename, `get_sentence_embedding_dimension` -> `get_embedding_dimension`)
  -- does not affect functionality or test results; flagged as a
  non-blocking follow-up, not fixed in this pass to avoid re-testing
  scope creep right before the milestone push.

**Challenges**:
- This build sandbox has no network access to HuggingFace Hub, so the
  Sentence Transformers model can't be downloaded here (same constraint
  as Milestone 1's sentiment model).
- The MedQuAD dataset's ADAM folder having empty answers wasn't obvious
  until actually testing against the real download.
- A stale `requirements.txt` on the deployment machine (missing the
  Milestone 2 dependency section) wasn't caught until running
  `pip install` there and noticing the installed-package log didn't
  mention any RAG-related libraries.

**Solutions**:
- Reused Milestone 1's established resilience pattern: added a TF-IDF
  fallback embedder, verified it produces genuinely relevant retrieval
  results, and made the backend used fully visible in the UI/logs.
- Fixed the file-selection logic to sample round-robin across all source
  folders rather than taking a naive alphabetical slice -- this was
  caught by testing against the real dataset, not just synthetic data.
- Replaced the deployed `requirements.txt` with the correct, current
  version and re-ran the full test suite (`79 passed`) to confirm the fix.

**Files Created**: `modules/medical/config.py`, `loader.py`, `preprocess.py`,
`embeddings.py`, `vector_store.py`, `retriever.py`, `prompts.py`,
`rag_pipeline.py`, `medical_chat.py`, `tests/test_medical.py`,
`docs/daily_report_milestone2.md`.

**Files Modified**: `modules/medical/__init__.py`, `app.py`,
`components/navigation.py`, `utils/storage.py`, `requirements.txt`,
`.gitignore`, `README.md`, `SUMMARY.md`, `tests/test_navigation.py`,
`tests/test_storage.py`.

**Git Commit Summary**: `Milestone 2 - Medical Knowledge Assistant (RAG)`

**Next Day Plan**: Await the Milestone 3 PRD (Dynamic Knowledge Base)
before any further implementation, per the current development workflow.
