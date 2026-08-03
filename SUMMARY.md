# SUMMARY

**Task**: Implement Milestone 3 (Dynamic Knowledge Base) on the existing
foundation + Milestone 1 + LLM Provider Abstraction + Milestone 2, per the
Milestone 3 PRD. Scope: `modules/knowledge_base/`, `components/`,
`database/`, `utils/`, `README.md`, `requirements.txt`, `app.py`.

**Stack**: PDF/TXT/Markdown upload -> pypdf/plain-text extraction ->
LangChain chunking (reuses Milestone 2's `clean_text()`) -> Sentence
Transformers embeddings (reuses Milestone 2's embedder class directly) or
a new stateless hashing fallback -> incremental FAISS `IndexIDMap` ->
existing configurable LLM provider for any future generation needs.

**Included**:
- True incremental indexing: new documents add to the FAISS index without touching existing vectors or rebuilding
- Explicit, separate "Rebuild Index" path for full regeneration
- Duplicate detection via SHA-256 content hash (not filename)
- Extensible file-format registry (PDF/TXT/MD today; new formats = one function + one registry entry)
- New, separate `knowledge_documents` SQLite table (additive, doesn't touch `messages` or `medical_queries`)
- "Update Index" (reprocess only pending/failed docs) vs "Rebuild Index" (full regenerate) as genuinely distinct operations
- Reuses Milestone 2's `SentenceTransformerEmbedder` and `clean_text()` directly via import -- per the PRD's explicit "do not duplicate existing RAG functionality" instruction
- New code written only where genuinely necessary: `HashingEmbedder` (Milestone 2's TF-IDF fallback can't support incremental use since it requires a fixed, pre-fit corpus) and the incremental-add-capable vector store
- Explicit error handling: invalid format, empty file, corrupted PDF, duplicate upload, embedding failure, vector DB failure, metadata-save failure -- all friendly messages, never crashes
- 114 tests total (35 new/updated for this milestone), all passing
- Full app smoke-tested; Milestone 1, Milestone 2, and the LLM Provider Abstraction confirmed regression-free

**Real-world verification** (not just unit tests): uploaded real TXT/MD/PDF files (PDF generated with reportlab, parsed with pypdf) through the actual manager; confirmed duplicate detection, true incremental search (each new doc immediately searchable without disturbing prior vectors), and both Update/Rebuild Index operations behaving correctly.

**Not touched**: `modules/research/`, `modules/multimodal/`, `modules/multilingual/` -- still empty placeholders. `modules/medical/` and `modules/sentiment/` untouched (only imported from, never modified). Root `config.py` intentionally untouched, same reasoning as Milestone 2.

**Not included / needs attention before production**:
- No per-document delete; removing a document requires deleting its stored text file + a full rebuild
- Sentence Transformers needs network access on first run (falls back gracefully otherwise -- verified)
- No authentication on the Streamlit app; no rate limiting

**How to run**:
```bash
pip install -r requirements.txt
cp .env.example .env   # set LLM_PROVIDER + that provider's key
streamlit run app.py   # sidebar -> Dynamic Knowledge Base to upload/search documents
pytest tests/ -v
```
