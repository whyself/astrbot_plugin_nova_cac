# Agentic Local RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the existing preselected local retrieval plus direct LLM call with the two-stage evidence-first Agent workflow used by `astrbot_plugin_nju_qa`, adapted to the embedded local NOVA Markdown pack.

**Architecture:** Keep `/cac`, mandatory Pack loading, answer voice, bounded `/cac` history, help/reset behavior, and session serialization unchanged. Build persistent Markdown chunks, a BM25-style keyword index, and a Chroma vector index backed by an AstrBot Embedding Provider. During research the LLM autonomously chooses hybrid search, exact grep, document reading, outline, and listing tools; during answering all tools are closed and only recorded evidence may be used.

**Tech Stack:** Python 3.10+, AstrBot `FunctionTool`/`tool_loop_agent`, SQLite, ChromaDB, AstrBot Embedding Provider API, `unittest`, Ruff.

---

### Task 1: Define local corpus and retrieval models

**Files:**
- Create: `nova_cac/models.py`
- Create: `nova_cac/chunking.py`
- Create: `nova_cac/chunk_store.py`
- Create: `nova_cac/local_corpus.py`
- Test: `tests/test_agentic_retrieval.py`

- [ ] Write failing tests proving Markdown front matter is retained as metadata, heading-aware chunks receive stable IDs, local files are rescanned by content signature, and changed/deleted documents replace stale chunks.
- [ ] Run `python -m unittest tests.test_agentic_retrieval -v` and confirm missing-module failures.
- [ ] Implement immutable document/chunk/search models, reference-compatible Markdown chunking, SQLite chunk persistence, and read-only scanning of `knowledge_pack/knowledge`.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Implement keyword and vector hybrid retrieval

**Files:**
- Create: `nova_cac/keyword_index.py`
- Create: `nova_cac/vector_index.py`
- Create: `nova_cac/retriever.py`
- Modify: `requirements.txt`
- Modify: `_conf_schema.json`
- Test: `tests/test_agentic_retrieval.py`

- [ ] Add failing tests for BM25 title/path boosts, vector-plus-keyword score merging, per-document diversity, threshold filtering, and keyword fallback.
- [ ] Port the reference BM25 tokenizer and persistent Chroma adapter, changing collection identity to NOVA CAC.
- [ ] Resolve embeddings through the configured AstrBot Embedding Provider; use the first available provider when no ID is specified and degrade to keyword-only retrieval with a logged reason when unavailable.
- [ ] Add `chromadb>=0.4,<0.6` and configuration for embedding provider ID, vector enablement, score threshold, chunk size/overlap, and diagnostics.
- [ ] Run focused retrieval tests.

### Task 3: Implement Agent tools and evidence tracking

**Files:**
- Create: `nova_cac/evidence.py`
- Create: `nova_cac/tools.py`
- Create: `nova_cac/prompts.py`
- Test: `tests/test_agent_tools.py`

- [ ] Write failing tests for hybrid search evidence, exact grep with line numbers, safe scoped reads, document listing, and outline extraction.
- [ ] Implement a `SourceTracker`-compatible evidence model and the tools `search_knowledge_base`, `grep_local_docs`, `read_doc`, `list_docs`, and `get_doc_outline`.
- [ ] Ensure search results are candidates only; concrete answer evidence is recorded by search snippets and exact document reads with traceable title/path/URL/line range.
- [ ] Run tool tests.

### Task 4: Port the two-stage evidence-first Agent

**Files:**
- Create: `nova_cac/agent.py`
- Test: `tests/test_agent_runtime.py`

- [ ] Write failing tests for research/answer separation, autonomous tool exposure only in research, no-evidence behavior, evidence-marker retry, marker stripping, conditional source display, Pack prompt inclusion, and propagation of recent `/cac` contexts.
- [ ] Port `NjuQaAgent` as `NovaCacAgent`: research through `tool_loop_agent`, select bounded concrete evidence, answer with no tools, validate `[E#]` usage, strip internal markers, reject unverified URLs, and preserve natural output.
- [ ] Combine the freshly read `AGENTS.md`, `soul.md`, `spirit.md`, and `voice.md` with research and answer instructions on every generated answer.
- [ ] Append a source list only when the user explicitly asks for sources, preserving the existing Pack behavior.
- [ ] Run Agent runtime tests.

### Task 5: Replace the current main integration

**Files:**
- Modify: `main.py`
- Modify: `nova_cac/core.py`
- Modify: `nova_cac/__init__.py`
- Delete obsolete retrieval tests and code from: `tests/test_core.py`

- [ ] Replace `KnowledgeIndex.search` and `context.llm_generate` with local corpus refresh plus `NovaCacAgent.answer`.
- [ ] Keep literal `/cac`, group/private parity, help/reset, six-turn/character-bounded plugin memory, and per-session locks unchanged.
- [ ] Initialize indexes in AstrBot's plugin data directory and close SQLite/Chroma resources on termination.
- [ ] Remove the obsolete `RetrievedChunk`, `KnowledgeIndex`, and grounded-user-prompt implementation.
- [ ] Run all tests.

### Task 6: Document, validate, review, and publish

**Files:**
- Modify: `README.md`
- Modify: `metadata.yaml`

- [ ] Document Agent-selected tools, two-stage evidence flow, local index lifecycle, Embedding Provider configuration, keyword fallback, and index storage.
- [ ] Bump plugin version and run:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q main.py nova_cac tests
ruff check .
git diff --check
```

- [ ] Request code review and fix every Critical or Important issue.
- [ ] Commit, fast-forward `main`, and push the same commit to `main` and `master`.
