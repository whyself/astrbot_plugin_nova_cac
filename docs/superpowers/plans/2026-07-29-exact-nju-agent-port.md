# Exact NJU Agent Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the simplified NOVA Agent implementation with a source-level port of `Gu-Heping/astrbot_plugin_nju_qa` commit `275d8b8d17aa86615fbb39f4a23291d32bed3643`.

**Architecture:** Copy the reference Agent, evidence, retrieval, chunk/index, document utility, and tool modules as one coherent subsystem. Replace Yuque synchronization with a deterministic adapter that mirrors embedded `knowledge_pack/knowledge` Markdown into the reference `DocumentStore`/`DocumentIndex`; retain the already-approved `/cac` trigger, Pack prompt, bounded `/cac` context, and local-source-only behavior.

**Tech Stack:** Python 3.10+, AstrBot `tool_loop_agent` and `FunctionTool`, SQLite, ChromaDB, OpenAI-compatible `/embeddings`, httpx, PyYAML, unittest, Ruff.

---

### Task 1: Establish source-level conformance

**Files:**
- Replace: `nova_cac/agent.py`
- Replace: `nova_cac/evidence.py`
- Replace: `nova_cac/prompts.py`
- Create: `nova_cac/doc_utils.py`
- Create: `nova_cac/formatting.py`
- Test: `tests/test_reference_conformance.py`

- [ ] Copy the reference files without simplifying their control flow, constants, evidence merging, version classification, retry behavior, URL verification, or research/answer tool separation.
- [ ] Rename only the public Agent class and domain-facing NJU wording required by NOVA.
- [ ] Add a conformance manifest recording the reference commit and adapted lines.
- [ ] Run the Agent and evidence tests copied or adapted from the reference repository.

### Task 2: Port the complete retrieval subsystem

**Files:**
- Replace: `nova_cac/models.py`
- Replace: `nova_cac/chunking.py`
- Replace: `nova_cac/chunk_store.py`
- Replace: `nova_cac/keyword_index.py`
- Replace: `nova_cac/vector_index.py`
- Replace: `nova_cac/retriever.py`
- Create: `nova_cac/config.py`
- Create: `nova_cac/document_index.py`
- Create: `nova_cac/document_store.py`
- Create: `nova_cac/knowledge_structure.py`
- Create: `nova_cac/chunk_indexer.py`

- [ ] Copy the reference types, chunk construction, BM25 scoring, hybrid merging, diversity, reliability thresholds, and Chroma lifecycle.
- [ ] Use the same OpenAI-compatible `embedding_api_key`, `embedding_base_url`, and `embedding_model` configuration and `/embeddings` request.
- [ ] Preserve the reference fallback and diagnostic behavior.
- [ ] Run retrieval, chunking, and persistence tests.

### Task 3: Port every research tool

**Files:**
- Create: `nova_cac/knowledge_tools.py`
- Create: `nova_cac/tools/__init__.py`
- Create: `nova_cac/tools/knowledge.py`
- Create: `nova_cac/tools/documents.py`

- [ ] Port search, grep, read, metadata, URL parsing, knowledge-base listing, repository listing/tree, outline, and stats tools.
- [ ] Preserve exact AstrBot `FunctionTool.call`/`run` contracts and `SourceTracker` side effects.
- [ ] Adapt only names and messages that refer to NJU or Yuque-specific administration.
- [ ] Run the complete tool test set.

### Task 4: Replace Yuque with the local Pack adapter

**Files:**
- Create: `nova_cac/local_sync.py`
- Modify: `main.py`
- Modify: `nova_cac/core.py`

- [ ] Scan embedded Markdown and parse its front matter into reference `Document` objects with stable IDs.
- [ ] Mirror documents into the managed AstrBot plugin data directory using `DocumentStore` and `DocumentIndex`.
- [ ] Delete stale mirrored documents and rebuild keyword/vector chunks with the same reference indexers.
- [ ] Inject freshly read `AGENTS.md`, `soul.md`, `spirit.md`, and `voice.md` plus `/cac` contexts into both Agent stages without changing reference research/evidence logic.
- [ ] Keep literal `/cac`, group/private parity, help/reset, locks, and successful-answer-only history unchanged.

### Task 5: Validate parity and publish

**Files:**
- Modify: `_conf_schema.json`
- Modify: `README.md`
- Modify: `metadata.yaml`
- Modify: `requirements.txt`

- [ ] Remove AstrBot Embedding Provider configuration and document the reference-compatible OpenAI embedding fields.
- [ ] Run `python -m unittest discover -s tests -v`.
- [ ] Run `python -m compileall -q main.py nova_cac tests`, `ruff check .`, and `git diff --check`.
- [ ] Request an independent code review comparing the port against reference commit `275d8b8d`.
- [ ] Commit, fast-forward `main`, and push the same commit to `main` and `master`.
