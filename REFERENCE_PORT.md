# Reference port manifest

Upstream: `Gu-Heping/astrbot_plugin_nju_qa`  
Commit: `275d8b8d17aa86615fbb39f4a23291d32bed3643`

The following modules were copied as one coherent subsystem instead of being
rewritten or simplified:

- Agent and evidence: `agent.py`, `evidence.py`, `prompts.py`, `doc_utils.py`, `formatting.py`
- Retrieval: `models.py`, `config.py`, `retriever.py`, `keyword_index.py`, `knowledge_structure.py`
- Persistence: `document_index.py`, `document_store.py`, `chunking.py`, `chunk_store.py`, `chunk_indexer.py`, `vector_index.py`
- Agent tools: `knowledge_tools.py`, `tools/knowledge.py`, `tools/documents.py`

Intentional adapter changes:

1. NJU-facing wording is replaced with NOVA-facing wording.
2. Yuque synchronization is replaced by `local_sync.py`, which mirrors the
   embedded `knowledge_pack/knowledge` Markdown into the same document and
   chunk interfaces consumed by the upstream Agent.
3. The public AstrBot adapter retains literal `/cac`, Pack-mandatory reads,
   `/cac`-only bounded history, help/reset, and per-session serialization.
4. The upstream strict two-stage class remains available for component parity,
   but `/cac` uses `NovaCacAgent`, a single-pass AstrBot Agent wrapper around
   the same retrieval tools. It does not require evidence markers or force the
   upstream no-evidence sentence.
5. The freshly read Pack prompt and recent `/cac` contexts are passed into that
   direct Agent call.
6. Titles, document paths, source URLs, and citation markers stay internal;
   the final `/cac` answer never appends a source section.

The upstream test modules for Agent runtime, chunking, hybrid retrieval,
document tools, evidence/version handling, scoped retrieval, and structure
tools are included under `tests/` and run against the port.
