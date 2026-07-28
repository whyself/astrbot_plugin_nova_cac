# NOVA CAC AstrBot Plugin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and publish an AstrBot plugin that answers NOVA questions through `/cac`, always applies the four-file knowledge-pack foundation, retrieves relevant local material, and preserves recent chat context.

**Architecture:** Keep AstrBot-specific event handling in `main.py` and framework-independent behavior in the `nova_cac` package. `PackLoader` rereads `AGENTS.md`, `soul.md`, `spirit.md`, and `voice.md` for every answer; `KnowledgeIndex` supplies only the highest-scoring Markdown sections; `ConversationMemory` maintains bounded per-session turns. The plugin accepts private `/cac` commands and group `/cac` commands only when the bot is explicitly mentioned.

**Tech Stack:** Python 3.10+, AstrBot plugin API, standard-library Markdown parsing and retrieval, `unittest`, Ruff.

---

## File map

- `main.py`: AstrBot registration, `/cac` event routing, provider invocation, and user-visible errors.
- `nova_cac/core.py`: core-file loading, Markdown indexing/retrieval, prompt assembly, and bounded conversation memory.
- `nova_cac/routing.py`: framework-independent command and mention decisions.
- `knowledge_pack/`: embedded authoritative NOVA pack.
- `_conf_schema.json`: configurable history and retrieval limits.
- `metadata.yaml`: AstrBot plugin metadata and compatibility.
- `tests/`: regression tests for mandatory reads, retrieval, memory, and routing.
- `README.md`: installation, trigger behavior, context behavior, commands, configuration, and privacy notes.

### Task 1: Scaffold and embed the knowledge pack

**Files:**
- Create: `.gitignore`
- Create: `metadata.yaml`
- Create: `_conf_schema.json`
- Create: `requirements.txt`
- Create: `nova_cac/__init__.py`
- Copy: `../nova-knowledge-pack/**` to `knowledge_pack/**`

- [ ] **Step 1: Create plugin metadata and configuration**

Declare the plugin as `astrbot_plugin_nova_cac`, require AstrBot `>=4.16,<5`, and expose `history_turns`, `history_max_chars`, `retrieval_top_k`, and `max_context_chars`.

- [ ] **Step 2: Copy the curated pack without rewriting it**

Run:

```powershell
Copy-Item -Recurse -Force ..\nova-knowledge-pack\* .\knowledge_pack\
```

Expected: `knowledge_pack` contains four root Markdown files and fourteen files below `knowledge/`.

- [ ] **Step 3: Initialize a feature branch**

Run:

```powershell
git init
git switch -c feat/nova-cac-plugin
```

Expected: the current branch is `feat/nova-cac-plugin`.

### Task 2: Implement and test framework-independent core behavior

**Files:**
- Create: `tests/test_core.py`
- Create: `tests/test_routing.py`
- Create: `nova_cac/core.py`
- Create: `nova_cac/routing.py`

- [ ] **Step 1: Write failing tests**

Cover these behaviors:

```python
def test_core_files_are_reread_for_every_prompt(): ...
def test_retrieval_prefers_current_rules_for_membership_question(): ...
def test_memory_is_bounded_and_can_be_cleared(): ...
def test_private_requires_cac_command(): ...
def test_group_requires_cac_command_and_bot_mention(): ...
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
python -m unittest discover -s tests -v
```

Expected: imports fail because the core modules do not exist yet.

- [ ] **Step 3: Implement the minimum core**

`PackLoader.build_system_prompt()` calls `Path.read_text(encoding="utf-8")` for all four core files on every invocation. `KnowledgeIndex` parses YAML-like front matter and Markdown heading sections, creates CJK bigram plus ASCII-word tokens, and returns the top relevant chunks. `ConversationMemory` uses bounded deques and a bounded session map. Routing helpers require a normalized `/cac` command, plus a bot mention in groups.

- [ ] **Step 4: Run tests and confirm success**

Run:

```powershell
python -m unittest discover -s tests -v
```

Expected: all tests pass.

### Task 3: Integrate with AstrBot

**Files:**
- Create: `main.py`

- [ ] **Step 1: Register the command**

Use `@register` with a regex filter that can match AstrBot's preprocessed command text, then validate the original plain message chain with the strict `/cac` parser. Private messages are accepted directly. Group messages silently stop unless the original command is `/cac` and the message chain includes an `At` component matching `event.get_self_id()`.

- [ ] **Step 2: Add command handling**

Implement:

```text
/cac <问题>    answer with retrieved knowledge and recent context
/cac help      show usage
/cac reset     clear this conversation's plugin-managed context
```

- [ ] **Step 3: Call the active AstrBot provider**

Resolve the provider through `get_current_chat_provider_id`, pass recent turns as `contexts`, pass the freshly assembled four-file prompt as `system_prompt`, and append the successful question/answer pair to memory. Serialize reset, context reads, generation, and history writes with a per-session async lock.

- [ ] **Step 4: Handle failures without leaking internals**

Return concise messages for an empty provider response or provider exception, log the detailed exception, and do not append failed calls to history.

### Task 4: Document and validate

**Files:**
- Create: `README.md`
- Create: `LICENSE`

- [ ] **Step 1: Document install and operation**

Explain that `AGENTS.md` is not automatically discovered by AstrBot: this plugin explicitly rereads all four core files for every answer. Document exact group/private triggers, `/cac reset`, in-memory context scope, configuration, updating the embedded pack, and data/privacy implications.

- [ ] **Step 2: Run verification**

Run:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q main.py nova_cac tests
ruff check .
```

Expected: tests pass, compilation succeeds, and Ruff reports no errors.

- [ ] **Step 3: Inspect repository scope**

Run:

```powershell
git status --short
git diff --check
```

Expected: only intended plugin files appear and no whitespace errors are reported.

### Task 5: Publish to GitHub

**Files:**
- Modify: Git history and remote configuration only.

- [ ] **Step 1: Create an intentional commit**

Run:

```powershell
git add .
git commit -m "feat: add NOVA CAC knowledge Q&A plugin"
```

Expected: one root commit containing the plugin, tests, docs, and embedded curated pack.

- [ ] **Step 2: Create a private GitHub repository and push**

Run:

```powershell
gh repo create whyself/astrbot_plugin_nova_cac --private --source . --remote origin --push
```

Expected: the repository exists under `whyself`, and `feat/nova-cac-plugin` is pushed.

- [ ] **Step 3: Verify remote state**

Run:

```powershell
git status --short --branch
gh repo view whyself/astrbot_plugin_nova_cac --json url,visibility,defaultBranchRef
```

Expected: the worktree is clean and the repository URL, visibility, and default branch are confirmed.
