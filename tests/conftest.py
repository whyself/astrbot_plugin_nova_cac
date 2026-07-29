"""Provide a minimal astrbot shim so tests can import the plugin code."""

from __future__ import annotations

import importlib.util
import logging
import math
import sys
import tempfile
import types
from collections.abc import Callable
from pathlib import Path

import pytest


def _install_chromadb_shim() -> None:
    if importlib.util.find_spec("chromadb") is not None:
        return
    module = types.ModuleType("chromadb")
    databases: dict[str, dict[str, object]] = {}

    class Collection:
        def __init__(self, metadata=None):
            self.metadata = metadata or {}
            self.rows: dict[str, tuple[str, list[float], dict]] = {}

        def count(self):
            return len(self.rows)

        def upsert(self, ids, documents, embeddings, metadatas):
            for key, document, embedding, metadata in zip(
                ids, documents, embeddings, metadatas
            ):
                self.rows[key] = (document, list(embedding), dict(metadata))

        def delete(self, where=None):
            if not where:
                self.rows.clear()
                return
            document_id = where.get("document_id")
            doomed = [
                key
                for key, (_, _, metadata) in self.rows.items()
                if metadata.get("document_id") == document_id
            ]
            for key in doomed:
                self.rows.pop(key, None)

        def query(self, query_embeddings, n_results, include, where=None):
            query = query_embeddings[0]
            allowed = None
            if where and isinstance(where.get("document_id"), dict):
                allowed = set(where["document_id"].get("$in", []))
            scored = []
            for row_id, (document, embedding, metadata) in self.rows.items():
                if allowed is not None and metadata.get("document_id") not in allowed:
                    continue
                denominator = math.sqrt(sum(x * x for x in query)) * math.sqrt(
                    sum(x * x for x in embedding)
                )
                similarity = (
                    sum(a * b for a, b in zip(query, embedding)) / denominator
                    if denominator
                    else 0.0
                )
                scored.append((1.0 - similarity, row_id, document, metadata))
            scored.sort(key=lambda item: item[0])
            selected = scored[:n_results]
            return {
                "ids": [[item[1] for item in selected]],
                "documents": [[item[2] for item in selected]],
                "metadatas": [[item[3] for item in selected]],
                "distances": [[item[0] for item in selected]],
            }

    class PersistentClient:
        def __init__(self, path):
            self.collections = databases.setdefault(str(path), {})

        def get_collection(self, name):
            if name not in self.collections:
                raise ValueError("collection not found")
            return self.collections[name]

        def create_collection(self, name, metadata=None):
            collection = Collection(metadata)
            self.collections[name] = collection
            return collection

        def delete_collection(self, name):
            if name not in self.collections:
                raise ValueError("collection not found")
            del self.collections[name]

    module.PersistentClient = PersistentClient
    sys.modules["chromadb"] = module


def _make_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def _install_astrbot_shim() -> None:
    if "astrbot" in sys.modules:
        return

    astrbot_pkg = _make_module("astrbot")
    api = _make_module("astrbot.api")
    astrbot_pkg.api = api

    # Logger used across the plugin.
    api.logger = logging.getLogger("astrbot")

    # Tool base class used by nju_qa.tools.
    class FunctionTool:
        def __init__(self, *args, **kwargs) -> None:
            pass

    api.FunctionTool = FunctionTool

    # message_components stubs used by main.py and retriever formatting.
    mc = _make_module("astrbot.api.message_components")

    class Plain:
        def __init__(self, text: str = "") -> None:
            self.text = text

    class At:
        def __init__(self, qq: str = "") -> None:
            self.qq = qq

    class Image:
        @staticmethod
        def fromFileSystem(path: str):  # noqa: N802
            return Image(path)

        def __init__(self, path: str = "") -> None:
            self.path = path

    mc.Plain = Plain
    mc.At = At
    mc.Image = Image
    api.message_components = mc

    # star stubs used for plugin registration.
    star = _make_module("astrbot.api.star")

    class Context:
        pass

    class Star:
        def __init__(self, context: object) -> None:
            self.context = context

    class StarTools:
        _data_dir = Path(tempfile.mkdtemp(prefix="nova-cac-tests-"))

        @classmethod
        def get_data_dir(cls, plugin_name: str | None = None) -> Path:
            return cls._data_dir

    def register(*args, **kwargs) -> Callable[[type], type]:
        def decorator(cls: type) -> type:
            return cls

        return decorator

    star.Context = Context
    star.Star = Star
    star.StarTools = StarTools
    star.register = register
    api.star = star

    # event/filter stubs used by command handlers.
    event_mod = _make_module("astrbot.api.event")

    class PermissionType:
        ADMIN = "admin"

    class EventMessageType:
        ALL = "all"

    class _Filter:
        @staticmethod
        def regex(pattern: str = "") -> Callable[[Callable], Callable]:
            def decorator(func: Callable) -> Callable:
                return func

            return decorator

        @staticmethod
        def command(name: str = "") -> Callable[[Callable], Callable]:
            def decorator(func: Callable) -> Callable:
                return func

            return decorator

        @staticmethod
        def permission_type(kind: object) -> Callable[[Callable], Callable]:
            def decorator(func: Callable) -> Callable:
                return func

            return decorator

        @staticmethod
        def event_message_type(kind: object) -> Callable[[Callable], Callable]:
            def decorator(func: Callable) -> Callable:
                return func

            return decorator

    event_mod.filter = _Filter()
    event_mod.PermissionType = PermissionType
    event_mod.EventMessageType = EventMessageType
    _Filter.PermissionType = PermissionType
    _Filter.EventMessageType = EventMessageType
    api.event = event_mod

    class AstrMessageEvent:
        def __init__(self) -> None:
            self.message_str = ""
            self.message_obj = types.SimpleNamespace(message=[])

    event_mod.AstrMessageEvent = AstrMessageEvent

    # core.agent.run_context stub used by nju_qa.tools.
    core = _make_module("astrbot.core")
    astrbot_pkg.core = core
    agent = _make_module("astrbot.core.agent")
    core.agent = agent
    run_context = _make_module("astrbot.core.agent.run_context")
    agent.run_context = run_context
    tool_mod = _make_module("astrbot.core.agent.tool")
    agent.tool = tool_mod

    class ContextWrapper:
        def __init__(self, *args, **kwargs) -> None:
            pass

    run_context.ContextWrapper = ContextWrapper

    class ToolSet:
        def __init__(self, tools=None):
            self.tools = list(tools) if tools is not None else []

    tool_mod.ToolSet = ToolSet

    # core.astr_agent_context stub used by nju_qa.tools.
    astr_agent_context = _make_module("astrbot.core.astr_agent_context")
    core.astr_agent_context = astr_agent_context

    class AstrAgentContext:
        def __init__(self, *args, **kwargs) -> None:
            pass

    astr_agent_context.AstrAgentContext = AstrAgentContext


_install_chromadb_shim()
_install_astrbot_shim()

# ---------------------------------------------------------------------------
# Load the plugin entry point once before test modules import submodules.
#
# main.py uses package-relative imports.  Loading it here populates nju_qa's
# attributes and lets us alias the submodule objects under their plain
# ``nju_qa.*`` names, so monkeypatch-based tests operate on the same module
# objects the plugin code uses.
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

import nova_cac as nju_qa  # noqa: E402

_pkg = types.ModuleType("astrbot_plugin_nova_cac")
_pkg.__path__ = [str(ROOT)]
sys.modules["astrbot_plugin_nova_cac"] = _pkg
sys.modules["astrbot_plugin_nova_cac.nova_cac"] = nju_qa
sys.modules["nju_qa"] = nju_qa

_main_spec = importlib.util.spec_from_file_location(
    "astrbot_plugin_nova_cac.main", ROOT / "main.py"
)
_main_mod = importlib.util.module_from_spec(_main_spec)
sys.modules["astrbot_plugin_nova_cac.main"] = _main_mod
_main_spec.loader.exec_module(_main_mod)

for _attr_name, _attr in list(nju_qa.__dict__.items()):
    if isinstance(_attr, types.ModuleType) and not _attr_name.startswith("_"):
        sys.modules.setdefault(f"nju_qa.{_attr_name}", _attr)


# ---------------------------------------------------------------------------
# Shared fixtures for integration tests that need to load main.py.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def plugin_class():
    """Return the plugin class loaded at conftest import time."""
    return sys.modules["astrbot_plugin_nova_cac.main"].NovaCacPlugin
