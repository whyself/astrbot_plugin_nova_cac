from __future__ import annotations

import asyncio
import importlib.util
import logging
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _install_astrbot_shim() -> None:
    if "astrbot.api.star" in sys.modules:
        return

    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    event = types.ModuleType("astrbot.api.event")
    star = types.ModuleType("astrbot.api.star")
    components = types.ModuleType("astrbot.api.message_components")
    core = types.ModuleType("astrbot.core")
    agent = types.ModuleType("astrbot.core.agent")
    tool = types.ModuleType("astrbot.core.agent.tool")

    class Plain:
        def __init__(self, text: str):
            self.text = text

    class FunctionTool:
        pass

    class AstrMessageEvent:
        pass

    class Filter:
        @staticmethod
        def regex(_pattern):
            return lambda function: function

    class Context:
        pass

    class Star:
        def __init__(self, context):
            self.context = context

    class StarTools:
        _data_dir = Path(tempfile.gettempdir()) / "nova-cac-plugin-test"

        @classmethod
        def get_data_dir(cls, _plugin_name=None):
            return cls._data_dir

    class ToolSet:
        def __init__(self, tools):
            self.tools = list(tools)

    def register(*_args, **_kwargs):
        return lambda cls: cls

    components.Plain = Plain
    api.FunctionTool = FunctionTool
    api.logger = logging.getLogger("astrbot-test")
    api.message_components = components
    event.AstrMessageEvent = AstrMessageEvent
    event.filter = Filter()
    star.Context = Context
    star.Star = Star
    star.StarTools = StarTools
    star.register = register
    tool.ToolSet = ToolSet

    astrbot.api = api
    astrbot.core = core
    core.agent = agent
    agent.tool = tool
    sys.modules.update(
        {
            "astrbot": astrbot,
            "astrbot.api": api,
            "astrbot.api.event": event,
            "astrbot.api.star": star,
            "astrbot.api.message_components": components,
            "astrbot.core": core,
            "astrbot.core.agent": agent,
            "astrbot.core.agent.tool": tool,
        }
    )


def _load_plugin_class():
    _install_astrbot_shim()
    package_name = "_nova_cac_plugin_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT)]
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.main",
        ROOT / "main.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.NovaCacPlugin


class FakeContext:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir


class FakeEvent:
    def __init__(self, text: str):
        plain = sys.modules["astrbot.api.message_components"].Plain(text)
        self.message_obj = types.SimpleNamespace(
            message=[plain],
            message_str=text,
        )
        self.unified_msg_origin = "group:test"
        self.stopped = False

    def stop_event(self):
        self.stopped = True

    def plain_result(self, text):
        return text

    def is_private_chat(self):
        return False

    def get_sender_id(self):
        return "sender"

    def get_group_id(self):
        return "group"


async def _collect(generator):
    return [item async for item in generator]


class MainIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plugin_class = _load_plugin_class()

    def test_plugin_indexes_embedded_pack_without_embedding_provider(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sys.modules["astrbot.api.star"].StarTools._data_dir = Path(temp_dir)
            plugin = self.plugin_class(
                FakeContext(Path(temp_dir)),
                {"enable_vector_search": False},
            )
            asyncio.run(plugin.initialize())
            self.assertGreater(plugin.index.document_count(), 0)
            self.assertGreater(plugin.chunk_store.chunk_count(), 0)
            asyncio.run(plugin.terminate())

    def test_only_literal_slash_command_enters_plugin_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sys.modules["astrbot.api.star"].StarTools._data_dir = Path(temp_dir)
            plugin = self.plugin_class(
                FakeContext(Path(temp_dir)),
                {"enable_vector_search": False},
            )

            async def answer(_event, query, **_kwargs):
                return f"回答：{query}"

            plugin.agent.answer = answer
            event = FakeEvent("/cac NOVA 是什么")
            output = asyncio.run(_collect(plugin.cac(event)))
            self.assertEqual(output, ["回答：NOVA 是什么"])
            self.assertTrue(event.stopped)
            self.assertEqual(len(plugin.memory.contexts("group:test")), 2)

            ignored = FakeEvent("cac 不带斜杠")
            output = asyncio.run(_collect(plugin.cac(ignored)))
            self.assertEqual(output, [])
            self.assertFalse(ignored.stopped)
            self.assertEqual(len(plugin.memory.contexts("group:test")), 2)

            async def failed_answer(_event, _query, **_kwargs):
                module = sys.modules["_nova_cac_plugin_test.main"]
                return module.AGENT_ERROR

            plugin.agent.answer = failed_answer
            asyncio.run(_collect(plugin.cac(FakeEvent("/cac 无证据问题"))))
            self.assertEqual(len(plugin.memory.contexts("group:test")), 2)
            asyncio.run(plugin.terminate())

    def test_source_list_requires_explicit_source_intent(self):
        asks = self.plugin_class._asks_for_sources
        for query in (
            "这段话的来源是什么",
            "给出参考资料",
            "引用依据呢",
            "请给出依据",
            "请给出证据",
            "URL",
            "把原文链接发我",
        ):
            with self.subTest(query=query):
                self.assertTrue(asks(query))

        for query in (
            "如何链接 GitHub",
            "把这两个概念链接起来",
            "NOVA 文献综述活动是什么？",
            "如何阅读引用内容",
            "NOVA 是什么",
        ):
            with self.subTest(query=query):
                self.assertFalse(asks(query))

if __name__ == "__main__":
    unittest.main()
