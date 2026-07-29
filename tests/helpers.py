"""Shared helpers for plugin integration tests."""

from __future__ import annotations

import types


class _FakeAgent:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def answer(self, event, prompt: str) -> str:
        self.calls.append(prompt)
        return f"答案：{prompt}"


class _FakeEvent:
    def __init__(
        self,
        group_id=None,
        sender_id="u1",
        text="nju test",
        raw_text=None,
        is_command=False,
        is_at_or_wake_command=False,
        is_wake=False,
        stopped=False,
        message_chain=None,
    ):
        self.group_id = group_id
        self.sender_id = sender_id
        self.message_str = text
        self.raw_text = raw_text if raw_text is not None else text
        self.unified_msg_origin = "test:session"
        self.is_command = is_command
        self.is_at_or_wake_command = is_at_or_wake_command
        self.is_wake = is_wake
        self.stopped = stopped
        self._result = None
        self._extras: dict[str, object] = {}
        self.call_llm = True
        self.should_call_llm_calls: list[bool] = []
        if message_chain is None:
            from astrbot.api import message_components as comp

            message_chain = [comp.Plain(self.raw_text)]
        self.message_obj = types.SimpleNamespace(message=message_chain)

    def get_extra(self, key: str, default=None):
        return self._extras.get(key, default)

    def set_extra(self, key: str, value: object) -> None:
        self._extras[key] = value

    def should_call_llm(self, flag: bool) -> None:
        self.should_call_llm_calls.append(flag)
        self.call_llm = flag

    def is_private_chat(self):
        return self.group_id in (None, "")

    def get_group_id(self):
        return self.group_id

    def get_sender_id(self):
        return self.sender_id

    def get_self_id(self):
        return "bot"

    def is_stopped(self):
        return self.stopped

    def stop_event(self):
        self.stopped = True

    def plain_result(self, text: str):
        return text

    def chain_result(self, chain: list):
        return chain

    def get_result(self):
        return self._result

    def set_result(self, result):
        self._result = result

    def clear_result(self):
        self._result = None


def _make_plugin(plugin_class, tmp_path, group_limit=2, private_limit=2, **config_overrides):
    class TestPlugin(plugin_class):
        def __init__(self, data_dir, config):
            self.data_dir = data_dir
            super().__init__(context=object(), config=config)
            self.agent = _FakeAgent()

    config = {
        "yuque_token": "x",
        "yuque_repositories": ["nju/guide"],
        "group_rate_limit": group_limit,
        "group_rate_limit_window": 3600,
        "private_rate_limit": private_limit,
        "private_rate_limit_window": 3600,
    }
    config.update(config_overrides)
    return TestPlugin(tmp_path, config)


async def _collect(generator):
    return [item async for item in generator]
