"""Bot-to-bot @mention routing for the Feishu adapter (registry, @ conversion, router)."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from tests.gateway.feishu_helpers import (
    make_adapter_skeleton,
    make_message,
    make_sender,
)

PEERS = {
    "bot_aliases": ["架构师Alex", "Alex"],
    "mention_map": {"架构师alex": "ou_arch", "架构师": "ou_arch", "alex": "ou_arch"},
    "profile_by_openid": {"ou_arch": "arch"},
    "profile_by_name": {"架构师alex": "arch", "架构师": "arch", "alex": "arch"},
    "canonical_by_profile": {"arch": "架构师Alex"},
}


def _with_registry(adapter, **overrides):
    adapter._bot_aliases = list(PEERS["bot_aliases"])
    adapter._peer_mention_map = dict(PEERS["mention_map"])
    adapter._peer_profile_by_openid = dict(PEERS["profile_by_openid"])
    adapter._peer_profile_by_name = dict(PEERS["profile_by_name"])
    adapter._peer_canonical_by_profile = dict(PEERS["canonical_by_profile"])
    for key, value in overrides.items():
        setattr(adapter, key, value)
    return adapter


# --- Inbound: text alias passes the require_mention gate --------------------


def test_text_alias_passes_mention_gate():
    from plugins.platforms.feishu.adapter import FeishuAdapter

    adapter = _with_registry(make_adapter_skeleton(bot_open_id="ou_me"))
    message = SimpleNamespace(
        message_id="om_1", message_type="text",
        content=json.dumps({"text": "@Alex 你好"}, ensure_ascii=False), mentions=[],
    )

    assert adapter._text_mentions_self(message.content) is True
    assert adapter._mentions_self(message) is True


def test_plain_text_without_alias_does_not_mention_self():
    adapter = _with_registry(make_adapter_skeleton(bot_open_id="ou_me"))
    content = json.dumps({"text": "@某个路人 你好"}, ensure_ascii=False)

    assert adapter._text_mentions_self(content) is False


def test_alias_mention_admitted_in_require_mention_group():
    adapter = _with_registry(
        make_adapter_skeleton(bot_open_id="ou_me", allow_bots="all",
                              group_policy="open", require_mention=True),
    )
    message = make_message(chat_type="group", chat_id="oc_g")
    message.content = json.dumps({"text": "@架构师Alex 帮忙看下"}, ensure_ascii=False)

    assert adapter._admit(make_sender(sender_type="bot", open_id="ou_peer"), message) is None


# --- Outbound: @ conversion into post <at> elements ------------------------


def test_inject_peer_mentions_builds_post_payload():
    from plugins.platforms.feishu.adapter import FeishuAdapter

    adapter = _with_registry(object.__new__(FeishuAdapter))
    payload = adapter._inject_peer_mentions("请 @架构师 看一下这个方案")

    assert payload is not None
    rows = json.loads(payload)["zh_cn"]["content"]
    at_elements = [el for row in rows for el in row if el.get("tag") == "at"]
    assert at_elements == [{"tag": "at", "user_id": "ou_arch", "user_name": "架构师"}]
    assert "看一下这个方案" in json.dumps(rows, ensure_ascii=False)


def test_inject_peer_mentions_skips_fenced_code():
    from plugins.platforms.feishu.adapter import FeishuAdapter

    adapter = _with_registry(object.__new__(FeishuAdapter))
    payload = adapter._inject_peer_mentions("```\n@架构师\n```\n@架构师 你好")

    rows = json.loads(payload)["zh_cn"]["content"]
    assert sum(1 for row in rows for el in row if el.get("tag") == "at") == 1


def test_inject_peer_mentions_returns_none_without_peers():
    from plugins.platforms.feishu.adapter import FeishuAdapter

    adapter = object.__new__(FeishuAdapter)
    adapter._peer_mention_map = {}

    assert adapter._inject_peer_mentions("@任何人 你好") is None


def test_parse_peer_targets_resolves_prefix_to_profile():
    from plugins.platforms.feishu.adapter import FeishuAdapter

    adapter = _with_registry(
        object.__new__(FeishuAdapter),
        _peer_mention_map={"架构师": "ou_arch"},
    )

    assert adapter._parse_peer_mention_targets("@架构师Alex 请接力") == [("arch", "架构师")]


# --- Gateway router --------------------------------------------------------


def _feishu_adapter(*, name: str, open_id: str, router_capture=None):
    from plugins.platforms.feishu.adapter import FeishuAdapter

    adapter = object.__new__(FeishuAdapter)
    adapter.platform = Platform.FEISHU
    adapter._bot_name = name
    adapter._bot_open_id = open_id

    async def _capture(event):
        if router_capture is not None:
            router_capture.append(event)

    adapter._handle_message_with_guards = _capture
    return adapter


def _runner(sender, profiles):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {Platform.FEISHU: sender}
    runner._profile_adapters = {
        name: {Platform.FEISHU: adapter} for name, adapter in profiles.items()
    }
    runner._primary_profile_name = "default"
    return runner


@pytest.mark.asyncio
async def test_route_bot_to_bot_injects_into_peer_and_sets_reply_profile():
    captured: list[MessageEvent] = []
    sender = _feishu_adapter(name="Tony", open_id="ou_sender")
    target = _feishu_adapter(name="Pete", open_id="ou_target", router_capture=captured)
    runner = _runner(sender, {"pm": target})

    await runner._route_bot_to_bot(
        [("pm", "Pete")], "hello @Pete", "oc_chat", sender,
        hop=0, thread_id="omt_topic", anchor_message_id="om_sender_msg",
    )

    assert len(captured) == 1
    event = captured[0]
    assert event.text == "[Tony → @Pete]: hello @Pete"
    assert event.source.chat_id == "oc_chat"
    assert event.source.thread_id == "omt_topic"
    assert event.reply_to_message_id == "om_sender_msg"
    assert event.metadata["hop"] == 1
    assert target._bot_to_bot_reply_profile == ("default", 1)


@pytest.mark.asyncio
async def test_route_bot_to_bot_skips_sender_profile_and_hop_limit():
    captured: list[MessageEvent] = []
    sender = _feishu_adapter(name="Tony", open_id="ou_sender")
    target = _feishu_adapter(name="Pete", open_id="ou_target", router_capture=captured)
    runner = _runner(sender, {"pm": target})

    await runner._route_bot_to_bot([("default", "Tony")], "self", "oc_chat", sender)
    await runner._route_bot_to_bot([("pm", "Pete")], "capped", "oc_chat", sender,
                                   hop=runner._BOT_TO_BOT_HOP_LIMIT)

    assert captured == []


# --- Reply back-route is one-shot and carries the group thread ------------


@pytest.mark.asyncio
async def test_reply_profile_routes_back_once_with_hop():
    from plugins.platforms.feishu.adapter import FeishuAdapter
    from unittest.mock import AsyncMock

    adapter = _with_registry(object.__new__(FeishuAdapter))
    adapter._bot_to_bot_reply_profile = ("sender-profile", 1)
    adapter.get_chat_info = AsyncMock(return_value={"type": "group"})
    routed: list[tuple] = []

    async def router(targets, content, chat_id, sender_adapter, **kwargs):
        routed.append((targets, chat_id, kwargs))

    adapter._bot_to_bot_router = router
    result = SendResult(success=True, message_id="om_reply")

    await adapter._route_peer_mentions(result, "oc_g", "收到", {"thread_id": "omt_1"})
    await asyncio.sleep(0)

    assert adapter._bot_to_bot_reply_profile is None
    assert len(routed) == 1
    targets, chat_id, kwargs = routed[0]
    assert targets == [("sender-profile", "sender-profile")]
    assert chat_id == "oc_g"
    assert kwargs == {"hop": 1, "thread_id": "omt_1", "anchor_message_id": "om_reply"}


@pytest.mark.asyncio
async def test_reply_profile_not_routed_out_of_dm():
    from plugins.platforms.feishu.adapter import FeishuAdapter
    from unittest.mock import AsyncMock

    adapter = _with_registry(object.__new__(FeishuAdapter))
    adapter._bot_to_bot_reply_profile = ("sender-profile", 1)
    adapter.get_chat_info = AsyncMock(return_value={"type": "dm"})
    routed: list[tuple] = []

    async def router(*args, **kwargs):
        routed.append((args, kwargs))

    adapter._bot_to_bot_router = router

    await adapter._route_peer_mentions(
        SendResult(success=True, message_id="om_reply"), "oc_dm", "收到", None,
    )

    assert routed == []
    assert adapter._bot_to_bot_reply_profile is None


# --- Loop budget -----------------------------------------------------------


def test_bot_loop_budget_drops_after_limit():
    adapter = make_adapter_skeleton(bot_open_id="ou_me", allow_bots="all", group_policy="open")
    adapter._peer_mention_map = {"arch": "ou_arch"}
    adapter._bot_interaction_history = {}
    adapter._BOT_INTERACTION_LIMIT = 2
    adapter._BOT_INTERACTION_WINDOW = 60
    adapter._mentions_self = lambda _message: True
    sender = make_sender(sender_type="bot", open_id="ou_peer")

    reasons = [adapter._admit(sender, make_message(chat_type="group", chat_id="oc_g")) for _ in range(3)]

    assert reasons == [None, None, "bot_loop_prevented"]


def test_loop_budget_inactive_without_peer_registry():
    adapter = make_adapter_skeleton(bot_open_id="ou_me", allow_bots="all", group_policy="open")
    adapter._peer_mention_map = {}
    adapter._bot_interaction_history = {}
    adapter._BOT_INTERACTION_LIMIT = 1
    adapter._BOT_INTERACTION_WINDOW = 60
    adapter._mentions_self = lambda _message: True
    sender = make_sender(sender_type="bot", open_id="ou_peer")

    reasons = [adapter._admit(sender, make_message(chat_type="group", chat_id="oc_g")) for _ in range(3)]

    assert reasons == [None, None, None]


# --- Missing registry = no-op ---------------------------------------------


def test_missing_registry_leaves_every_new_path_disabled(tmp_path, monkeypatch):
    import plugins.platforms.feishu.adapter as feishu_mod
    from plugins.platforms.feishu.adapter import FeishuAdapter

    monkeypatch.setattr(feishu_mod, "get_hermes_home", lambda: tmp_path)
    adapter = object.__new__(FeishuAdapter)
    adapter._app_id = "cli_a"
    adapter._bot_aliases = []
    adapter._peer_mention_map = {}
    adapter._peer_profile_by_openid = {}
    adapter._peer_profile_by_name = {}
    adapter._peer_canonical_by_profile = {}

    adapter._load_peer_bot_registry()

    assert adapter._bot_aliases == []
    assert adapter._peer_mention_map == {}
    assert adapter._build_peer_colleague_hint() == ""
    assert adapter._inject_peer_mentions("@任何人") is None
    assert adapter._text_mentions_self("@任何人") is False
