"""Regression tests for stream consumer thread/topic routing fix.

Verifies that GatewayStreamConsumer correctly passes reply_to on the first
message send, ensuring messages land in the correct topic/thread instead of
the main group chat.

Covers: #6969, #9916, #7355
"""
import json
from unittest.mock import AsyncMock, MagicMock
from types import SimpleNamespace

import pytest

from gateway.stream_consumer import (
    GatewayStreamConsumer,
)


def _make_adapter(send_result=None, edit_result=None, max_length=4096):
    adapter = MagicMock()
    adapter.send = AsyncMock(
        return_value=send_result or SimpleNamespace(success=True, message_id="msg_1")
    )
    adapter.edit_message = AsyncMock(
        return_value=edit_result or SimpleNamespace(success=True)
    )
    adapter.MAX_MESSAGE_LENGTH = max_length
    return adapter


class TestInitialReplyToId:
    """Verify initial_reply_to_id is passed as reply_to on first send."""

    @pytest.mark.asyncio
    async def test_first_send_uses_initial_reply_to_id(self):
        """When initial_reply_to_id is set, first adapter.send() should
        include reply_to=initial_reply_to_id."""
        adapter = _make_adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            metadata={"thread_id": "omt_topic123"},
            initial_reply_to_id="om_user_msg_456",
        )
        await consumer._send_or_edit("Hello world")

        adapter.send.assert_called_once()
        call_kwargs = adapter.send.call_args[1]
        assert call_kwargs["reply_to"] == "om_user_msg_456", (
            "First send should pass initial_reply_to_id as reply_to"
        )
        assert call_kwargs["chat_id"] == "chat_123"


    @pytest.mark.asyncio
    async def test_subsequent_edits_ignore_initial_reply_to_id(self):
        """After first send, edits should use message_id, not initial_reply_to_id."""
        adapter = _make_adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            metadata={"thread_id": "omt_topic123"},
            initial_reply_to_id="om_user_msg_456",
        )

        # First send
        await consumer._send_or_edit("Hello world")
        assert adapter.send.call_count == 1

        # Second call should edit, not send
        await consumer._send_or_edit("Hello world updated")
        assert adapter.send.call_count == 1, "Should edit, not send again"
        adapter.edit_message.assert_called_once()
        edit_kwargs = adapter.edit_message.call_args[1]
        assert edit_kwargs["message_id"] == "msg_1"
        assert edit_kwargs["chat_id"] == "chat_123"


class TestOverflowFirstMessage:
    """Verify thread routing is preserved when the first message overflows."""

    @pytest.mark.asyncio
    async def test_overflow_first_send_uses_initial_reply_to_id(self):
        """When first message exceeds platform limit and is split into chunks,
        each chunk should be threaded to initial_reply_to_id, not None."""
        adapter = _make_adapter(max_length=10)
        adapter.truncate_message = MagicMock(
            return_value=["chunk_1", "chunk_2"]
        )
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            metadata={"thread_id": "omt_topic123"},
            initial_reply_to_id="om_user_msg_789",
        )

        # Inject oversized accumulated text to trigger overflow path
        consumer._accumulated = "A" * 100
        consumer._current_edit_interval = 999
        await consumer._send_new_chunk("chunk_1", consumer._message_id or consumer._initial_reply_to_id)

        adapter.send.assert_called_once()
        call_kwargs = adapter.send.call_args[1]
        assert call_kwargs["reply_to"] == "om_user_msg_789", (
            "Overflow first chunk should use initial_reply_to_id"
        )


class TestFeishuFallbackThreadRouting:
    """Verify FeishuAdapter._send_raw_message routes to topic on fallback."""

    @staticmethod
    def _wire(adapter, *, anchor=None):
        """Real builders + passthrough _run_blocking; mocked SDK client."""
        from plugins.platforms.feishu.adapter import FeishuAdapter
        client = MagicMock()
        client.im.v1.message.create = MagicMock(
            return_value=SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="new_msg_1")))
        client.im.v1.message.reply = MagicMock(
            return_value=SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="reply_1")))
        adapter._client = client
        adapter._build_create_message_body = FeishuAdapter._build_create_message_body
        adapter._build_create_message_request = FeishuAdapter._build_create_message_request
        adapter._build_reply_message_body = FeishuAdapter._build_reply_message_body
        adapter._build_reply_message_request = FeishuAdapter._build_reply_message_request
        adapter._warn_unroutable_thread_once = (
            FeishuAdapter._warn_unroutable_thread_once.__get__(adapter))
        adapter._fetch_last_message_in_thread = AsyncMock(return_value=anchor)

        async def _run_blocking_passthrough(func, *args):
            return func(*args)
        adapter._run_blocking = _run_blocking_passthrough
        return client

    @staticmethod
    def _body(request):
        # Lark SDK exposes .body; the in-tree fallback exposes .request_body.
        body = getattr(request, "request_body", None) or getattr(request, "body", None)
        assert body is not None, "request has neither .body nor .request_body"
        return body

    @pytest.mark.asyncio
    async def test_topic_without_reply_to_replies_to_thread_anchor(self):
        """#78975: an ``omt_`` topic has no create receive_id_type — the topic's last
        message becomes the reply anchor and the send stays inside the topic."""
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = MagicMock(spec=FeishuAdapter)
        client = self._wire(adapter, anchor="om_topic_last")

        await FeishuAdapter._send_raw_message(
            adapter,
            chat_id="oc_main_chat",
            msg_type="text",
            payload=json.dumps({"text": "hello"}),
            reply_to=None,
            metadata={"thread_id": "omt_topic_abc"},
        )

        client.im.v1.message.create.assert_not_called()
        client.im.v1.message.reply.assert_called_once()
        request = client.im.v1.message.reply.call_args[0][0]
        assert getattr(request, "message_id", None) == "om_topic_last"
        assert self._body(request).reply_in_thread is True

    @pytest.mark.asyncio
    async def test_unresolvable_topic_falls_back_to_chat_id(self, caplog):
        """No anchor (e.g. the read API lacks im:message.group_msg) must degrade to a
        top-level chat send, never to receive_id_type='thread_id'."""
        import logging
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = MagicMock(spec=FeishuAdapter)
        client = self._wire(adapter, anchor=None)

        with caplog.at_level(logging.WARNING):
            await FeishuAdapter._send_raw_message(
                adapter,
                chat_id="oc_main_chat",
                msg_type="text",
                payload=json.dumps({"text": "hello"}),
                reply_to=None,
                metadata={"thread_id": "omt_topic_abc"},
            )

        client.im.v1.message.reply.assert_not_called()
        client.im.v1.message.create.assert_called_once()
        request = client.im.v1.message.create.call_args[0][0]
        assert getattr(request, "receive_id_type", None) == "chat_id"
        assert self._body(request).receive_id == "oc_main_chat"
        assert "cannot be a create receive_id" in caplog.text

    @pytest.mark.asyncio
    async def test_message_id_in_thread_slot_is_used_as_anchor(self):
        """The thread slot can hold an ``om_`` message id (reply-chain root); reply to
        it directly instead of asking the thread read API."""
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = MagicMock(spec=FeishuAdapter)
        client = self._wire(adapter, anchor="om_should_not_be_used")

        await FeishuAdapter._send_raw_message(
            adapter,
            chat_id="oc_main_chat",
            msg_type="text",
            payload=json.dumps({"text": "hello"}),
            reply_to=None,
            metadata={"thread_id": "om_origin_1"},
        )

        adapter._fetch_last_message_in_thread.assert_not_awaited()
        client.im.v1.message.create.assert_not_called()
        request = client.im.v1.message.reply.call_args[0][0]
        assert getattr(request, "message_id", None) == "om_origin_1"

    @pytest.mark.asyncio
    async def test_message_id_anchor_is_not_threaded(self):
        """A bare ``om_`` anchor is a plain conversation message: reply stays
        unthreaded (reply_in_thread=False) so no second 99992402 class."""
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = MagicMock(spec=FeishuAdapter)
        client = self._wire(adapter, anchor="om_should_not_be_used")

        await FeishuAdapter._send_raw_message(
            adapter,
            chat_id="oc_main_chat",
            msg_type="text",
            payload=json.dumps({"text": "hello"}),
            reply_to=None,
            metadata={"thread_id": "om_origin_1"},
        )

        assert self._body(client.im.v1.message.reply.call_args[0][0]).reply_in_thread is False

    @pytest.mark.asyncio
    async def test_polluted_thread_value_warns_once_and_falls_back(self, caplog):
        """A value that is neither omt_ nor om_ is ignored; a retrying cron re-sending
        the same origin must not flood the log."""
        import logging
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = MagicMock(spec=FeishuAdapter)
        client = self._wire(adapter, anchor=None)

        with caplog.at_level(logging.WARNING):
            for _ in range(3):
                await FeishuAdapter._send_raw_message(
                    adapter,
                    chat_id="oc_main_chat",
                    msg_type="text",
                    payload=json.dumps({"text": "hello"}),
                    reply_to=None,
                    metadata={"thread_id": "gc-not-a-topic"},
                )

        assert caplog.text.count("cannot be a create receive_id") == 1
        assert client.im.v1.message.create.call_count == 3
        request = client.im.v1.message.create.call_args[0][0]
        assert getattr(request, "receive_id_type", None) == "chat_id"


class TestFallbackResendThreading:
    """#103068: a full fallback resend replaces the preview, so it must land in
    the originating thread; tail continuations keep their unthreaded delivery."""

    @pytest.mark.asyncio
    async def test_full_fallback_resend_threads_first_chunk_only(self):
        adapter = _make_adapter(max_length=700)
        adapter.send.side_effect = [
            SimpleNamespace(success=True, message_id=f"full_{i}") for i in range(10)
        ]
        consumer = GatewayStreamConsumer(adapter, "chat_123", initial_reply_to_id="om_user_1")
        consumer._message_id = "om_preview"
        consumer._last_sent_text = "truncated snapshot"
        consumer._already_sent = True
        consumer._fallback_final_send = True

        final = " ".join(["word"] * 400)  # not prefixed by the snapshot -> full resend
        await consumer._send_fallback_final(final)

        calls = adapter.send.await_args_list
        assert len(calls) > 1
        assert calls[0].kwargs["reply_to"] == "om_user_1"
        assert all("reply_to" not in c.kwargs for c in calls[1:])
