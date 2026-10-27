"""Topic/thread send routing for FeishuAdapter (#78975).

Moved out of adapter.py to offset its growth (the file is over the 2,000-line
ratchet and may only go down). Mixed into FeishuAdapter, so methods keep
accessing adapter state (``_client``, ``_run_blocking``, response helpers).
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)


class FeishuTopicSendMixin:
    """Reply-API routing for topic sends: Feishu's create API accepts only
    open_id/user_id/union_id/email/chat_id as receive_id_type — "thread_id" is
    rejected with 99992402, and the plain-text fallback takes the same branch,
    so both sends were dropped. A topic send needs a reply anchor."""

    _client: Any
    _run_blocking: Any
    _response_succeeded: Any
    _build_reply_message_body: Any
    _build_reply_message_request: Any
    _build_create_message_body: Any
    _build_create_message_request: Any

    async def _fetch_last_message_in_thread(self, thread_id: str) -> Optional[str]:
        """Fetch the last message_id in a thread for reply-based routing."""
        if not self._client or not thread_id:
            return None
        try:
            from lark_oapi.api.im.v1 import ListMessageRequest
            request = ListMessageRequest.builder().container_id_type("thread").container_id(thread_id).page_size(1).build()
            response = await self._run_blocking(self._client.im.v1.message.list, request)
            if self._response_succeeded(response):
                items = getattr(getattr(response, "data", None), "items", None)
                if items and len(items) > 0:
                    return getattr(items[0], "message_id", None)
        except Exception as exc:
            logger.debug("[Feishu] Failed to fetch last message in thread %s: %s", thread_id, exc)
        return None

    async def _send_raw_message(
        self, *, chat_id: str, msg_type: str, payload: str, reply_to: Optional[str], metadata: Optional[dict[str, Any]],
    ) -> Any:
        thread_id = (metadata or {}).get("thread_id")
        effective_reply_to = reply_to or ((metadata or {}).get("reply_to_message_id") if thread_id else None)
        if not effective_reply_to and thread_id:
            # The topic's newest message is the anchor for an ``omt_`` id; the
            # slot can also hold a bare ``om_`` message id (reply-chain root).
            if str(thread_id).startswith("omt_"):
                effective_reply_to = await self._fetch_last_message_in_thread(thread_id)
            elif str(thread_id).startswith("om_"):
                effective_reply_to = thread_id
            if not effective_reply_to:
                self._warn_unroutable_thread_once(str(thread_id), chat_id)
        if effective_reply_to:
            body = self._build_reply_message_body(
                content=payload, msg_type=msg_type,
                reply_in_thread=str(thread_id or "").startswith("omt_"), uuid_value=str(uuid.uuid4()),
            )
            request = self._build_reply_message_request(effective_reply_to, body)
            return await self._run_blocking(self._client.im.v1.message.reply, request)
        if chat_id.startswith("feishu_user_id:"):
            receive_id, receive_id_type = chat_id.split(":", 1)[1], "user_id"
        else:
            receive_id, receive_id_type = chat_id, "open_id" if chat_id.startswith("ou_") else "chat_id"
        body = self._build_create_message_body(
            receive_id=receive_id, msg_type=msg_type, content=payload, uuid_value=str(uuid.uuid4()),
        )
        request = self._build_create_message_request(receive_id_type, body)
        return await self._run_blocking(self._client.im.v1.message.create, request)

    def _warn_unroutable_thread_once(self, thread_id: str, chat_id: str) -> None:
        """Warn once per distinct unusable thread_id: a retrying cron re-sends the same origin."""
        warned = self.__dict__.setdefault("_warned_unroutable_threads", set())
        if thread_id in warned:
            return
        warned.add(thread_id)
        logger.warning(
            "[Feishu] thread_id %s cannot be a create receive_id (create accepts only "
            "open_id/user_id/union_id/email/chat_id) — delivering to %s instead (#78975)",
            thread_id, chat_id,
        )
