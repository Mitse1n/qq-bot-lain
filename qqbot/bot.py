import asyncio
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Callable, List, Optional, Protocol, Sequence, Set

import aiohttp
import httpx
from pydantic import TypeAdapter, ValidationError

from qqbot.config_loader import settings as default_settings
from qqbot.database import GroupDatabase
from qqbot.message_parsing import parse_message_content_from_history
from qqbot.models import AtMessageSegment, GroupMessageEvent, Message, MessageSegment, TokenBucket
from qqbot.services import ChatService, EventService, GeminiService, ImageService
from qqbot.streaming import split_message_stream
from qqbot.text_utils import (
    convert_md_2_pure_text,
    delete_formatted_prefix,
    delete_qq_prefix,
)


class GroupMessageStore(Protocol):
    def insert_message(self, message: Message) -> bool:
        ...

    def insert_messages_bulk(self, messages: List[Message]) -> int:
        ...

    def get_recent_messages(self, limit: int = 100) -> List[Message]:
        ...

    def get_max_real_seq(self) -> Optional[int]:
        ...


DbFactory = Callable[[int], GroupMessageStore]


class ChatBot:
    def __init__(
        self,
        *,
        settings=default_settings,
        http_client: Optional[httpx.AsyncClient] = None,
        aiohttp_session: Optional[aiohttp.ClientSession] = None,
        image_service: Optional[ImageService] = None,
        event_service: Optional[EventService] = None,
        gemini_service: Optional[GeminiService] = None,
        chat_service: Optional[ChatService] = None,
        db_factory: Optional[DbFactory] = None,
        max_concurrency: int = 10,
        start_image_cleanup: bool = True,
    ):
        self.settings = settings

        self._owns_http_client = http_client is None
        self.http_client = http_client or httpx.AsyncClient()

        self._owns_aiohttp_session = aiohttp_session is None
        self.aiohttp_session = aiohttp_session or aiohttp.ClientSession()

        self.image_service = image_service or ImageService(
            self.aiohttp_session, settings=self.settings, start_cleanup=start_image_cleanup
        )
        self.event_service = event_service or EventService(settings=self.settings)
        self.gemini_service = gemini_service or GeminiService(
            self.image_service, settings=self.settings
        )
        self.chat_service = chat_service or ChatService(
            self.http_client, settings=self.settings
        )

        if db_factory is None:
            data_dir = self.settings.get("data_dir", "./data")
            self._db_factory: DbFactory = lambda group_id: GroupDatabase.get_instance(
                group_id, data_dir=data_dir
            )
        else:
            self._db_factory = db_factory

        self.pulled_groups: Set[int] = set()
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.active_group_tasks: Set[int] = set()

        self.rate_limiters = defaultdict(
            lambda: TokenBucket(
                max_tokens=int(
                    self.settings.get("rate_limit.max_messages_per_hour", 3) or 3
                ),
                time_window=int(
                    self.settings.get("rate_limit.time_window_seconds", 3600) or 3600
                ),
            )
        )

    def _get_group_db(self, group_id: int) -> GroupMessageStore:
        return self._db_factory(group_id)

    async def run(self):
        print("Lain Bot is running...")
        async for event_data in self.event_service.listen():
            if not isinstance(event_data, dict) or "post_type" not in event_data:
                continue
            try:
                message_event = GroupMessageEvent.model_validate(event_data)
                asyncio.create_task(self.handle_event(message_event))
            except ValidationError as e:
                print(f"Error validating event data: {e}")
                print(event_data)

    def _is_bot_mentioned(self, message_segments: list) -> bool:
        return any(
            isinstance(msg, AtMessageSegment)
            and msg.data.qq == self.settings.get("bot_qq_id")
            for msg in message_segments
        )

    def _should_process_message(self, message: Message) -> bool:
        text, _ = message.get_formatted_text(
            vision_enabled=self.settings.get("enable_vision", False)
        )
        return text.strip() != ""

    async def _process_message(self, msg_data: dict, enable_vision: bool) -> Message:
        sender = msg_data.get("sender", {})
        user_id = sender.get("user_id")
        timestamp = msg_data.get("time")

        real_seq_raw = msg_data.get("real_seq")
        real_seq = int(real_seq_raw) if real_seq_raw is not None else None

        message_content_raw = await self.image_service.process_message_images(
            msg_data, enable_vision
        )

        parsed_content = TypeAdapter(List[MessageSegment]).validate_python(
            message_content_raw
        )

        card = sender.get("card")
        return Message(
            timestamp=datetime.fromtimestamp(timestamp, tz=timezone(timedelta(hours=8))),
            user_id=str(user_id),
            card=str(card) if card else None,
            nickname=sender.get("nickname") or "",
            real_seq=real_seq,
            content=parsed_content,
        )

    async def handle_event(self, event: GroupMessageEvent):
        if event.post_type != "message" or event.message_type != "group":
            return

        group_id = event.group_id

        if group_id not in self.pulled_groups:
            await self._pull_and_store_history(group_id)
            self.pulled_groups.add(group_id)

        message = await self._process_message(
            event.model_dump(), enable_vision=self.settings.get("enable_vision", False)
        )

        if not self._should_process_message(message):
            return

        db = self._get_group_db(group_id)
        db.insert_message(message)

        if not self._is_bot_mentioned(event.message):
            return

        if group_id in self.active_group_tasks:
            print(f"Task for group {group_id} already in progress. Ignoring.")
            return

        rate_limiter = self.rate_limiters[group_id]
        if not rate_limiter.consume():
            minutes_left = rate_limiter.time_until_next_token()
            cooldown_message = f" 服务器冷却中，请{minutes_left}分钟后再试"
            await self.chat_service.send_group_message(
                group_id, cooldown_message, event.message_id, event.user_id
            )
            return

        self.active_group_tasks.add(group_id)
        try:
            async with self.semaphore:
                await self.handle_chat_request(group_id, event.message_id, event.user_id)
        except Exception as e:
            print(f"Error handling event for group {group_id}: {e}")
        finally:
            self.active_group_tasks.remove(group_id)

    async def _pull_and_store_history(self, group_id: int):
        db = self._get_group_db(group_id)
        max_seq = db.get_max_real_seq()

        print(f"Pulling history for group {group_id}, max_seq in db: {max_seq}")

        max_messages_history = int(self.settings.get("max_messages_history", 800) or 800)
        history_response = await self.chat_service.get_group_msg_history(
            group_id, count=int(max_messages_history * 1.5)
        )

        if not history_response or not history_response.data:
            print(f"No history data received for group {group_id}")
            return

        history_messages = []
        for msg in history_response.data.messages:
            if not (msg.sender and msg.message):
                continue

            msg_real_seq = int(msg.real_seq) if msg.real_seq else None
            if max_seq is not None and msg_real_seq is not None and msg_real_seq <= max_seq:
                continue

            processed_msg = await self._process_message(msg.model_dump(), False)

            if self._should_process_message(processed_msg):
                history_messages.append(processed_msg)

        if history_messages:
            inserted = db.insert_messages_bulk(history_messages)
            print(f"Inserted {inserted} historical messages for group {group_id}")
        else:
            print(f"No new historical messages to insert for group {group_id}")

    async def handle_chat_request(self, group_id: int, reply_id: int, mention_id: int):
        db = self._get_group_db(group_id)

        max_messages_history = int(self.settings.get("max_messages_history", 800) or 800)
        history = db.get_recent_messages(limit=max_messages_history)

        response_buffer = ""
        first_chunk = True
        try:
            async for chunk in self.gemini_service.generate_content_stream(history):
                if chunk.text is None:
                    continue

                ready_parts, response_buffer = split_message_stream(
                    response_buffer, chunk.text, min_length=400
                )

                for part in ready_parts:
                    part = delete_formatted_prefix(part)
                    part = delete_qq_prefix(part)
                    part = convert_md_2_pure_text(part)

                    text_to_parse = " " + part if first_chunk else part
                    parsed_segments = parse_message_content_from_history(
                        text_to_parse, history
                    )

                    if first_chunk:
                        await self.chat_service.send_group_message(
                            group_id, parsed_segments, reply_id, mention_id
                        )
                        first_chunk = False
                    else:
                        await self.chat_service.send_group_message(group_id, parsed_segments)

            if response_buffer:
                response_buffer = delete_qq_prefix(response_buffer)
                response_buffer = convert_md_2_pure_text(response_buffer)
                response_buffer = " " + response_buffer if first_chunk else response_buffer
                parsed_segments = parse_message_content_from_history(response_buffer, history)
                if first_chunk:
                    await self.chat_service.send_group_message(
                        group_id, parsed_segments, reply_id, mention_id
                    )
                else:
                    await self.chat_service.send_group_message(group_id, parsed_segments)

        except Exception as e:
            if "503" in str(e):
                error_message = " 哎呀，我的思绪暂时有点混乱,拜托稍后再试试吧？"
            elif "429" in str(e):
                error_message = " 哎呀，我被问得太多了，明天试试吧？"
            else:
                error_message = " 哎呀，出现奇怪的错误了"
            print(f"Error generating content: {e}")
            await self.chat_service.send_group_message(
                group_id, error_message, reply_id, mention_id
            )

    async def close(self):
        await self.image_service.stop_cleanup_task()
        if self._owns_http_client:
            await self.http_client.aclose()
        if self._owns_aiohttp_session:
            await self.aiohttp_session.close()

