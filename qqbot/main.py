import asyncio
import httpx
import aiohttp
from pydantic import ValidationError, TypeAdapter
from collections import defaultdict, deque
from datetime import datetime
from typing import List

from qqbot.config_loader import settings
from qqbot.models import (
    GroupMessageEvent,
    AtMessageSegment,
    TextMessageSegment,
    Message,
    TextData,
    MessageSegment,
)
from qqbot.services import EventService, GeminiService, ChatService, ImageService


class ChatBot:
    def __init__(self):
        self.http_client = httpx.AsyncClient()
        self.aiohttp_session = aiohttp.ClientSession()
        self.image_service = ImageService(self.aiohttp_session)
        self.event_service = EventService()
        self.gemini_service = GeminiService(self.image_service)
        self.chat_service = ChatService(self.http_client)
        self.message_queues = defaultdict(lambda: deque(maxlen=settings.get('max_messages_history')))
        self.group_states = defaultdict(lambda: {"has_history": False})
        self.semaphore = asyncio.Semaphore(10)  # Global concurrency limit
        self.active_group_tasks = set()  # Per-group concurrency control

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

    # def _extract_text(self, message: Message) -> str:
    #     return message.get_formatted_text(vision_enabled=settings.get('enable_vision'))

    def _is_bot_mentioned(self, message_segments: list) -> bool:
        return any(
            isinstance(msg, AtMessageSegment) and msg.data.qq == settings.get('bot_qq_id')
            for msg in message_segments
        )

    def _is_message_from_bot(self, message: Message) -> bool:
        return message.user_id == str(settings.get('bot_qq_id'))

    def _should_process_message(self, message: Message) -> bool:
        text, _ = message.get_formatted_text(vision_enabled=settings.get('enable_vision'))
        return text.strip() != ""

    async def _process_message(self, msg_data: dict) -> Message:
        """Processes a message, downloads images, and returns a Message object."""
        sender = msg_data.get("sender", {})
        user_id = sender.get("user_id")
        timestamp = msg_data.get("time")

        # Process images using ImageService
        message_content_raw = await self.image_service.process_message_images(msg_data)

        parsed_content = TypeAdapter(List[MessageSegment]).validate_python(
            message_content_raw
        )

        return Message(
            timestamp=datetime.fromtimestamp(timestamp),
            user_id=str(user_id),
            card=str(sender.get("card")),
            nickname=sender.get("nickname"),
            content=parsed_content,
        )

    async def handle_event(self, event: GroupMessageEvent):
        if event.post_type != "message" or event.message_type != "group":
            return

        message = await self._process_message(event.model_dump())
        
        if not self._should_process_message(message):
            return

        self.message_queues[event.group_id].append(message)

        if not self._is_bot_mentioned(event.message):
            return

        group_id = event.group_id
        if group_id in self.active_group_tasks:
            print(f"Task for group {group_id} already in progress. Ignoring.")
            return

        self.active_group_tasks.add(group_id)
        try:
            async with self.semaphore:
                await self.handle_chat_request(group_id, event.message_id,event.user_id)
        except Exception as e:
            print(f"Error handling event for group {group_id}: {e}")
        finally:
            self.active_group_tasks.remove(group_id)

    async def handle_chat_request(self, group_id: int, reply_id: int, mention_id:int):
        group_state = self.group_states[group_id]
        message_queue = self.message_queues[group_id]
        if len(message_queue) < 50 and not group_state["has_history"]:
            history_response = await self.chat_service.get_group_msg_history(
                group_id, count=1000
            )
            if history_response and history_response.data:
                history_messages = []
                for msg in history_response.data.messages:
                    if not (msg.sender and msg.message):
                        continue

                    processed_msg = await self._process_message(msg.model_dump())

                    if self._should_process_message(processed_msg):
                        history_messages.append(processed_msg)

                new_queue = deque(maxlen=settings.get('max_messages_history'))
                new_queue.extend(history_messages)
                self.message_queues[group_id] = new_queue
                group_state["has_history"] = True

        history = self.message_queues[group_id]
        response_sentence = ""
        first_chunk = True
        try:
            async for chunk in self.gemini_service.generate_content_stream(history):
                response_sentence += chunk.text
                parts = response_sentence.split("\n\n")
                if len(parts) > 1:
                    for part in parts[:-1]:
                        if part:  # 确保不发送空消息
                            if first_chunk:
                                await self.chat_service.send_group_message(
                                    group_id, " " +part, reply_id, mention_id
                                )
                                first_chunk = False
                            else:
                                await self.chat_service.send_group_message(
                                    group_id, part
                                )
                            self.message_queues[group_id].append(
                                Message(
                                    timestamp=datetime.now(),
                                    user_id=settings.get("bot_qq_id"),
                                    nickname=settings.get("bot_name"),
                                    content=[
                                        TextMessageSegment(
                                            type="text",
                                            data=TextData(text=part),
                                        )
                                    ],
                                )
                            )
                    response_sentence = parts[-1]

            if response_sentence:  # 发送剩余的消息
                if first_chunk:
                    await self.chat_service.send_group_message(
                        group_id, " " + response_sentence, reply_id, mention_id
                    )
                else:
                    await self.chat_service.send_group_message(
                        group_id, response_sentence
                    )
                self.message_queues[group_id].append(
                    Message(
                        timestamp=datetime.now(),
                        user_id=settings.get("bot_qq_id"),
                        nickname=settings.get("bot_name"),
                        content=[
                            TextMessageSegment(
                                type="text", data=TextData(text=response_sentence)
                            )
                        ],
                    )
                )

        except Exception as e:
            if "503" in str(e):
                error_message = " 哎呀，我的思绪暂时有点混乱,拜托稍后再试试吧？"
            elif "429" in str(e):
                error_message = " 哎呀，我被问得太多了，明天试试吧？"
            else:
                error_message = " 哎呀，出现奇怪的错误了"
            print(f"Error generating content: {e}")
            await self.chat_service.send_group_message(group_id, error_message, reply_id, mention_id)
            self.message_queues[group_id].append(
                Message(
                    timestamp=datetime.now(),
                    user_id=settings.get('bot_qq_id'),
                    nickname=settings.get('bot_name'),
                    content=[TextMessageSegment(type="text", data=TextData(text=error_message))],
                ))
            print(f"Error handling request: {e}")

    async def close(self):
        await self.image_service.stop_cleanup_task()
        await self.http_client.aclose()
        await self.aiohttp_session.close()


async def main():
    bot = ChatBot()
    try:
        await bot.run()
    finally:
        await bot.close()


if __name__ == "__main__":
    print("Lain Bot is staring...")
    asyncio.run(main())

