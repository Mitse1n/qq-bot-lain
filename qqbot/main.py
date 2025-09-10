import asyncio
import httpx
import aiohttp
from pydantic import ValidationError, TypeAdapter
from collections import defaultdict, deque
from datetime import datetime
from typing import List, Deque
import pprint

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
        self.gemini_service = GeminiService()
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

    def _extract_text(self, message: Message) -> str:
        return message.get_formatted_text()

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
        group_id = msg_data.get("group_id")
        real_seq = msg_data.get("real_seq")

        image_count = 0
        message_content_raw = msg_data.get("message", [])
        if "message" in msg_data and isinstance(message_content_raw, list):
            for segment in message_content_raw:
                if segment.get("type") == "image":
                    image_count += 1
                    if settings.get('enable_vision'):
                        image_url = segment.get("data", {}).get("url")
                        if image_url:
                            # Create a unique filename for each image
                            image_name = (
                                f"{group_id}-{real_seq}-{user_id}-{timestamp}-{image_count}.jpeg"
                            )
                            await self.image_service.download_image(image_url, image_name)
                            # Update the segment to store the local filename
                            segment["data"]["file"] = image_name

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
        print( f"Received message in group {event.group_id} from user {event.user_id}, content: {message.content}" )

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

        history  :Deque[Message] = self.message_queues[group_id]

        try:
            response_text = self.gemini_service.generate_content(history)
        except Exception as e:
            if "503" in str(e):
                response_text = " server overloaded, try again later"
            elif "429" in str(e):
                response_text = " over quota, try again after 1pm"
            else:
                response_text = " Sorry, I had a problem generating a response."
        self.message_queues[group_id].append(
            Message(
                timestamp=datetime.now(),
                user_id=settings.get('bot_qq_id'),
                nickname=settings.get('bot_name'),
                content=[TextMessageSegment(type="text", data=TextData(text=response_text))],
            )
        )
        await self.chat_service.send_group_message(group_id, response_text, reply_id,mention_id)

    async def close(self):
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
    print("当前 config.yaml 配置如下：")
    pprint.pprint(settings.as_dict())
    print("-" * 50)
    asyncio.run(main())
