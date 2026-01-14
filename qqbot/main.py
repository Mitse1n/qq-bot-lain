import asyncio
import httpx
import aiohttp
import re
from pydantic import ValidationError, TypeAdapter
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import List

from qqbot.config_loader import settings
from qqbot.models import (
    GroupMessageEvent,
    AtMessageSegment,
    TextMessageSegment,
    Message,
    TextData,
    MessageSegment,
    TokenBucket,
)
from qqbot.services import EventService, GeminiService, ChatService, ImageService
from qqbot.agentic_memory import GroupMemoryManager


def split_message_stream(current_buffer: str, new_chunk: str, min_length: int = 400) -> tuple[list[str], str]:
    """
    Split message stream into parts based on \n\n separator with a minimum length threshold.
    
    Args:
        current_buffer: The current accumulated message buffer
        new_chunk: New chunk of text to append
        min_length: Minimum length for a part to be split out (default: 400)
        
    Returns:
        tuple: (list of ready parts to send, remaining buffer)
    """
    buffer = current_buffer + new_chunk
    parts = buffer.split("\n\n")
    
    # If there's only one part (no separator found), keep accumulating
    if len(parts) == 1:
        return [], buffer
    
    ready_parts = []
    remaining_buffer = ""
    
    # Process all parts except the last one (which might be incomplete)
    for i, part in enumerate(parts[:-1]):
        if not part:  # Skip empty parts
            continue
            
        # If we have a remaining buffer, check if combining makes sense
        if remaining_buffer:
            combined = remaining_buffer + "\n\n" + part
            if len(combined) >= min_length:
                ready_parts.append(combined)
                remaining_buffer = ""
            else:
                # Keep accumulating if under threshold
                remaining_buffer = combined
        else:
            # Start new buffer or emit if long enough
            if len(part) >= min_length:
                ready_parts.append(part)
            else:
                remaining_buffer = part
    
    # Handle the last part (potentially incomplete)
    last_part = parts[-1]
    if remaining_buffer:
        remaining_buffer = remaining_buffer + "\n\n" + last_part if last_part else remaining_buffer
    else:
        remaining_buffer = last_part
    
    return ready_parts, remaining_buffer


class ChatBot:
    def __init__(self):
        self.http_client = httpx.AsyncClient()
        self.aiohttp_session = aiohttp.ClientSession()
        self.image_service = ImageService(self.aiohttp_session)
        self.event_service = EventService()
        self.gemini_service = GeminiService(self.image_service)
        self.chat_service = ChatService(self.http_client)
        self.group_memory = GroupMemoryManager(self.gemini_service)
        self.message_queues = defaultdict(lambda: deque(maxlen=settings.get('max_messages_history')))
        self.group_states = defaultdict(lambda: {"has_history": False})
        self.semaphore = asyncio.Semaphore(10)  # Global concurrency limit
        self.active_group_tasks = set()  # Per-group concurrency control
        self.rate_limiters = defaultdict(lambda: TokenBucket(
            max_tokens=settings.get('rate_limit.max_messages_per_hour', 3),
            time_window=settings.get('rate_limit.time_window_seconds', 3600)
        ))

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

    async def _process_message(self, msg_data: dict, enable_vision: bool) -> Message:
        """Processes a message, downloads images, and returns a Message object."""
        sender = msg_data.get("sender", {})
        user_id = sender.get("user_id")
        timestamp = msg_data.get("time")
        real_seq = str(msg_data.get("real_seq", ""))

        # Process images using ImageService
        message_content_raw = await self.image_service.process_message_images(msg_data, enable_vision)


        parsed_content = TypeAdapter(List[MessageSegment]).validate_python(
            message_content_raw
        )

        return Message(
            timestamp=datetime.fromtimestamp(timestamp, tz=timezone(timedelta(hours=8))),
            user_id=str(user_id),
            nickname=sender.get("nickname"),
            real_seq=real_seq,
            card=str(sender.get("card")),
            content=parsed_content,
        )

    async def handle_event(self, event: GroupMessageEvent):
        if event.post_type != "message" or event.message_type != "group":
            return

        message = await self._process_message(event.model_dump(), enable_vision=settings.get('enable_vision'))
        
        if not self._should_process_message(message):
            return

        self.message_queues[event.group_id].append(message)
        try:
            self.group_memory.ingest_message(event.group_id, message)
        except Exception as e:
            print(f"agentic_memory ingest error for group {event.group_id}: {e}")

        if not self._is_bot_mentioned(event.message):
            return

        group_id = event.group_id
        if group_id in self.active_group_tasks:
            print(f"Task for group {group_id} already in progress. Ignoring.")
            return

        # 检查限流
        rate_limiter = self.rate_limiters[group_id]
        if not rate_limiter.consume():
            # Token 用完了，发送冷却消息
            minutes_left = rate_limiter.time_until_next_token()
            cooldown_message = f" 服务器冷却中，请{minutes_left}分钟后再试"
            await self.chat_service.send_group_message(group_id, cooldown_message, event.message_id, event.user_id)
            return

        self.active_group_tasks.add(group_id)
        try:
            async with self.semaphore:
                await self.handle_chat_request(group_id, event.message_id,event.user_id)
        except Exception as e:
            print(f"Error handling event for group {group_id}: {e}")
        finally:
            self.active_group_tasks.remove(group_id)

    def _parse_message_content(self, text: str, group_id: int) -> List[dict]:
        history = self.message_queues[group_id]
        known_users = set()
        for msg in history:
            known_users.add(str(msg.user_id))
            
        segments = []
        last_end = 0
        pattern = re.compile(r'@(\d{5,})')
        
        for match in pattern.finditer(text):
            qq_id = match.group(1)
            start, end = match.span()
            
            if qq_id in known_users:
                if start > last_end:
                    segments.append({
                        "type": "text",
                        "data": {"text": text[last_end:start]}
                    })
                segments.append({
                    "type": "at",
                    "data": {"qq": qq_id}
                })
                last_end = end
                
        if last_end < len(text):
            segments.append({
                "type": "text",
                "data": {"text": text[last_end:]}
            })
            
        if not segments:
             segments.append({"type": "text", "data": {"text": ""}})
             
        return segments

    async def handle_chat_request(self, group_id: int, reply_id: int, mention_id:int):
        group_state = self.group_states[group_id]
        message_queue = self.message_queues[group_id]
        if len(message_queue) < 50 and not group_state["has_history"]:
            history_response = await self.chat_service.get_group_msg_history(
                group_id, count=int(settings.get('max_messages_history') * 1.5)
            )
            if history_response and history_response.data:
                history_messages = []
                for msg in history_response.data.messages[:-settings.get('img_context_length')]:
                    if not (msg.sender and msg.message):
                        continue

                    processed_msg = await self._process_message(msg.model_dump(),False)

                    if self._should_process_message(processed_msg):
                        history_messages.append(processed_msg)
                for msg in history_response.data.messages[-settings.get('img_context_length'):]:
                    if not (msg.sender and msg.message):
                        continue
                    processed_msg = await self._process_message(msg.model_dump(),settings.get('enable_vision'))
                    if self._should_process_message(processed_msg):
                        history_messages.append(processed_msg)

                new_queue = deque(maxlen=settings.get('max_messages_history'))
                new_queue.extend(history_messages)
                self.message_queues[group_id] = new_queue
                group_state["has_history"] = True

        history = self.message_queues[group_id]
        response_buffer = ""
        first_chunk = True
        try:
            memory_prompt = None
            try:
                memory_prompt = await self.group_memory.build_memory_prompt(
                    group_id, history, mention_user_id=str(mention_id)
                )
            except Exception as e:
                print(f"agentic_memory retrieval error for group {group_id}: {e}")

            async for chunk in self.gemini_service.generate_content_stream(
                history, memory_prompt=memory_prompt
            ):
                if chunk.text is not None:
                    ready_parts, response_buffer = split_message_stream(
                        response_buffer, chunk.text, min_length=400
                    )
                    
                    for part in ready_parts:
                        text_to_parse = " " + part if first_chunk else part
                        parsed_segments = self._parse_message_content(text_to_parse, group_id)
                        
                        if first_chunk:
                            await self.chat_service.send_group_message(
                                group_id, parsed_segments, reply_id, mention_id
                            )
                            first_chunk = False
                        else:
                            await self.chat_service.send_group_message(
                                group_id, parsed_segments
                            )
                        
                        content_segments = TypeAdapter(List[MessageSegment]).validate_python(parsed_segments)
                        self.message_queues[group_id].append(
                            Message(
                                timestamp=datetime.now(timezone(timedelta(hours=8))),
                                user_id=str(settings.get("bot_qq_id")),
                                nickname=settings.get("bot_name"),
                                real_seq="",  # Bot messages don't have real_seq
                                content=content_segments,
                            )
                        )

            if response_buffer:  # 发送剩余的消息
                text_to_parse = " " + response_buffer if first_chunk else response_buffer
                parsed_segments = self._parse_message_content(text_to_parse, group_id)
                
                if first_chunk:
                    await self.chat_service.send_group_message(
                        group_id, parsed_segments, reply_id, mention_id
                    )
                else:
                    await self.chat_service.send_group_message(
                        group_id, parsed_segments
                    )
                
                content_segments = TypeAdapter(List[MessageSegment]).validate_python(parsed_segments)
                self.message_queues[group_id].append(
                    Message(
                        timestamp=datetime.now(timezone(timedelta(hours=8))),
                        user_id=str(settings.get("bot_qq_id")),
                        nickname=settings.get("bot_name"),
                        real_seq="",  # Bot messages don't have real_seq
                        content=content_segments,
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
                    timestamp=datetime.now(timezone(timedelta(hours=8))),
                    user_id=settings.get('bot_qq_id'),
                    nickname=settings.get('bot_name'),
                    real_seq="",  # Bot messages don't have real_seq
                    content=[TextMessageSegment(type="text", data=TextData(text=error_message))],
                ))
            print(f"Error handling request: {e}")

    async def close(self):
        await self.image_service.stop_cleanup_task()
        await self.group_memory.close()
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

