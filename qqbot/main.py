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
from qqbot.memory_service import GroupMemoryService


class ChatBot:
    def __init__(self):
        self.http_client = httpx.AsyncClient()
        self.aiohttp_session = aiohttp.ClientSession()
        self.image_service = ImageService(self.aiohttp_session)
        self.event_service = EventService()
        self.gemini_service = GeminiService(self.image_service)
        self.chat_service = ChatService(self.http_client)
        self.memory_service = GroupMemoryService(self.gemini_service, self.chat_service)
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

    def _binary_search_seq(self, messages: list, target_seq: str) -> int:
        """
        二分查找定位到第一个 real_seq 大于 target_seq 的消息的索引。
        返回索引，如果所有消息的 real_seq 都小于等于 target_seq，则返回 len(messages)。
        """
        left, right = 0, len(messages)
        
        while left < right:
            mid = (left + right) // 2
            
            # 获取当前消息的 real_seq，如果不存在则使用默认值
            current_seq = messages[mid].real_seq
            
            # 比较 real_seq（作为字符串比较，因为它们通常是数字字符串）
            try:
                # 尝试转换为整数进行比较
                if int(current_seq) <= int(target_seq):
                    left = mid + 1
                else:
                    right = mid
            except (ValueError, TypeError):
                # 如果转换失败，使用字符串比较
                if current_seq <= target_seq:
                    left = mid + 1
                else:
                    right = mid
        
        return left

    async def _process_message(self, msg_data: dict, enable_vision: bool) -> Message:
        """Processes a message, downloads images, and returns a Message object."""
        sender = msg_data.get("sender", {})
        user_id = sender.get("user_id")
        timestamp = msg_data.get("time")

        # Process images using ImageService
        message_content_raw = await self.image_service.process_message_images(msg_data, enable_vision)


        parsed_content = TypeAdapter(List[MessageSegment]).validate_python(
            message_content_raw
        )

        return Message.from_group_event(msg_data, parsed_content)

    async def handle_event(self, event: GroupMessageEvent):
        if event.post_type != "message" or event.message_type != "group":
            return

        message = await self._process_message(event.model_dump(), enable_vision=settings.get('enable_vision'))
        
        if not self._should_process_message(message):
            return

        group_id = event.group_id
        message_queue = self.message_queues[group_id]

        if not (self._is_bot_mentioned(event.message) or message_queue):
            return
        # Check if we're about to hit capacity and need to process half batch
        if len(message_queue) == message_queue.maxlen:
            # Calculate half size (1/2 of max capacity)
            half_size = max(1, message_queue.maxlen // 2)

            # Extract the oldest half before adding new message
            oldest_half = list(message_queue)[:half_size]

            # Remove the oldest half from deque to make room
            # We need to reconstruct the deque with the remaining messages

            remaining_messages = list(message_queue)[half_size:]
            message_queue.clear()
            message_queue.extend(remaining_messages)

            # Now add the new message
            message_queue.append(message)
            # Process the half batch for memory update
            if oldest_half:
                print(f"Processing half batch of {len(oldest_half)}"
                      f" messages for memory update in group {group_id}")
                asyncio.create_task(self.memory_service.update_memory(group_id, oldest_half))


        if not self._is_bot_mentioned(event.message):
            return

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
        if not group_state["has_history"]:
            history_response = await self.chat_service.get_group_msg_history(
                group_id, count=int(settings.get('max_messages_history') * 4)
            )
            if history_response and history_response.data:
                group_memory = await self.memory_service.get_group_memory(group_id)
                start_index = 0
                messages = history_response.data.messages
                history_messages = []
                max_history = settings.get('max_messages_history')
                if group_memory:
                    memory_last_seq = group_memory.last_seq
                    start_index = self._binary_search_seq(messages, memory_last_seq)
                
                messages = messages[start_index:]
                for msg in messages:
                    if not (msg.sender and msg.message):
                        continue
                    
                    # 根据位置决定是否启用视觉功能
                    msg_index = messages.index(msg)
                    is_recent = msg_index >= len(messages) - settings.get('img_context_length')
                    processed_msg = await self._process_message(msg.model_dump(), is_recent and settings.get('enable_vision'))
                    
                    if self._should_process_message(processed_msg):
                        history_messages.append(processed_msg)
                if len(history_messages) <= max_history:
                    # 所有新消息都放入队列
                    queue_messages = history_messages
                    memory_update_messages = []
                else:
                    # 取最后的 max_messages_history/2 条放入队列，前面的用于更新 memory
                    memory_update_messages = history_messages[:-max_history//2]
                    queue_messages = history_messages[-max_history//2:]
                new_queue = deque(maxlen=settings.get('max_messages_history'))
                new_queue.extend(queue_messages)
                self.message_queues[group_id] = new_queue
                if memory_update_messages:
                    asyncio.create_task(self.memory_service.update_memory(group_id, memory_update_messages))
                group_state["has_history"] = True

        history = self.message_queues[group_id]
        
        group_memory = await self.memory_service.get_group_memory(group_id)
        if group_memory:
            print(f"Using memory for group {group_id}: {group_memory.memory[:200]}...")
        else:
            print(f"No memory for group {group_id}")

        response_sentence = ""
        first_chunk = True
        try:
            async for chunk in self.gemini_service.generate_content_stream(history, group_memory.memory if group_memory else ""):
                if chunk.text is not None:
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
    
    # Set the memory service in the API server
    
    try:
        # Run both the bot and API server concurrently
        await bot.run()
    finally:
        await bot.close()


if __name__ == "__main__":
    print("Lain Bot is starting...")
    print(f"API server will be available at http://0.0.0.0:{settings.get('api_port', 8000)}")
    asyncio.run(main())

