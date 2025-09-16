import dataclasses
import os
import asyncio
import json
from typing import Optional, List, Dict, Any, TYPE_CHECKING, Tuple
from datetime import datetime
from pathlib import Path
import re

from qqbot.config_loader import settings
from qqbot.models import Message, GroupHistoryMessageEvent

if TYPE_CHECKING:
    from qqbot.services import GeminiService, ChatService
@dataclasses
class Memory:
    def __init__(self, group_id: int,
                 first_seq:str,
                 last_seq:str,
                 memory:str) -> None:
        self.group_id = group_id
        self.first_seq = first_seq
        self.last_seq = last_seq
        self.memory = memory

class GroupMemoryService:
    """
    Manages group-specific memory using LLM to compress and maintain chat history.
    """
    
    # Memory configuration constants
    MINIMUM_MESSAGES_FOR_MEMORY = 400
    MEMORY_UPDATE_INTERVAL = 400
    MEMORY_TARGET_LENGTH = 1600
    INITIAL_BATCH_SIZE = 800
    
    def __init__(self, gemini_service: 'GeminiService', chat_service: 'ChatService'):
        self.gemini_service = gemini_service
        self.chat_service = chat_service
        self.memory_dir = Path("data/memory")
        self.memory_dir.mkdir(exist_ok=True)
        
        # Track message counts per group for periodic updates
        self.group_message_counts: Dict[int, int] = {}
        # Track if memory has been initialized for each group
        self.group_memory_initialized: Dict[int, bool] = {}
        # Cache for group memories
        self.group_memories: Dict[int, Memory] = {}
        # This is now handled by ChatBot's message_queues - no longer needed here
        
    def _get_memory_file_path(self, group_id: int, first_seq: str, last_seq: str) -> Path:
        """Generate memory file path based on group_id and message sequence range."""
        filename = f"{group_id}-{first_seq}-{last_seq}.txt"
        return self.memory_dir / filename
        
    def _get_latest_memory_file(self, group_id: int) -> Optional[Path]:
        """Get the latest memory file for a group."""
        pattern = f"{group_id}-*.txt"
        memory_files = list(self.memory_dir.glob(pattern))
        
        if not memory_files:
            return None
            
        # Sort by modification time, get the latest
        memory_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return memory_files[0]
    
    def _extract_seq_from_filename(self, path: Path) -> Tuple[str, str]:
        """Extracts sequence numbers from a memory file path."""
        match = re.match(r"memory_(\d+)_(\d+)_(\d+)\.json", path.name)
        if match:
            return match.group(2), match.group(3)
        return "", ""
    
    def _get_senders_text(self, messages: List[Any]) -> str:
        """Create a formatted string of senders from a list of messages."""
        senders: Dict[str, Dict[str, str]] = {}
        for msg in messages:
            user_id = ""
            nickname = ""
            card = ""

            if isinstance(msg, Message):
                user_id = str(msg.user_id)
                nickname = msg.nickname
                card = msg.card
            elif isinstance(msg, GroupHistoryMessageEvent):
                if msg.sender:
                    user_id = str(msg.user_id)
                    nickname = msg.sender.nickname
                    card = msg.sender.card
            
            if user_id and user_id not in senders:
                senders[user_id] = {"user_id": user_id, "nickname": nickname, "card": card}

        sender_lines = []
        for sender in senders.values():
            card_text = f" 群昵称: {sender['card']}" if sender['card'] else ""
            sender_lines.append(f"ID: {sender['user_id']} 账号名: {sender['nickname']}{card_text}")
            
        return "\n".join(sender_lines)
    
    def _get_initial_memory_prompt(self, messages: str, senders_text: str) -> str:
        """返回生成初始群聊记忆的 prompt。"""
        return f"""作为这个群聊的 AI 助手{settings.get('bot_name')}, id 是 {settings.get('bot_qq_id')}。
        你的任务是创建“群聊记忆”的第一个版本。这份记忆应该是一份简洁的聊天记录摘要，方便你以后回复群聊时使用。

**你的目标：** 创建一份约 {self.MEMORY_TARGET_LENGTH} 个字符的摘要, 实在没有足够有价值的信息可以缩短。

**操作指南:**
- **识别关键信息:** 记录下群里做出的决定、约定、计划的活动或提出的重要问题。
- **捕捉群聊氛围:** 如果有独特的群聊文化（例如：固定的玩笑、昵称），可以简要描述。
- **过滤无关内容:** 忽略日常闲聊、打招呼等。
- **保持客观中立:** 仅总结，不要添加主观臆断。
- **关于群员:** 聊天记录中的群员使用 ID 显示，你可以参考下面的列表来了解他们的昵称。在生成记忆时，请优先使用群员 ID 来指代他们，以避免混淆。

**涉及到的群员:**
---
{senders_text}
---

**群聊记录:**
---
{messages}
---

请根据以上聊天记录，生成初始的群聊记忆。"""

    def _get_update_memory_prompt(self, existing_memory: str, new_messages: str, senders_text: str) -> str:
        """返回更新群聊记忆的 prompt。"""
        return \
            f"""作为这个群聊的 AI 助手{settings.get('bot_name')}, id 是 {settings.get('bot_qq_id')}。
            你的任务是根据最新的消息更新“群聊记忆”。请保持记忆的时效性、简洁性和相关性。

**操作指南:**
1.  **阅读现有记忆:** 理解当前已记录的群聊重点。
2.  **分析最新消息:** 从新消息中提取重要的信息和对话。
3.  **整合关键信息:**
    - 将新的重要话题、事件或决定加入记忆。
    - 如果现有话题有新进展，请更新。
    - 移除不再重要或已过时的信息。
4.  **保持简洁客观:** 过滤和自己无关的闲聊，确保记忆内容准确、中立。
5.  **维持适当长度:** 更新后的记忆长度约 {self.MEMORY_TARGET_LENGTH} 个字符。
6.  **关于群员:** 聊天记录中的群员使用 ID 显示。在更新记忆时，请优先使用群员 ID 来指代他们，以避免混淆。

**涉及到的群员:**
---
{senders_text}
---

**现有群聊记忆:**
---
{existing_memory}
---

**最新消息:**
---
{new_messages}
---

请根据最新消息，生成更新后的群聊记忆。"""
    
    async def _generate_memory_with_llm(self, prompt: str) -> str:
        """Generate memory content using LLM."""
        try:
            # Create a simple message structure for the LLM
            from collections import deque
            from qqbot.models import TextMessageSegment, TextData
            
            # Create a message with the prompt
            message = Message.create_system_message(prompt)
            
            messages = deque([message])
            
            # Generate response using the small model for memory processing
            response_text = ""
            retry_delay = 1;
            retry_count = 0;
            while  retry_count <= 8:
                try:
                
                    async for chunk in self.gemini_service.generate_content_stream(messages):
                        if chunk.text is not None:
                            response_text += chunk.text
                    break
                except Exception as e:
                    retry_count += 1
                    print(f"Error generating memory with LLM: {e}, retry_count: {retry_count}")
                    if "503" in str(e):
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                    elif "429" in str(e):
                        self.gemini_service._rotate_api_key()
                        retry_delay = 1
                    else:
                        print(f"Error generating memory after all retries with LLM: {e}")
                        raise e
                        
                
            return response_text.strip()
            
        except Exception as e:
            print(f"Error generating memory with LLM: {e}")
            return ""
    
    def _format_messages_for_memory(self, messages: List[GroupHistoryMessageEvent]) -> str:
        """Format messages for memory processing."""
        formatted_messages = []
        
        for msg in messages:
            if not msg.sender or not msg.message:
                continue
                
            # Extract text content from message segments
            text_parts = []
            for segment in msg.message:
                if hasattr(segment, 'type') and segment.type == "text":
                    text_parts.append(segment.data.text)
                elif hasattr(segment, 'type') and segment.type == "at":
                    text_parts.append(f"@{segment.data.qq}")
                elif hasattr(segment, 'type') and segment.type == "image":
                    text_parts.append("[图片]")
            
            message_text = "".join(text_parts).strip()
            if message_text:
                timestamp = datetime.fromtimestamp(msg.time).strftime('%m-%d %H:%M')
                user_info = str(msg.user_id)
                formatted_messages.append(f"({timestamp}) {user_info}: {message_text}")
        
        return "\n".join(formatted_messages)
    
    async def _convert_history_message_to_message(self, msg: GroupHistoryMessageEvent) -> Optional[Message]:
        """Convert a GroupHistoryMessageEvent to a Message object."""
        return Message.from_history_event(msg)
    
    async def initialize_group_memory(self, group_id: int) -> bool:
        """
        Initialize memory for a new group by fetching recent messages and processing them.
        Only generates memory if at least 400 messages are available.
        Returns True if initialization was successful.
        """
        if self.group_memory_initialized.get(group_id, False):
            return True
            
        print(f"Initializing memory for group {group_id}")
        
        try:
            # Fetch the most recent 1600 messages
            history_response = await self.chat_service.get_group_msg_history(
                group_id, count=1600
            )
            
            if not history_response or not history_response.data or not history_response.data.messages:
                print(f"No messages found for group {group_id}")
                return False
            
            messages = history_response.data.messages
            print(f"Fetched {len(messages)} messages for group {group_id}")
            
            if len(messages) == 0:
                return False
            
            senders_text = self._get_senders_text(messages)
            
            # Check if we have enough messages for initial memory generation
            if len(messages) < self.MINIMUM_MESSAGES_FOR_MEMORY:
                print(f"Insufficient messages ({len(messages)}) for group {group_id}, "
                      f"will accumulate until {self.MINIMUM_MESSAGES_FOR_MEMORY} messages are reached")
                # With the new unified design, initial memory generation is handled by ChatBot
                # when it accumulates enough messages in its deque
                self.group_message_counts[group_id] = len(messages)
                return False  # Not enough messages to generate initial memory yet
            
            # Get sequence numbers for filename
            first_seq = messages[0].real_seq if messages[0].real_seq else "0"
            last_seq = messages[-1].real_seq if messages[-1].real_seq else "0"
            
            # Process first batch of messages (oldest)
            first_batch = messages[:self.INITIAL_BATCH_SIZE] if len(messages) >= self.INITIAL_BATCH_SIZE else messages
            first_batch_text = self._format_messages_for_memory(first_batch)
            
            if not first_batch_text.strip():
                print(f"No valid text content found in messages for group {group_id}")
                return False
            
            # Generate initial memory
            initial_prompt = self._get_initial_memory_prompt(first_batch_text, senders_text)
            
            memory_content = await self._generate_memory_with_llm(initial_prompt)
            
            if not memory_content:
                print(f"Failed to generate initial memory for group {group_id}")
                return False
            
            # Process remaining messages if any
            if len(messages) > self.INITIAL_BATCH_SIZE:
                remaining_batch = messages[self.INITIAL_BATCH_SIZE:]
                remaining_batch_text = self._format_messages_for_memory(remaining_batch)
                
                if remaining_batch_text.strip():
                    update_prompt = self._get_update_memory_prompt(memory_content, remaining_batch_text, senders_text)
                    
                    updated_memory = await self._generate_memory_with_llm(update_prompt)
                    if updated_memory:
                        memory_content = updated_memory
            
            # Save memory to file
            memory_file = self._get_memory_file_path(group_id, first_seq, last_seq)
            with open(memory_file, 'w', encoding='utf-8') as f:
                f.write(memory_content)
            
            # Cache the memory
            self.group_memories[group_id] = Memory(
                group_id, first_seq, last_seq, memory_content
            )
            self.group_memory_initialized[group_id] = True
            self.group_message_counts[group_id] = len(messages)
            
            print(f"Successfully initialized memory for group {group_id}")
            return True
            
        except Exception as e:
            print(f"Error initializing memory for group {group_id}: {e}")
            return False
    
    async def process_quarter_batch_for_memory(self, group_id: int, quarter_messages: List[Message]) -> bool:
        """
        Process a quarter batch of messages to update memory.
        This is called when the message deque reaches max capacity and we need to process the oldest 1/4.
        Returns True if memory update was successful.
        """
        if not quarter_messages:
            return False
            
        print(f"Processing quarter batch of {len(quarter_messages)} messages for group {group_id}")
        
        try:
            # Check if we have existing memory
            current_memory = await self.get_group_memory(group_id)
            
            if not current_memory:
                return await self._generate_initial_memory_from_messages(group_id, quarter_messages)
            else:
                return await self.update_group_memory(group_id, quarter_messages)
                
        except Exception as e:
            print(f"Error processing quarter batch for group {group_id}: {e}")
            return False
    
    async def _generate_initial_memory_from_messages(self, group_id: int, messages: List[Message]) -> bool:
        """Generate initial memory from a list of Message objects."""
        try:
            senders_text = self._get_senders_text(messages)
            
            # Get sequence numbers (using approximation since we don't have real_seq in Message objects)
            first_seq = str(int(messages[0].timestamp.timestamp()))
            last_seq = str(int(messages[-1].timestamp.timestamp()))
            
            # Process first batch of messages (or all if fewer)
            first_batch = messages[:self.INITIAL_BATCH_SIZE] if len(messages) >= self.INITIAL_BATCH_SIZE else messages
            first_batch_formatted = self._format_messages_for_memory_from_messages(first_batch)
            
            if not first_batch_formatted.strip():
                print(f"No valid text content found in messages for group {group_id}")
                return False
            
            # Generate initial memory
            initial_prompt = self._get_initial_memory_prompt(first_batch_formatted, senders_text)
            
            memory_content = await self._generate_memory_with_llm(initial_prompt)
            
            if not memory_content:
                print(f"Failed to generate initial memory for group {group_id}")
                return False
            
            # Process remaining messages if any
            if len(messages) > self.INITIAL_BATCH_SIZE:
                remaining_batch = messages[self.INITIAL_BATCH_SIZE:]
                remaining_batch_formatted = self._format_messages_for_memory_from_messages(remaining_batch)
                
                if remaining_batch_formatted.strip():
                    update_prompt = self._get_update_memory_prompt(memory_content,
                                                                   remaining_batch_formatted, senders_text)
                    
                    updated_memory = await self._generate_memory_with_llm(update_prompt)
                    if updated_memory:
                        memory_content = updated_memory
            
            # Save memory to file
            memory_file = self._get_memory_file_path(group_id, first_seq, last_seq)
            with open(memory_file, 'w', encoding='utf-8') as f:
                f.write(memory_content)
            
            # Cache the memory and mark as initialized
            self.group_memories[group_id] = Memory(group_id, first_seq, last_seq, memory_content)
            self.group_memory_initialized[group_id] = True
            self.group_message_counts[group_id] = len(messages)
            
            print(f"Successfully generated initial memory for group {group_id}")
            return True
            
        except Exception as e:
            print(f"Error generating initial memory for group {group_id}: {e}")
            return False
    
    def _format_messages_for_memory_from_messages(self, messages: List[Message]) -> str:
        """Format Message objects for memory processing."""
        formatted_messages = []
        
        for msg in messages:
            # Extract text content from message segments
            text_parts = []
            for segment in msg.content:
                if hasattr(segment, 'type') and segment.type == "text":
                    text_parts.append(segment.data.text)
                elif hasattr(segment, 'type') and segment.type == "at":
                    text_parts.append(f"@{segment.data.qq}")
                elif hasattr(segment, 'type') and segment.type == "image":
                    text_parts.append("[图片]")
            
            message_text = "".join(text_parts).strip()
            if message_text:
                timestamp = msg.timestamp.strftime('%m-%d %H:%M')
                user_info = str(msg.user_id)
                formatted_messages.append(f"({timestamp}) {user_info}: {message_text}")
        
        return "\n".join(formatted_messages)
    
    async def update_group_memory(self, group_id: int, new_messages: List[Message]) -> bool:
        """
        Update group memory with new messages.
        Returns True if update was successful.
        """
        try:
            # Load existing memory
            current_memory = await self.get_group_memory(group_id)
            if not current_memory:
                # If no memory exists, try to initialize it first
                if not await self.initialize_group_memory(group_id):
                    return False
                #TODO: 这里在做什么?
                current_memory = await self.get_group_memory(group_id)
            
            # Format new messages
            formatted_new_messages = []
            for msg in new_messages:
                # Convert Message to a format similar to GroupHistoryMessageEvent
                text_parts = []
                for segment in msg.content:
                    if hasattr(segment, 'type') and segment.type == "text":
                        text_parts.append(segment.data.text)
                    elif hasattr(segment, 'type') and segment.type == "at":
                        text_parts.append(f"@{segment.data.qq}")
                    elif hasattr(segment, 'type') and segment.type == "image":
                        text_parts.append("[图片]")
                
                message_text = "".join(text_parts).strip()
                if message_text:
                    timestamp = msg.timestamp.strftime('%m-%d %H:%M')
                    user_info = str(msg.user_id)
                    formatted_new_messages.append(f"({timestamp}) {user_info}: {message_text}")
            
            if not formatted_new_messages:
                return True  # No valid messages to process
            
            new_messages_text = "\n".join(formatted_new_messages)
            
            senders_text = self._get_senders_text(new_messages)
            
            # Generate updated memory
            update_prompt = self._get_update_memory_prompt(current_memory, new_messages_text, senders_text)
            
            updated_memory = await self._generate_memory_with_llm(update_prompt)
            
            if not updated_memory:
                print(f"Failed to generate updated memory for group {group_id}")
                return False
            
            # Get the latest memory file to update sequence numbers
            latest_file = self._get_latest_memory_file(group_id)
            if latest_file:
                first_seq, _ = self._extract_seq_from_filename(latest_file)
                # Remove old file

                #TODO: delete or not?
                # latest_file.unlink()
            else:
                first_seq = new_messages[0].real_seq if new_messages and hasattr(new_messages[0], 'real_seq') else "0"
            
            # Use the sequence from the last message
            last_seq = "0"  # Default fallback
            if new_messages:
                # We don't have real_seq in Message objects, so we'll use timestamp as approximation
                last_seq = str(int(new_messages[-1].timestamp.timestamp()))
            
            # Save updated memory
            memory_file = self._get_memory_file_path(group_id, first_seq, last_seq)
            with open(memory_file, 'w', encoding='utf-8') as f:
                f.write(updated_memory)
            
            # Update cache
            self.group_memories[group_id] = Memory(group_id, first_seq, last_seq, updated_memory)
            
            print(f"Successfully updated memory for group {group_id}")
            return True
            
        except Exception as e:
            print(f"Error updating memory for group {group_id}: {e}")
            return False
    
    def get_cached_memory(self, group_id: int) -> Optional[Memory]:
        """
        Get the cached memory for a group without any I/O operations.
        This is non-blocking and safe to use in response paths.
        """
        return self.group_memories.get(group_id)
    
    async def get_group_memory(self, group_id: int) -> Optional[Memory]:
        """
        Get the current memory for a group.
        This method can perform I/O and should be used in background tasks.
        """
        # Check cache first
        if group_id in self.group_memories:
            return self.group_memories[group_id]
        
        # Try to load from file
        memory_file = self._get_latest_memory_file(group_id)
        if memory_file and memory_file.exists():
            try:
                with open(memory_file, 'r', encoding='utf-8') as f:
                    memory_content = f.read().strip()
                    memory = Memory(
                        group_id,
                        *self._extract_seq_from_filename(memory_file),
                        memory_content
                    )
                    self.group_memories[group_id] = memory
                    return memory
            except Exception as e:
                print(f"Error reading memory file for group {group_id}: {e}")
        
        return None
    
    # These methods are no longer needed with the new unified message management
    
    def is_memory_initialized(self, group_id: int) -> bool:
        """Check if memory has been initialized for a group."""
        return self.group_memory_initialized.get(group_id, False)
    
    def is_accumulating_messages(self, group_id: int) -> bool:
        """Check if a group is in message accumulation mode (waiting for initial memory generation)."""
        # With the new design, we consider a group "accumulating" if memory is not yet initialized
        return not self.group_memory_initialized.get(group_id, False)
    
    def get_accumulated_message_count(self, group_id: int) -> int:
        """This method is deprecated with the new unified message storage design."""
        # Return 0 since we no longer track this separately
        return 0
    
    async def preload_memory_async(self, group_id: int):
        """
        Preload memory for a group in the background.
        This ensures memory is available in cache for fast access during responses.
        """
        if group_id not in self.group_memories:
            memory = await self.get_group_memory(group_id)
            if memory:
                print(f"Preloaded memory for group {group_id}")
            else:
                print(f"No memory found to preload for group {group_id}")
