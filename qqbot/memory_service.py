import dataclasses
from email import message
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

@dataclasses.dataclass
class Memory:
    group_id: int
    first_seq: str
    last_seq: str
    memory: str

class GroupMemoryService:
    """
    Manages group-specific memory using LLM to compress and maintain chat history.
    """
    
    # Memory configuration constants
    MINIMUM_MESSAGES_FOR_MEMORY = 400
    MEMORY_UPDATE_INTERVAL = 400
    MEMORY_TARGET_LENGTH = 16000
    INITIAL_BATCH_SIZE = 3000
    
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
        self.memory_operation_locks = set()
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
        return f"""你是这个群聊的 AI 群员{settings.get('bot_name')}, id 是 {settings.get('bot_qq_id')}。
        你的任务是创建“群聊记忆”的第一个版本。这份记忆应该是一份详细的聊天记录摘要，目的是让你和群友聊天时, 保持上下文。

**你的目标：** 创建一份尽量长的摘要。

**操作指南:**
- 识别关键信息: 记录下群里做出的决定、约定、计划的活动或提出的重要问题。
- 捕捉群聊氛围: 如果有独特的群聊文化（例如：固定的玩笑、昵称）。
- 过滤无关内容: 忽略日常闲聊、打招呼等。
- 保持客观中立: 仅总结，不要添加主观臆断。
- 记录和自己相关的信息, 尤其是和群友的约定, 重要互动。如"以后你和我说话一直要扮演一只兔子"。这样的聊天内容必须要详细记录。
- 关于群员: 聊天记录中的群员使用 ID 显示，你可以参考下面的列表来了解他们的昵称。在生成记忆时，请优先使用群员 ID 来指代他们，以避免混淆。
- 记录时间: 记录重要事件, 约定, 转变等的时间。
**本次涉及到的群员:**
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
            你的任务是根据最新的消息更新“群聊记忆”。这份记忆应该是一份详细的聊天记录摘要，目的是让你和群友聊天时, 保持上下文。

**操作指南:**
- 识别关键信息: 记录下群里做出的决定、约定、计划的活动或提出的重要问题。
- 捕捉群聊氛围: 如果有独特的群聊文化（例如：固定的玩笑、昵称）。
- 过滤无关内容: 忽略日常闲聊、打招呼等。
- 保持客观中立: 仅总结，不要添加主观臆断。
- 记录和自己相关的信息, 尤其是和群友的约定, 重要互动。如"以后你和我说话一直要扮演一只兔子"。这样的聊天内容必须要详细记录。
- 关于群员: 聊天记录中的群员使用 ID 显示，你可以参考下面的列表来了解他们的昵称。在生成记忆时，请优先使用群员 ID 来指代他们，以避免混淆。
- 记录时间: 记录重要事件, 约定, 转变等的时间。
- 只增不删改: 除非信息过时或者不再重要, 否则尽量只增,不减, 以保证记忆尽可能长。

**本次涉及到的群员:**
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
                
                    async for chunk in self.gemini_service.generate_content_stream(messages, ""):
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
    
    async def _initialize_group_memory(self, group_id: int, messages: List[Message]) -> bool:
        """
        Initialize memory for a new group by fetching recent messages and processing them.
        Only generates memory if at least 400 messages are available.
        Returns True if initialization was successful.
        """
        
        print(f"Initializing memory for group {group_id}")
        
        try:
            
            if len(messages) == 0:
                print(f"No messages found for group {group_id}")
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
        
        finally:
            print(f"Released memory operation lock for group {group_id}")
    
    async def update_memory(self, group_id: int, messages: List[Message]) -> bool:
        if group_id in self.memory_operation_locks:
            return True
        self.memory_operation_locks.add(group_id)
        print(f"Acquired memory operation lock for group {group_id}")
        
        if not messages:
            return True
            
        print(f"Processing half batch of {len(messages)} messages for group {group_id}")
        try:
            # Check if we have existing memory
            current_memory = await self.get_group_memory(group_id)
            
            if not current_memory:
                await self._initialize_group_memory(group_id, messages)
                messages = messages[self.MEMORY_UPDATE_INTERVAL:] if len(messages) > self.MEMORY_UPDATE_INTERVAL else []
                # 重新获取更新后的memory信息
                current_memory = await self.get_group_memory(group_id)
            while messages:
                if current_memory:
                    messages = [msg for msg in messages if int(msg.real_seq) >= int(current_memory.last_seq)]
                if not messages:  # 如果过滤后没有消息需要处理，退出循环
                    break
                await self._update_group_memory(group_id, messages)
                current_memory = await self.get_group_memory(group_id)
                messages = messages[self.MEMORY_UPDATE_INTERVAL:] if len(messages) > self.MEMORY_UPDATE_INTERVAL else []
        except Exception as e:
            print(f"Error processing quarter batch for group {group_id}: {e}")
            return False
        finally:
            self.memory_operation_locks.discard(group_id)
            print(f"Released memory operation lock for group {group_id}")
    
    async def _generate_initial_memory_from_messages(self, group_id: int, messages: List[Message]) -> List[Message]:
        """Generate initial memory from a list of Message objects."""
        try:
            if not messages:
                print(f"No messages provided for initial memory generation for group {group_id}")
                return False
                
            senders_text = self._get_senders_text(messages)
            
            
            messages_formatted = ""
            messages_formatted = "\n".join(
                msg.get_formatted_text(False)[0] for msg in messages
            )
            if not messages_formatted.strip():
                print(f"No valid text content found in messages for group {group_id}")
                return False
            
            # Generate initial memory
            initial_prompt = self._get_initial_memory_prompt(messages_formatted, senders_text)
            
            memory_content = await self._generate_memory_with_llm(initial_prompt)
            
            if not memory_content:
                print(f"Failed to generate initial memory for group {group_id}")
                return False
            first_seq = str(int(messages[0].real_seq))
            last_seq = str(int(messages[-1].real_seq))
            # Save memory to file
            memory_file = self._get_memory_file_path(group_id, first_seq, last_seq)
            with open(memory_file, 'w', encoding='utf-8') as f:
                f.write(memory_content)
            
            # Cache the memory and mark as initialized
            self.group_memories[group_id] = Memory(group_id, first_seq, last_seq, memory_content)
            self.group_memory_initialized[group_id] = True
            
            print(f"Successfully generated initial memory for group {group_id}")
            return messages
            
        except Exception as e:
            print(f"Error generating initial memory for group {group_id}: {e}")
            return False
    
    async def _update_group_memory(self, group_id: int, messages: List[Message]) -> bool:
        """
        Update group memory with new messages.
        Returns True if update was successful.
        """
        try:
            # Load existing memory
            current_memory = await self.get_group_memory(group_id)
            
            # Filter out messages that are older than the last processed sequence
            # Use list comprehension to avoid modifying list while iterating
            current_messages = [msg for msg in messages if int(msg.real_seq) >= int(current_memory.last_seq)]
            
            # Format new messages
            new_messages_text = "\n".join(
                msg.get_formatted_text(False)[0] for msg in current_messages
            )
            senders_text = self._get_senders_text(current_messages)
            if not new_messages_text.strip():
                print(f"No new valid text content to update memory for group {group_id}")
                return True
            # Generate updated memory
            update_prompt = self._get_update_memory_prompt(current_memory.memory, new_messages_text, senders_text)
            
            updated_memory = await self._generate_memory_with_llm(update_prompt)
            
            if not updated_memory:
                print(f"Failed to generate updated memory for group {group_id}")
                return False
            # Use the sequence from the last message

                # We don't have real_seq in Message objects, so we'll use timestamp as approximation
            first_seq = str(int(current_messages[0].real_seq))
            last_seq = str(int(current_messages[-1].real_seq))

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
