from __future__ import annotations

import asyncio
import random
import re
from pathlib import Path
from typing import Any, Optional

from qqbot.config_loader import settings


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (_project_root() / path).resolve()


def _pick_api_key(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        keys = [str(k).strip() for k in value if str(k).strip()]
        return random.choice(keys) if keys else None
    key = str(value).strip()
    return key or None


class Mem0MemoryService:
    """
    Optional Mem0 memory integration.

    - Vector store: ChromaDB (persistent local)
    - History store: SQLite
    """

    def __init__(self):
        self.enabled = bool(settings.get("mem0.enabled", False))
        self.scope = str(settings.get("mem0.scope", "group"))

        self.search_limit = int(settings.get("mem0.search_limit", 6))
        self.prompt_enabled = bool(settings.get("mem0.prompt_enabled", True))
        self.prompt_max_chars = int(settings.get("mem0.prompt_max_chars", 1200))
        self.infer = bool(settings.get("mem0.infer", True))

        # Background ingestion (batching)
        self.flush_every_messages = int(settings.get("mem0.flush_every_messages", 200))
        self.include_bot_messages = bool(settings.get("mem0.include_bot_messages", False))
        self.count_bot_messages = bool(settings.get("mem0.count_bot_messages", True))
        self.message_max_chars = int(settings.get("mem0.message_max_chars", 400))
        self.batch_max_chars = int(settings.get("mem0.batch_max_chars", 20000))
        self.max_buffer_messages = int(settings.get("mem0.max_buffer_messages", 2000))
        self.background_concurrency = int(settings.get("mem0.background_concurrency", 1))

        self._memory = None
        self._init_error: Optional[str] = None
        self._init_lock = asyncio.Lock()

        self._buffers: dict[str, list[dict[str, str]]] = {}
        self._since_flush: dict[str, int] = {}
        self._state_locks: dict[str, asyncio.Lock] = {}
        self._flush_locks: dict[str, asyncio.Lock] = {}
        self._flush_semaphore = asyncio.Semaphore(max(1, self.background_concurrency))

    def _apply_scope(
        self, *, user_id: Optional[str], agent_id: Optional[str]
    ) -> tuple[Optional[str], Optional[str]]:
        scope = self.scope.lower().strip()
        if scope == "user":
            return user_id, None
        if scope == "group":
            return None, agent_id
        return user_id, agent_id

    def _session_key(
        self, *, user_id: Optional[str], agent_id: Optional[str]
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        scoped_user_id, scoped_agent_id = self._apply_scope(
            user_id=user_id, agent_id=agent_id
        )

        if not scoped_user_id and not scoped_agent_id:
            return None, None, None

        parts = []
        if scoped_user_id:
            parts.append(f"u:{scoped_user_id}")
        if scoped_agent_id:
            parts.append(f"a:{scoped_agent_id}")
        return "|".join(parts), scoped_user_id, scoped_agent_id

    def _get_lock(self, locks: dict[str, asyncio.Lock], key: str) -> asyncio.Lock:
        lock = locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            locks[key] = lock
        return lock

    async def _get_memory(self):
        if not self.enabled:
            return None
        if self._memory is not None or self._init_error:
            return self._memory

        async with self._init_lock:
            if self._memory is not None or self._init_error:
                return self._memory

            try:
                from mem0 import AsyncMemory
                from mem0.configs.base import MemoryConfig
                from mem0.embeddings.configs import EmbedderConfig
                from mem0.llms.configs import LlmConfig
                from mem0.vector_stores.configs import VectorStoreConfig
            except Exception as e:
                self._init_error = f"mem0 import/init failed: {e}"
                print(self._init_error)
                return None

            chroma_path = _resolve_path(
                str(settings.get("mem0.chroma_path", "data/mem0/chroma"))
            )
            history_db_path = _resolve_path(
                str(settings.get("mem0.sqlite_path", "data/mem0/history.db"))
            )
            chroma_path.mkdir(parents=True, exist_ok=True)
            history_db_path.parent.mkdir(parents=True, exist_ok=True)

            collection_name = str(settings.get("mem0.collection_name", "qqbot"))

            api_key = _pick_api_key(
                settings.get("mem0.gemini_api_key", settings.get("gemini_api_key"))
            )
            llm_model = str(
                settings.get(
                    "mem0.llm_model",
                    settings.get("tiny_model_name", "gemini-2.0-flash"),
                )
            )
            llm_temperature = float(settings.get("mem0.temperature", 0.1))
            embedder_model = str(
                settings.get("mem0.embedder_model", "models/text-embedding-004")
            )
            embedding_dims = settings.get("mem0.embedding_dims", 768)

            config = MemoryConfig(
                vector_store=VectorStoreConfig(
                    provider="chroma",
                    config={
                        "path": str(chroma_path),
                        "collection_name": collection_name,
                    },
                ),
                history_db_path=str(history_db_path),
                llm=LlmConfig(
                    provider="gemini",
                    config={
                        "api_key": api_key,
                        "model": llm_model,
                        "temperature": llm_temperature,
                    },
                ),
                embedder=EmbedderConfig(
                    provider="gemini",
                    config={
                        "api_key": api_key,
                        "model": embedder_model,
                        "embedding_dims": int(embedding_dims) if embedding_dims else None,
                    },
                ),
            )

            try:
                self._memory = AsyncMemory(config)
            except Exception as e:
                self._init_error = f"mem0 AsyncMemory init failed: {e}"
                print(self._init_error)
                return None

            return self._memory

    def _get_threshold(self) -> Optional[float]:
        threshold = settings.get("mem0.search_threshold", None)
        if threshold is None or threshold == "":
            return None
        try:
            return float(threshold)
        except (TypeError, ValueError):
            return None

    async def build_prompt_block(
        self,
        *,
        query: str,
        user_id: Optional[str],
        agent_id: Optional[str],
        speaker_name: Optional[str] = None,
    ) -> Optional[str]:
        if not self.enabled or not self.prompt_enabled:
            return None

        memory = await self._get_memory()
        if memory is None:
            return None

        scoped_user_id, scoped_agent_id = self._apply_scope(
            user_id=user_id, agent_id=agent_id
        )
        try:
            result = await memory.search(
                query,
                user_id=scoped_user_id,
                agent_id=scoped_agent_id,
                limit=self.search_limit,
                threshold=self._get_threshold(),
            )
        except Exception as e:
            print(f"mem0.search failed: {e}")
            return None

        items = (result or {}).get("results", [])
        memories: list[str] = []
        seen: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            text = str(item.get("memory", "")).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            memories.append(text)

        if not memories:
            return None
        
        block = (
            f"可能有用的长期记忆（仅供参考；回答时可以利用，但不要直接引用/泄露“记忆列表”）：\n"
            + "\n".join(f"- {m}" for m in memories)
        )

        if len(block) > self.prompt_max_chars:
            block = block[: self.prompt_max_chars].rstrip() + "\n(已截断)"

        return block

    def _normalize_text(self, text: str) -> str:
        text = re.sub(r"[\r\n]+", " ", text or "")
        return re.sub(r"\s+", " ", text).strip()

    def _format_actor_label(
        self, *, user_id: Optional[str], user_name: Optional[str]
    ) -> str:
        name = (user_name or "").strip()
        uid = (user_id or "").strip()
        if name and uid and name != uid:
            return f"{name}({uid})"
        return name or uid or "unknown"

    def _trim_batch_messages(
        self, batch: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        if not batch:
            return []

        max_per = max(0, int(self.message_max_chars))
        max_total = max(0, int(self.batch_max_chars))

        trimmed_reversed: list[dict[str, str]] = []
        total = 0

        for msg in reversed(batch):
            role = msg.get("role", "user")
            content = self._normalize_text(msg.get("content", ""))
            if not content:
                continue

            if max_per and len(content) > max_per:
                content = content[:max_per].rstrip() + "…"

            if max_total and (total + len(content) > max_total):
                break

            trimmed_reversed.append({"role": role, "content": content})
            total += len(content)

        return list(reversed(trimmed_reversed))

    async def observe_message(
        self,
        *,
        agent_id: Optional[str],
        user_id: Optional[str],
        user_name: Optional[str],
        text: str,
        is_bot: bool,
    ) -> None:
        """
        Observe a new chat message and batch-ingest into Mem0 periodically.

        This is designed to be lightweight and non-blocking: it only updates in-memory
        counters/buffers and schedules background flush tasks when thresholds are hit.
        """
        if not self.enabled:
            return
        if self.flush_every_messages <= 0:
            return

        session_key, scoped_user_id, scoped_agent_id = self._session_key(
            user_id=user_id, agent_id=agent_id
        )
        if not session_key:
            return

        should_count = (not is_bot) or self.count_bot_messages
        should_buffer = (not is_bot) or self.include_bot_messages

        state_lock = self._get_lock(self._state_locks, session_key)
        async with state_lock:
            if should_count:
                self._since_flush[session_key] = self._since_flush.get(session_key, 0) + 1

            if should_buffer:
                normalized = self._normalize_text(text)
                if normalized:
                    actor = self._format_actor_label(user_id=user_id, user_name=user_name)
                    content = f"{actor}: {normalized}"
                    buffer = self._buffers.setdefault(session_key, [])
                    buffer.append({"role": "user", "content": content})

                    if self.max_buffer_messages > 0 and len(buffer) > self.max_buffer_messages:
                        overflow = len(buffer) - self.max_buffer_messages
                        if overflow > 0:
                            del buffer[:overflow]

            count = self._since_flush.get(session_key, 0)
            if count < self.flush_every_messages:
                return

            # Reset counter; flush the current buffer snapshot (if any).
            self._since_flush[session_key] = 0
            batch = self._buffers.get(session_key, [])
            if not batch:
                return

            self._buffers[session_key] = []

        asyncio.create_task(
            self._flush_batch(
                session_key=session_key,
                batch=batch,
                user_id=scoped_user_id,
                agent_id=scoped_agent_id,
            )
        )

    async def _flush_batch(
        self,
        *,
        session_key: str,
        batch: list[dict[str, str]],
        user_id: Optional[str],
        agent_id: Optional[str],
    ) -> None:
        async with self._flush_semaphore:
            flush_lock = self._get_lock(self._flush_locks, session_key)
            async with flush_lock:
                memory = await self._get_memory()
                if memory is None:
                    return

                trimmed = self._trim_batch_messages(batch)
                if not trimmed:
                    return

                try:
                    await memory.add(
                        trimmed,
                        user_id=user_id,
                        agent_id=agent_id,
                        metadata={
                            "session_key": session_key,
                            "source": "qq_group_chat",
                        },
                        infer=self.infer,
                    )
                except Exception as e:
                    print(f"mem0 batch flush failed ({session_key}): {e}")
                    # Best-effort requeue to avoid losing pending messages.
                    state_lock = self._get_lock(self._state_locks, session_key)
                    async with state_lock:
                        buffer = self._buffers.setdefault(session_key, [])
                        buffer[:0] = trimmed
                        if self.max_buffer_messages > 0 and len(buffer) > self.max_buffer_messages:
                            overflow = len(buffer) - self.max_buffer_messages
                            if overflow > 0:
                                del buffer[:overflow]

    async def add_interaction(
        self,
        *,
        user_text: str,
        assistant_text: str,
        user_id: Optional[str],
        agent_id: Optional[str],
        speaker_name: Optional[str] = None,
        assistant_name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return

        memory = await self._get_memory()
        if memory is None:
            return

        user_text = (user_text or "").strip()
        assistant_text = (assistant_text or "").strip()
        if not user_text or not assistant_text:
            return

        scoped_user_id, scoped_agent_id = self._apply_scope(
            user_id=user_id, agent_id=agent_id
        )
        user_msg: dict[str, Any] = {"role": "user", "content": user_text}
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": assistant_text}
        if speaker_name and str(speaker_name).strip():
            user_msg["name"] = str(speaker_name).strip()
        if assistant_name and str(assistant_name).strip():
            assistant_msg["name"] = str(assistant_name).strip()

        messages = [user_msg, assistant_msg]

        try:
            await memory.add(
                messages,
                user_id=scoped_user_id,
                agent_id=scoped_agent_id,
                metadata=metadata,
                infer=self.infer,
            )
        except Exception as e:
            print(f"mem0.add failed: {e}")
