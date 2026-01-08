from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha1
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import chromadb

from qqbot.config_loader import settings
from qqbot.models import AtMessageSegment, Message
from qqbot.services import GeminiService

logger = logging.getLogger(__name__)

_NOTE_ID_NAMESPACE = uuid.UUID("a4a7b21e-6c62-4a14-9e9b-6df9ab1df3e1")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (_project_root() / path).resolve()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse_iso(dt: Optional[str]) -> Optional[datetime]:
    if not dt:
        return None
    try:
        # Python 3.13: fromisoformat supports offsets.
        parsed = datetime.fromisoformat(dt)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except Exception:
        return None


def _stable_note_id(group_id: str, note_key: str) -> str:
    return str(uuid.uuid5(_NOTE_ID_NAMESPACE, f"{group_id}|{note_key}"))


def _safe_collection_name(prefix: str, group_id: str) -> str:
    base = f"{prefix}__g{group_id}"
    if 3 <= len(base) <= 63:
        return base
    digest = sha1(group_id.encode("utf-8")).hexdigest()[:10]
    base = f"{prefix}__g{digest}"
    return base[:63]


def _normalize_text(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ")
    return " ".join(text.split()).strip()


@dataclass(frozen=True)
class BufferedMessage:
    timestamp: datetime
    user_id: str
    display_name: str
    text: str

    def signature(self) -> str:
        # Seconds resolution is enough for de-duping history replays.
        return f"{int(self.timestamp.timestamp())}:{self.user_id}:{self.text}"


@dataclass
class _GroupBufferState:
    ref_messages: Deque[BufferedMessage]
    pending_messages: Deque[BufferedMessage]
    recent_signatures: Deque[str]
    signature_set: set[str]
    flush_task: Optional[asyncio.Task] = None


@dataclass(frozen=True)
class GroupMemoryNote:
    note_id: str
    group_id: str
    note_key: str
    type: str
    subject_user_ids: List[str]
    summary: str
    details: List[str]
    tags: List[str]
    importance: int
    confidence: float
    evidence: List[str]
    created_at: str
    updated_at: str
    expire_at: Optional[str] = None


class SQLiteNoteStore:
    def __init__(self, sqlite_path: Path):
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(sqlite_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_schema()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _init_schema(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS notes (
                  note_id TEXT PRIMARY KEY,
                  group_id TEXT NOT NULL,
                  note_key TEXT NOT NULL,
                  type TEXT NOT NULL,
                  subject_user_ids TEXT NOT NULL,
                  summary TEXT NOT NULL,
                  details TEXT NOT NULL,
                  tags TEXT NOT NULL,
                  importance INTEGER NOT NULL,
                  confidence REAL NOT NULL,
                  evidence TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  expire_at TEXT,
                  UNIQUE (group_id, note_key)
                );
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_notes_group_type ON notes(group_id, type);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_notes_group_expire ON notes(group_id, expire_at);"
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS user_directory (
                  group_id TEXT NOT NULL,
                  user_id TEXT NOT NULL,
                  display_name TEXT NOT NULL,
                  last_seen_at TEXT NOT NULL,
                  PRIMARY KEY (group_id, user_id)
                );
                """
            )
            self._conn.commit()

    def upsert_user_display_name(self, group_id: str, user_id: str, display_name: str) -> None:
        now = _iso(_utc_now())
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO user_directory(group_id, user_id, display_name, last_seen_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(group_id, user_id) DO UPDATE SET
                  display_name=excluded.display_name,
                  last_seen_at=excluded.last_seen_at;
                """,
                (group_id, user_id, display_name, now),
            )
            self._conn.commit()

    def get_user_display_names(self, group_id: str, user_ids: Sequence[str]) -> Dict[str, str]:
        if not user_ids:
            return {}
        placeholders = ",".join("?" for _ in user_ids)
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT user_id, display_name FROM user_directory
                WHERE group_id=? AND user_id IN ({placeholders});
                """,
                (group_id, *user_ids),
            ).fetchall()
        return {str(r["user_id"]): str(r["display_name"]) for r in rows}

    def get_note_by_key(self, group_id: str, note_key: str) -> Optional[GroupMemoryNote]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM notes WHERE group_id=? AND note_key=?;",
                (group_id, note_key),
            ).fetchone()
        return self._row_to_note(row) if row else None

    def get_notes_by_ids(self, group_id: str, note_ids: Sequence[str]) -> List[GroupMemoryNote]:
        if not note_ids:
            return []
        placeholders = ",".join("?" for _ in note_ids)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM notes WHERE group_id=? AND note_id IN ({placeholders});",
                (group_id, *note_ids),
            ).fetchall()
        return [self._row_to_note(r) for r in rows]

    def list_notes(
        self,
        group_id: str,
        *,
        types: Optional[Sequence[str]] = None,
        min_importance: int = 1,
        include_expired: bool = False,
        limit: Optional[int] = None,
    ) -> List[GroupMemoryNote]:
        clauses = ["group_id=? AND importance>=?"]
        params: list[object] = [group_id, int(min_importance)]
        if types:
            placeholders = ",".join("?" for _ in types)
            clauses.append(f"type IN ({placeholders})")
            params.extend(types)
        if not include_expired:
            now = _iso(_utc_now())
            clauses.append("(expire_at IS NULL OR expire_at > ?)")
            params.append(now)
        sql = f"SELECT * FROM notes WHERE {' AND '.join(clauses)} ORDER BY updated_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(sql + ";", tuple(params)).fetchall()
        return [self._row_to_note(r) for r in rows]

    def upsert_notes(self, notes: Sequence[GroupMemoryNote]) -> None:
        if not notes:
            return
        with self._lock:
            cur = self._conn.cursor()
            cur.executemany(
                """
                INSERT INTO notes(
                  note_id, group_id, note_key, type, subject_user_ids, summary, details, tags,
                  importance, confidence, evidence, created_at, updated_at, expire_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(group_id, note_key) DO UPDATE SET
                  type=excluded.type,
                  subject_user_ids=excluded.subject_user_ids,
                  summary=excluded.summary,
                  details=excluded.details,
                  tags=excluded.tags,
                  importance=excluded.importance,
                  confidence=excluded.confidence,
                  evidence=excluded.evidence,
                  updated_at=excluded.updated_at,
                  expire_at=excluded.expire_at;
                """,
                [
                    (
                        n.note_id,
                        n.group_id,
                        n.note_key,
                        n.type,
                        json.dumps(n.subject_user_ids, ensure_ascii=False),
                        n.summary,
                        json.dumps(n.details, ensure_ascii=False),
                        json.dumps(n.tags, ensure_ascii=False),
                        int(n.importance),
                        float(n.confidence),
                        json.dumps(n.evidence, ensure_ascii=False),
                        n.created_at,
                        n.updated_at,
                        n.expire_at,
                    )
                    for n in notes
                ],
            )
            self._conn.commit()

    def delete_by_keys(self, group_id: str, note_keys: Sequence[str]) -> List[str]:
        if not note_keys:
            return []
        placeholders = ",".join("?" for _ in note_keys)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT note_id, note_key FROM notes WHERE group_id=? AND note_key IN ({placeholders});",
                (group_id, *note_keys),
            ).fetchall()
            self._conn.execute(
                f"DELETE FROM notes WHERE group_id=? AND note_key IN ({placeholders});",
                (group_id, *note_keys),
            )
            self._conn.commit()
        return [str(r["note_id"]) for r in rows]

    def _row_to_note(self, row: sqlite3.Row) -> GroupMemoryNote:
        return GroupMemoryNote(
            note_id=str(row["note_id"]),
            group_id=str(row["group_id"]),
            note_key=str(row["note_key"]),
            type=str(row["type"]),
            subject_user_ids=list(json.loads(row["subject_user_ids"])),
            summary=str(row["summary"]),
            details=list(json.loads(row["details"])),
            tags=list(json.loads(row["tags"])),
            importance=int(row["importance"]),
            confidence=float(row["confidence"]),
            evidence=list(json.loads(row["evidence"])),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            expire_at=str(row["expire_at"]) if row["expire_at"] else None,
        )


class ChromaIndex:
    def __init__(self, chroma_path: Path, *, collection_prefix: str):
        chroma_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(chroma_path))
        self._collection_prefix = collection_prefix

    def _collection(self, group_id: str):
        name = _safe_collection_name(self._collection_prefix, group_id)
        return self._client.get_or_create_collection(name=name)

    def upsert(
        self,
        group_id: str,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Dict[str, object]],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if not ids:
            return
        col = self._collection(group_id)
        col.upsert(ids=list(ids), documents=list(documents), metadatas=list(metadatas), embeddings=list(embeddings))

    def delete(self, group_id: str, *, ids: Sequence[str]) -> None:
        if not ids:
            return
        col = self._collection(group_id)
        col.delete(ids=list(ids))

    def query(self, group_id: str, *, embedding: Sequence[float], k: int) -> Dict:
        col = self._collection(group_id)
        return col.query(query_embeddings=[list(embedding)], n_results=int(k), include=["metadatas", "distances", "documents", "ids"])


class GroupMemoryManager:
    """
    Group-scoped, batch-updated long-term memory manager.

    - Identity is user_id (stable); display_name is presentation-only.
    - Per group: every `flush_every_messages` new messages triggers an async memory update.
      Each update sees `reference_messages + flush_every_messages` messages where the
      reference window is context-only (MUST NOT be used to write memories).
    - Notes are persisted (SQLite) and indexed per group (Chroma collection).
    """

    _ALLOWED_TYPES = {
        "group_profile",
        "user_profile",
        "decision",
        "task",
        "resource",
        "topic_summary",
    }

    def __init__(self, gemini: GeminiService):
        self._gemini = gemini
        self._enabled = bool(settings.get("agentic_memory.enabled", False))
        self._ingest_mode = str(settings.get("agentic_memory.ingest_mode", "all")).lower()
        self._include_bot_messages = bool(
            settings.get("agentic_memory.include_bot_messages", False)
        )
        self._flush_every = int(settings.get("agentic_memory.flush_every_messages", 400))
        self._ref_count = int(settings.get("agentic_memory.reference_messages", 50))
        self._max_buffer = int(settings.get("agentic_memory.max_buffer_messages", 2000))
        self._message_max_chars = int(settings.get("agentic_memory.message_max_chars", 240))
        self._ref_message_max_chars = int(
            settings.get(
                "agentic_memory.reference_message_max_chars",
                min(self._message_max_chars, 120),
            )
        )
        self._batch_llm_model = str(
            settings.get("agentic_memory.llm_model", settings.get("tiny_model_name"))
        )
        self._embed_model = str(
            settings.get("agentic_memory.embedder_model", "models/text-embedding-004")
        )
        self._embed_dims = int(settings.get("agentic_memory.embedding_dims", 768))
        self._prompt_max_chars = int(settings.get("agentic_memory.prompt_max_chars", 1200))
        self._search_limit = int(settings.get("agentic_memory.search_limit", 8))
        self._min_importance_store = int(settings.get("agentic_memory.min_importance_to_store", 3))
        self._min_confidence_store = float(settings.get("agentic_memory.min_confidence_to_store", 0.6))
        self._min_importance_inject = int(settings.get("agentic_memory.min_importance_to_inject", 3))
        self._max_upserts_per_batch = int(settings.get("agentic_memory.max_upserts_per_batch", 20))
        self._max_evidence_per_note = int(settings.get("agentic_memory.max_evidence_per_note", 3))
        self._evidence_max_chars = int(settings.get("agentic_memory.evidence_max_chars", 160))
        self._snapshot_limit = int(settings.get("agentic_memory.snapshot_limit", 20))
        self._dedup_max = int(settings.get("agentic_memory.dedup_max", 5000))

        chroma_path = _resolve_path(str(settings.get("agentic_memory.chroma_path", "data/agentic_memory/chroma")))
        sqlite_path = _resolve_path(str(settings.get("agentic_memory.sqlite_path", "data/agentic_memory/notes.db")))
        collection_prefix = str(settings.get("agentic_memory.collection_name_prefix", "qqbot_agentic"))

        self._store = SQLiteNoteStore(sqlite_path)
        self._index = ChromaIndex(chroma_path, collection_prefix=collection_prefix)

        self._states: Dict[str, _GroupBufferState] = {}
        self._group_locks: Dict[str, asyncio.Lock] = {}
        self._background_sem = asyncio.Semaphore(
            int(settings.get("agentic_memory.background_concurrency", 1))
        )

        self._user_name_cache: Dict[Tuple[str, str], str] = {}

    async def close(self) -> None:
        for state in self._states.values():
            if state.flush_task and not state.flush_task.done():
                state.flush_task.cancel()
        await asyncio.sleep(0)
        self._store.close()

    def ingest_message(self, group_id: int, message: Message) -> None:
        if not self._enabled:
            return

        bot_id = str(settings.get("bot_qq_id"))
        if self._ingest_mode == "bot":
            mentioned_bot = any(
                isinstance(seg, AtMessageSegment) and str(seg.data.qq) == bot_id
                for seg in message.content
            )
            is_bot_msg = str(message.user_id) == bot_id
            if not mentioned_bot and not is_bot_msg:
                return
        if not self._include_bot_messages and str(message.user_id) == bot_id:
            return

        text, _ = message.get_formatted_text(vision_enabled=False)
        text = _normalize_text(text)
        if not text:
            return

        display_name = str(message.card) if message.card else str(message.nickname)
        if not display_name:
            display_name = "（未知）"

        group_id_str = str(group_id)
        user_id_str = str(message.user_id)

        cache_key = (group_id_str, user_id_str)
        if self._user_name_cache.get(cache_key) != display_name:
            self._user_name_cache[cache_key] = display_name
            try:
                self._store.upsert_user_display_name(group_id_str, user_id_str, display_name)
            except Exception:
                logger.exception("agentic_memory: failed to upsert user directory")

        max_chars = max(1, self._message_max_chars)
        text = text[:max_chars]

        buffered = BufferedMessage(
            timestamp=message.timestamp,
            user_id=user_id_str,
            display_name=display_name,
            text=text,
        )
        self._enqueue(group_id_str, buffered)

    async def build_memory_prompt(
        self,
        group_id: int,
        messages: Deque[Message],
        *,
        mention_user_id: Optional[str] = None,
    ) -> Optional[str]:
        if not self._enabled:
            return None

        group_id_str = str(group_id)
        mention_user_id = str(mention_user_id) if mention_user_id is not None else None

        latest = list(messages)[-1] if messages else None
        query_text = self._build_query_text(messages)

        group_profile = self._store.get_note_by_key(group_id_str, "group_profile")

        user_ids_to_include: List[str] = []
        if mention_user_id:
            user_ids_to_include.append(mention_user_id)

        mentioned_user_ids = set()
        if latest:
            for seg in latest.content:
                if isinstance(seg, AtMessageSegment):
                    mentioned_user_ids.add(str(seg.data.qq))
        for uid in sorted(mentioned_user_ids):
            if uid not in user_ids_to_include:
                user_ids_to_include.append(uid)

        user_profiles: List[GroupMemoryNote] = []
        for uid in user_ids_to_include:
            note = self._store.get_note_by_key(group_id_str, f"user_profile:{uid}")
            if note:
                user_profiles.append(note)

        retrieved_notes: List[GroupMemoryNote] = []
        try:
            retrieved_notes = await self._search_notes(group_id_str, query_text, limit=self._search_limit)
        except Exception:
            logger.exception("agentic_memory: retrieval failed; falling back to recent notes")
            retrieved_notes = self._store.list_notes(
                group_id_str,
                min_importance=self._min_importance_inject,
                limit=self._search_limit,
            )

        exclude_keys = {"group_profile"}
        exclude_keys.update({n.note_key for n in user_profiles})
        retrieved_notes = [n for n in retrieved_notes if n.note_key not in exclude_keys]

        display_names = self._build_display_name_map(group_id_str, messages, user_profiles, retrieved_notes)

        parts: list[str] = []
        parts.append(
            "【群长期记忆（仅供参考）】\n"
            "若与当前聊天冲突，以当前聊天为准；不确定就向群里确认；回复中不要提及内部 uid。\n"
        )

        if group_profile:
            parts.append("群信息/规则/偏好:\n")
            parts.append(f"- {group_profile.summary}\n")

        if user_profiles:
            parts.append("成员画像:\n")
            for note in user_profiles:
                name = display_names.get(note.subject_user_ids[0], "（未知）") if note.subject_user_ids else "（未知）"
                parts.append(f"- {name}: {note.summary}\n")

        if retrieved_notes:
            parts.append("相关记忆:\n")
            for note in retrieved_notes:
                parts.append(f"- ({note.type}) {note.summary}\n")

        prompt = "".join(parts).strip()
        if len(prompt) > self._prompt_max_chars:
            prompt = self._trim_to_chars(prompt, self._prompt_max_chars)
        return prompt

    def _build_display_name_map(
        self,
        group_id: str,
        messages: Deque[Message],
        user_profiles: Sequence[GroupMemoryNote],
        retrieved: Sequence[GroupMemoryNote],
    ) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for msg in messages:
            name = str(msg.card) if msg.card else str(msg.nickname)
            if name:
                mapping[str(msg.user_id)] = name

        needed_ids: set[str] = set()
        for note in user_profiles:
            needed_ids.update(note.subject_user_ids)
        for note in retrieved:
            needed_ids.update(note.subject_user_ids)
        missing = [uid for uid in needed_ids if uid not in mapping]
        if missing:
            mapping.update(self._store.get_user_display_names(group_id, missing))
        return mapping

    def _build_query_text(self, messages: Deque[Message]) -> str:
        texts: list[str] = []
        for msg in list(messages)[-12:]:
            t, _ = msg.get_formatted_text(vision_enabled=False)
            t = _normalize_text(t)
            if t:
                texts.append(t)
        return " ".join(texts).strip()[:800]

    def _trim_to_chars(self, text: str, max_chars: int) -> str:
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        truncated = text[:max_chars]
        last_newline = truncated.rfind("\n")
        if last_newline > max_chars * 0.7:
            return truncated[:last_newline].rstrip() + "\n…"
        return truncated.rstrip() + "…"

    def _get_state(self, group_id: str) -> _GroupBufferState:
        state = self._states.get(group_id)
        if state is None:
            state = _GroupBufferState(
                ref_messages=deque(maxlen=self._ref_count),
                pending_messages=deque(),
                recent_signatures=deque(),
                signature_set=set(),
            )
            self._states[group_id] = state
        return state

    def _enqueue(self, group_id: str, msg: BufferedMessage) -> None:
        state = self._get_state(group_id)

        sig = msg.signature()
        if sig in state.signature_set:
            return
        if len(state.recent_signatures) >= self._dedup_max:
            old = state.recent_signatures.popleft()
            state.signature_set.discard(old)
        state.recent_signatures.append(sig)
        state.signature_set.add(sig)

        state.pending_messages.append(msg)
        while len(state.pending_messages) > self._max_buffer:
            dropped = state.pending_messages.popleft()
            state.signature_set.discard(dropped.signature())

        if len(state.pending_messages) >= self._flush_every:
            if state.flush_task is None or state.flush_task.done():
                state.flush_task = asyncio.create_task(self._flush_loop(group_id))

    async def _flush_loop(self, group_id: str) -> None:
        async with self._background_sem:
            lock = self._group_locks.get(group_id)
            if lock is None:
                lock = asyncio.Lock()
                self._group_locks[group_id] = lock

            async with lock:
                state = self._states.get(group_id)
                if state is None:
                    return
                while len(state.pending_messages) >= self._flush_every:
                    ref = list(state.ref_messages)[-self._ref_count:]
                    update = list(state.pending_messages)[: self._flush_every]
                    if not update:
                        return

                    try:
                        await self._run_one_flush(group_id, ref, update)
                    except Exception:
                        logger.exception("agentic_memory: flush failed (group_id=%s)", group_id)
                        return

                    for _ in range(self._flush_every):
                        if state.pending_messages:
                            state.pending_messages.popleft()
                        else:
                            break

                    window = ref + update
                    state.ref_messages.clear()
                    state.ref_messages.extend(window[-self._ref_count :])

    async def _run_one_flush(
        self, group_id: str, ref_msgs: Sequence[BufferedMessage], update_msgs: Sequence[BufferedMessage]
    ) -> None:
        snapshot = await self._build_memory_snapshot(group_id, update_msgs)
        patch = await self._memory_update_llm(group_id, ref_msgs, update_msgs, snapshot)
        await self._apply_patch(group_id, update_msgs, patch)

    async def _build_memory_snapshot(self, group_id: str, update_msgs: Sequence[BufferedMessage]) -> str:
        group_profile = self._store.get_note_by_key(group_id, "group_profile")

        involved_counts: Dict[str, int] = {}
        for m in update_msgs:
            involved_counts[m.user_id] = involved_counts.get(m.user_id, 0) + 1
        involved_users = [uid for uid, _ in sorted(involved_counts.items(), key=lambda kv: kv[1], reverse=True)]

        user_profiles: List[GroupMemoryNote] = []
        for uid in involved_users[: min(20, len(involved_users))]:
            note = self._store.get_note_by_key(group_id, f"user_profile:{uid}")
            if note:
                user_profiles.append(note)

        query = " ".join(m.text for m in update_msgs[-30:])[:800]
        relevant_notes: List[GroupMemoryNote] = []
        try:
            relevant_notes = await self._search_notes(group_id, query, limit=self._snapshot_limit)
        except Exception:
            relevant_notes = self._store.list_notes(
                group_id,
                min_importance=self._min_importance_store,
                limit=self._snapshot_limit,
            )

        lines: list[str] = []
        if group_profile:
            lines.append("[existing group_profile]")
            lines.append(f"- key={group_profile.note_key} summary={group_profile.summary}")

        if user_profiles:
            lines.append("[existing user_profile]")
            for note in user_profiles:
                uid = note.subject_user_ids[0] if note.subject_user_ids else "?"
                lines.append(f"- uid={uid} summary={note.summary}")

        other = [n for n in relevant_notes if n.type not in {"group_profile", "user_profile"}]
        if other:
            lines.append("[existing relevant notes]")
            for note in other[: self._snapshot_limit]:
                lines.append(f"- ({note.type}) key={note.note_key} summary={note.summary}")

        return "\n".join(lines).strip()

    async def _memory_update_llm(
        self,
        group_id: str,
        ref_msgs: Sequence[BufferedMessage],
        update_msgs: Sequence[BufferedMessage],
        snapshot: str,
    ) -> Dict:
        update_count = len(update_msgs)
        speaker_map: Dict[str, str] = {}
        for m in list(ref_msgs) + list(update_msgs):
            speaker_map[m.user_id] = m.display_name
        speakers = "\n".join([f"- uid={uid} name={name}" for uid, name in speaker_map.items()])

        ref_lines: list[str] = []
        for i, m in enumerate(ref_msgs[-self._ref_count :], start=1):
            text = m.text[: self._ref_message_max_chars]
            ref_lines.append(
                f"R{i:02d} {m.timestamp.strftime('%Y-%m-%d %H:%M')} uid={m.user_id} name={m.display_name}: {text}"
            )

        upd_lines: list[str] = []
        for i, m in enumerate(update_msgs[: self._flush_every], start=1):
            upd_lines.append(
                f"U{i:03d} {m.timestamp.strftime('%Y-%m-%d %H:%M')} uid={m.user_id} name={m.display_name}: {m.text}"
            )

        system_instruction = (
            "你是群聊长期记忆管理器。你的任务是从 UPDATE 区消息中提取高价值长期记忆，并输出可执行的 JSON 变更集。\n"
            "硬性规则：\n"
            "1) REFERENCE 区仅用于理解上下文，禁止从 REFERENCE 提取/更新任何事实或记忆；证据也禁止引用 REFERENCE。\n"
            f"2) 只有 UPDATE 区允许写入；所有 evidence_indices 必须来自 UPDATE（范围 1..{update_count}）。\n"
            "3) 只记录长期可复用/可验证/可行动的信息；不确定就不记（宁缺毋滥）。\n"
            f"4) 每条 note 必须原子化，按语义实体/事件聚合；禁止把 {update_count} 条消息合成一个 note；summary 以概括为主。\n"
            "5) 与人相关信息必须绑定 user_id；昵称只是展示，可变。\n"
            "6) 记忆仅对当前 group_id 生效，不得跨群泛化。\n"
            "7) summary/details/tags 中不要出现 user_id/uid 等内部标识；身份用 subject_user_ids 与 note_key 表达。\n"
            "输出必须为严格 JSON，不要使用 Markdown 代码块。\n"
        )

        prompt = (
            f"group_id={group_id}\n\n"
            "speaker_directory (id stable; name can change):\n"
            f"{speakers}\n\n"
            "existing_memory_snapshot:\n"
            f"{snapshot or '(empty)'}\n\n"
            "messages:\n"
            "[REFERENCE_ONLY]\n"
            + ("\n".join(ref_lines) if ref_lines else "(empty)")
            + "\n\n[UPDATE]\n"
            + "\n".join(upd_lines)
            + "\n\n"
            "Return JSON with schema:\n"
            "{\n"
            '  "batch_summary": "one short sentence",\n'
            '  "upserts": [\n'
            "    {\n"
            '      "type": "group_profile|user_profile|decision|task|resource|topic_summary",\n'
            '      "note_key": "stable key within this group",\n'
            '      "subject_user_ids": ["..."],\n'
            '      "summary": "<=400 chars, summary-first",\n'
            '      "details": ["short bullet", "..."],\n'
            '      "tags": ["keyword", "..."],\n'
            '      "importance": 1,\n'
            '      "confidence": 0.0,\n'
            '      "ttl_days": 7,\n'
            '      "evidence_indices": [1, 2]\n'
            "    }\n"
            "  ],\n"
            '  "deletes": ["note_key"]\n'
            "}\n"
            "Constraints:\n"
            f"- evidence_indices must be from UPDATE only (1..{update_count}).\n"
            "- For type=group_profile, note_key MUST be 'group_profile'.\n"
            "- For type=user_profile, subject_user_ids MUST have exactly 1 user_id and note_key MUST be 'user_profile:<user_id>'.\n"
            "- Do not put user_id/uid into summary/details; use subject_user_ids + note_key instead.\n"
            "- Prefer <= 20 upserts per batch.\n"
        )

        try:
            return await self._gemini.generate_json(
                prompt,
                model_name=self._batch_llm_model,
                system_instruction=system_instruction,
                max_output_tokens=2048,
                temperature=0.1,
            )
        except Exception:
            logger.exception("agentic_memory: LLM memory update call failed")
            return {}

    async def _apply_patch(
        self, group_id: str, update_msgs: Sequence[BufferedMessage], patch: Dict
    ) -> None:
        deletes = patch.get("deletes", []) if isinstance(patch, dict) else []
        if not isinstance(deletes, list):
            deletes = []
        delete_keys = [str(k).strip() for k in deletes if str(k).strip()]

        deleted_ids: List[str] = []
        if delete_keys:
            deleted_ids = self._store.delete_by_keys(group_id, delete_keys)
            try:
                self._index.delete(group_id, ids=deleted_ids)
            except Exception:
                logger.exception("agentic_memory: failed deleting from chroma (group_id=%s)", group_id)

        upserts = patch.get("upserts", []) if isinstance(patch, dict) else []
        if not isinstance(upserts, list):
            upserts = []
        if len(upserts) > self._max_upserts_per_batch:
            upserts = upserts[: self._max_upserts_per_batch]

        now = _utc_now()

        notes_to_upsert: List[GroupMemoryNote] = []
        embed_texts: List[str] = []
        metadatas: List[Dict[str, object]] = []
        note_ids: List[str] = []
        documents: List[str] = []

        for raw in upserts:
            if not isinstance(raw, dict):
                continue
            note_type = str(raw.get("type", "")).strip()
            if note_type not in self._ALLOWED_TYPES:
                continue

            subject_user_ids = raw.get("subject_user_ids", [])
            if subject_user_ids is None:
                subject_user_ids = []
            if not isinstance(subject_user_ids, list):
                subject_user_ids = []
            subject_user_ids = [str(x).strip() for x in subject_user_ids if str(x).strip()]

            note_key = str(raw.get("note_key", "")).strip()
            if note_type == "group_profile":
                note_key = "group_profile"
                subject_user_ids = []
            if note_type == "user_profile":
                if len(subject_user_ids) != 1:
                    continue
                note_key = f"user_profile:{subject_user_ids[0]}"

            if not note_key:
                continue

            summary = _normalize_text(str(raw.get("summary", "")).strip())
            if not summary:
                continue
            if len(summary) > 420:
                summary = summary[:420].rstrip() + "…"

            details = raw.get("details", [])
            if details is None:
                details = []
            if not isinstance(details, list):
                details = []
            details = [_normalize_text(str(d))[:120] for d in details if _normalize_text(str(d))]
            details = details[:8]

            tags = raw.get("tags", [])
            if tags is None:
                tags = []
            if not isinstance(tags, list):
                tags = []
            tags = [_normalize_text(str(t))[:40] for t in tags if _normalize_text(str(t))]
            tags = tags[:8]

            try:
                importance = int(raw.get("importance", 1))
            except Exception:
                importance = 1
            importance = max(1, min(5, importance))

            try:
                confidence = float(raw.get("confidence", 0.0))
            except Exception:
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))

            if note_type not in {"group_profile", "user_profile"}:
                if importance < self._min_importance_store or confidence < self._min_confidence_store:
                    continue
            else:
                if confidence < 0.3:
                    continue

            ttl_days = raw.get("ttl_days")
            expire_at: Optional[str] = None
            if ttl_days is not None:
                try:
                    ttl_days_int = int(ttl_days)
                    if ttl_days_int > 0:
                        expire_at = _iso(now + timedelta(days=ttl_days_int))
                except Exception:
                    expire_at = None

            evidence_indices = raw.get("evidence_indices", [])
            if evidence_indices is None:
                evidence_indices = []
            if not isinstance(evidence_indices, list):
                evidence_indices = []
            cleaned_indices: List[int] = []
            for idx in evidence_indices:
                try:
                    i = int(idx)
                except Exception:
                    continue
                if 1 <= i <= len(update_msgs):
                    cleaned_indices.append(i)
            cleaned_indices = cleaned_indices[: self._max_evidence_per_note]

            evidence_lines: List[str] = []
            for i in cleaned_indices:
                m = update_msgs[i - 1]
                snippet = m.text[: self._evidence_max_chars]
                evidence_lines.append(f"U{i:03d} {m.display_name}: {snippet}")

            note_id = _stable_note_id(group_id, note_key)
            created_at = _iso(now)
            existing = self._store.get_note_by_key(group_id, note_key)
            if existing:
                created_at = existing.created_at

            note = GroupMemoryNote(
                note_id=note_id,
                group_id=group_id,
                note_key=note_key,
                type=note_type,
                subject_user_ids=subject_user_ids,
                summary=summary,
                details=details,
                tags=tags,
                importance=importance,
                confidence=confidence,
                evidence=evidence_lines,
                created_at=created_at,
                updated_at=_iso(now),
                expire_at=expire_at,
            )
            notes_to_upsert.append(note)
            note_ids.append(note.note_id)
            documents.append(note.summary)
            embed_texts.append(f"[{note.type}] {note.summary}")
            metadatas.append(
                {
                    "note_key": note.note_key,
                    "type": note.type,
                    "importance": note.importance,
                    "updated_at": note.updated_at,
                    "expire_at": note.expire_at or "",
                }
            )

        if notes_to_upsert:
            self._store.upsert_notes(notes_to_upsert)
            try:
                embeddings = await self._gemini.embed_texts(
                    embed_texts,
                    model_name=self._embed_model,
                    output_dimensionality=self._embed_dims,
                    task_type="RETRIEVAL_DOCUMENT",
                )
                if len(embeddings) != len(note_ids):
                    raise ValueError(
                        f"embed_texts length mismatch: got {len(embeddings)} expected {len(note_ids)}"
                    )
                self._index.upsert(
                    group_id,
                    ids=note_ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings,
                )
            except Exception:
                logger.exception("agentic_memory: embedding/index upsert failed (group_id=%s)", group_id)

        logger.info(
            "agentic_memory: flush applied group_id=%s upserts=%d deletes=%d",
            group_id,
            len(notes_to_upsert),
            len(delete_keys),
        )

    async def _search_notes(self, group_id: str, query: str, *, limit: int) -> List[GroupMemoryNote]:
        query = _normalize_text(query)
        if not query:
            return []
        embeddings = await self._gemini.embed_texts(
            [query],
            model_name=self._embed_model,
            output_dimensionality=self._embed_dims,
            task_type="RETRIEVAL_QUERY",
        )
        if not embeddings:
            return []
        results = self._index.query(group_id, embedding=embeddings[0], k=limit)
        ids = []
        if isinstance(results, dict) and results.get("ids") and results["ids"][0]:
            ids = [str(i) for i in results["ids"][0]]
        notes = self._store.get_notes_by_ids(group_id, ids)
        now = _utc_now()
        filtered: List[GroupMemoryNote] = []
        for n in notes:
            if n.importance < self._min_importance_inject:
                continue
            exp = _parse_iso(n.expire_at)
            if exp is not None and exp <= now:
                continue
            filtered.append(n)
        return filtered
