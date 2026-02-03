"""
Database module for persisting group chat messages.
Each group has its own SQLite database file at ./data/{group_id}.db
"""

import json
import os
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict

from pydantic import TypeAdapter
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.exc import IntegrityError

from qqbot.models import Message, MessageSegment

Base = declarative_base()


class MessageRecord(Base):
    """SQLAlchemy model for storing messages."""
    __tablename__ = 'messages'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    real_seq = Column(Integer, unique=True, index=True)  # 严格递增，用于去重
    timestamp = Column(DateTime)
    user_id = Column(String(32))
    nickname = Column(String(128))
    card = Column(String(128), nullable=True)
    content = Column(Text)  # JSON序列化的 List[MessageSegment]


class GroupDatabase:
    """Database operations for a single group's messages."""
    
    # Class-level cache for database instances
    _instances: Dict[int, 'GroupDatabase'] = {}
    
    def __init__(self, group_id: int, data_dir: str = "./data"):
        self.group_id = group_id
        self.data_dir = data_dir
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        self.db_path = os.path.join(data_dir, f"{group_id}.db")
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            echo=False,
            connect_args={"check_same_thread": False}
        )
        
        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)
        
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    @classmethod
    def get_instance(cls, group_id: int, data_dir: str = "./data") -> 'GroupDatabase':
        """Get or create a GroupDatabase instance for the given group_id."""
        if group_id not in cls._instances:
            cls._instances[group_id] = cls(group_id, data_dir)
        return cls._instances[group_id]
    
    def _message_to_record(self, message: Message) -> MessageRecord:
        """Convert a Message dataclass to a MessageRecord ORM object."""
        # Serialize content to JSON
        content_dicts = []
        for segment in message.content:
            if hasattr(segment, 'model_dump'):
                content_dicts.append(segment.model_dump())
            else:
                # Fallback for non-pydantic objects
                content_dicts.append({"type": getattr(segment, 'type', 'unknown')})
        
        return MessageRecord(
            real_seq=message.real_seq,
            timestamp=message.timestamp,
            user_id=message.user_id,
            nickname=message.nickname,
            card=message.card,
            content=json.dumps(content_dicts, ensure_ascii=False)
        )
    
    def _record_to_message(self, record: MessageRecord) -> Message:
        """Convert a MessageRecord ORM object to a Message dataclass."""
        # Deserialize content from JSON
        content_dicts = json.loads(record.content)
        parsed_content = TypeAdapter(List[MessageSegment]).validate_python(content_dicts)
        
        return Message(
            timestamp=record.timestamp,
            real_seq=record.real_seq,
            user_id=record.user_id,
            nickname=record.nickname,
            card=record.card,
            content=parsed_content
        )
    
    def insert_message(self, message: Message) -> bool:
        """
        Insert a single message into the database.
        Returns True if inserted, False if duplicate (based on real_seq).
        """
        if message.real_seq is None:
            # Bot messages may not have real_seq, skip them or handle differently
            return False
            
        session: Session = self.SessionLocal()
        try:
            record = self._message_to_record(message)
            session.add(record)
            session.commit()
            return True
        except IntegrityError:
            # Duplicate real_seq, ignore
            session.rollback()
            return False
        finally:
            session.close()
    
    def insert_messages_bulk(self, messages: List[Message]) -> int:
        """
        Insert multiple messages into the database.
        Skips duplicates based on real_seq.
        Returns the number of messages actually inserted.
        """
        if not messages:
            return 0
        
        # Filter out messages without real_seq
        valid_messages = [m for m in messages if m.real_seq is not None]
        if not valid_messages:
            return 0
        
        session: Session = self.SessionLocal()
        inserted_count = 0
        
        try:
            for message in valid_messages:
                try:
                    record = self._message_to_record(message)
                    session.add(record)
                    session.flush()  # Flush to catch IntegrityError early
                    inserted_count += 1
                except IntegrityError:
                    session.rollback()
                    # Reopen session after rollback
                    session = self.SessionLocal()
            
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error inserting messages: {e}")
        finally:
            session.close()
        
        return inserted_count
    
    def get_recent_messages(self, limit: int = 100) -> List[Message]:
        """
        Get the most recent messages from the database.
        Returns messages ordered by real_seq ascending (oldest first).
        """
        session: Session = self.SessionLocal()
        try:
            # Get the most recent N messages, ordered by real_seq descending
            records = session.query(MessageRecord)\
                .order_by(MessageRecord.real_seq.desc())\
                .limit(limit)\
                .all()
            
            # Reverse to get ascending order (oldest first)
            records = list(reversed(records))
            
            return [self._record_to_message(r) for r in records]
        finally:
            session.close()
    
    def get_max_real_seq(self) -> Optional[int]:
        """
        Get the maximum real_seq in the database.
        Returns None if the database is empty.
        """
        session: Session = self.SessionLocal()
        try:
            from sqlalchemy import func
            result = session.query(func.max(MessageRecord.real_seq)).scalar()
            return result
        finally:
            session.close()
    
    def get_message_count(self) -> int:
        """Get the total number of messages in the database."""
        session: Session = self.SessionLocal()
        try:
            from sqlalchemy import func
            return session.query(func.count(MessageRecord.id)).scalar() or 0
        finally:
            session.close()
