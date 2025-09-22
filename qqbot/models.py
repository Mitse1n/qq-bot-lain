from typing import List, Literal, Union, Annotated, Optional
from dataclasses import dataclass, field
from datetime import datetime
import time

from pydantic import BaseModel, Field, field_validator

from qqbot.config_loader import settings


class Sender(BaseModel):
    user_id: int
    nickname: str
    card: str
    role: str = ""


class TextData(BaseModel):
    text: str


class AtData(BaseModel):
    qq: str


class ReplyData(BaseModel):
    id: str


class ImageData(BaseModel):
    summary: str
    file: str
    sub_type: Optional[int] = None
    url: str
    file_size: Optional[str] = None


class JsonData(BaseModel):
    data: str


class FaceData(BaseModel):
    id: str
    chainCount: Optional[int] = None


class RecordData(BaseModel):
    file: str
    url: str
    file_size: Optional[str] = None


class VideoData(BaseModel):
    file: str
    url: str
    file_size: Optional[str] = None


class ForwardData(BaseModel):
    id: str


class TextMessageSegment(BaseModel):
    type: Literal["text"]
    data: TextData


class AtMessageSegment(BaseModel):
    type: Literal["at"]
    data: AtData


class ReplyMessageSegment(BaseModel):
    type: Literal["reply"]
    data: ReplyData


class ImageMessageSegment(BaseModel):
    type: Literal["image"]
    data: ImageData


class JsonMessageSegment(BaseModel):
    type: Literal["json"]
    data: JsonData


class FaceMessageSegment(BaseModel):
    type: Literal["face"]
    data: FaceData


class RecordMessageSegment(BaseModel):
    type: Literal["record"]
    data: RecordData


class VideoMessageSegment(BaseModel):
    type: Literal["video"]
    data: VideoData


class ForwardMessageSegment(BaseModel):
    type: Literal["forward"]
    data: ForwardData


MessageSegment = Annotated[
    Union[
        TextMessageSegment,
        AtMessageSegment,
        ImageMessageSegment,
        ReplyMessageSegment,
        JsonMessageSegment,
        FaceMessageSegment,
        RecordMessageSegment,
        VideoMessageSegment,
        ForwardMessageSegment,
    ],
    Field(discriminator="type"),
]


@dataclass
class Message:
    timestamp: datetime
    user_id: str
    nickname: str
    card: str = ""
    content: List[MessageSegment] = field(default_factory=list)

    def get_formatted_text(self,vision_enabled: bool, image_count_start: int = 0) -> tuple[str, int]:
        """
        Formats the message content into a string representation.
        e.g., "[1] a picture [2] another picture @user"
        Returns the formatted text and the updated image count.
        """
        text_parts = []
        image_count = image_count_start
        for segment in self.content:
            if isinstance(segment, TextMessageSegment):
                text_parts.append(segment.data.text)
            elif isinstance(segment, ImageMessageSegment):
                if vision_enabled:
                    image_count += 1
                    text_parts.append(f"[{image_count}]")
            elif isinstance(segment, AtMessageSegment):
                text_parts.append(f"@{segment.data.qq}")
        return "".join(text_parts), image_count

    def get_images(self) -> List[str]:
        """
        Returns a list of image filenames from the message content.
        These should be the locally saved file names.
        """
        if not settings.get("enable_vision"):
            return []
        images = []
        for segment in self.content:
            if isinstance(segment, ImageMessageSegment):
                images.append(segment.data.file)
        return images


class GroupHistoryMessageEvent(BaseModel):
    self_id: int
    user_id: int
    time: int
    message_id: int
    message_seq: Optional[int] = None
    real_id: Optional[int] = None
    real_seq: Optional[str] = None
    message_type: Optional[Literal["group"]] = None
    sender: Optional[Sender] = None
    raw_message: Optional[str] = None
    font: Optional[int] = None
    sub_type: Optional[str] = None
    message: Optional[List[MessageSegment]] = None
    message_format: Optional[str] = None
    post_type: str
    group_id: int
    message_sent_type: Optional[str] = None

## Long Time State Summary aka Background
## Key Messages
## Recent Messages

class GroupMessageEvent(BaseModel):
    self_id: int
    user_id: int
    time: int
    message_id: int
    message_seq: int
    real_id: int
    real_seq: str
    message_type: Literal["group"]
    sender: Sender
    raw_message: str
    font: int
    sub_type: str
    message: List[MessageSegment]
    message_format: str
    post_type: str
    group_id: int
    message_sent_type: Optional[str] = None
    image_filename: Optional[str] = None


class HistoryMessageData(BaseModel):
    """
    Represents the 'data' field in the group message history response.
    """

    messages: List[GroupHistoryMessageEvent]
    
    @field_validator('messages', mode='before')
    @classmethod
    def validate_messages(cls, v):
        """
        Custom validator to filter out invalid messages and keep valid ones.
        """
        if not isinstance(v, list):
            return v
            
        valid_messages = []
        invalid_count = 0
        
        for i, message_data in enumerate(v):
            try:
                # Try to validate individual message
                validated_message = GroupHistoryMessageEvent.model_validate(message_data)
                valid_messages.append(validated_message)
            except Exception as e:
                invalid_count += 1
                print(f"Skipping invalid message at index {i}: {e}")
                
        if invalid_count > 0:
            print(f"Filtered out {invalid_count} invalid messages, kept {len(valid_messages)} valid messages")
            
        return valid_messages


class GroupMessageHistoryResponse(BaseModel):
    """
    Represents the full response for a request to get group message history.
    """

    status: str
    retcode: int
    data: Optional[HistoryMessageData] = None
    message: str
    wording: str
    echo: Optional[str] = None


class TokenBucket:
    """Token bucket for rate limiting. Allows max_tokens per time_window."""
    
    def __init__(self, max_tokens: int = 3, time_window: int = 3600):
        """
        Initialize token bucket.
        
        Args:
            max_tokens: Maximum number of tokens (default: 3)
            time_window: Time window in seconds (default: 3600 = 1 hour)
        """
        self.max_tokens = max_tokens
        self.time_window = time_window
        self.tokens = max_tokens
        self.last_refill = time.time()
    
    def consume(self) -> bool:
        """
        Try to consume a token.
        
        Returns:
            True if token was consumed, False if no tokens available
        """
        self._refill()
        if self.tokens > 0:
            self.tokens -= 1
            return True
        return False
    
    def time_until_next_token(self) -> int:
        """
        Calculate minutes until next token is available.
        
        Returns:
            Minutes until next token (minimum 1)
        """
        self._refill()
        if self.tokens > 0:
            return 0
        
        # Calculate time since last refill
        time_since_refill = time.time() - self.last_refill
        time_until_refill = self.time_window - time_since_refill
        
        # Convert to minutes and ensure minimum of 1
        minutes = max(1, int(time_until_refill / 60))
        return minutes
    
    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        time_passed = now - self.last_refill
        
        if time_passed >= self.time_window:
            # Full refill if time window has passed
            self.tokens = self.max_tokens
            self.last_refill = now


