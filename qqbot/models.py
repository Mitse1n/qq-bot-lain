from typing import List, Literal, Union, Annotated, Optional
from dataclasses import dataclass, field
from datetime import datetime


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


class FileData(BaseModel):
    file: str
    url: Optional[str] = None
    file_size: Optional[str] = None


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


class FileMessageSegment(BaseModel):
    type: Literal["file"]
    data: FileData


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
        FileMessageSegment,
    ],
    Field(discriminator="type"),
]


@dataclass
class Message:
    timestamp: datetime
    user_id: str
    nickname: str
    card: str = ""
    real_seq: Optional[str] = None
    content: List[MessageSegment] = field(default_factory=list)
    
    @classmethod
    def from_group_event(cls, event, content: List[MessageSegment]) -> "Message":
        """从群消息事件创建 Message 对象"""
        return cls(
            timestamp=datetime.fromtimestamp(event.get("time")),
            user_id=str(event.get("user_id")),
            card=str(event.get("sender", {}).get("card", "")),
            nickname=event.get("sender", {}).get("nickname", ""),
            content=content,
            real_seq=event.get("real_seq"),
        )
    
    @classmethod
    def from_history_event(cls, msg: "GroupHistoryMessageEvent") -> Optional["Message"]:
        """从历史消息事件创建 Message 对象"""
        try:
            return cls(
                timestamp=datetime.fromtimestamp(msg.time),
                user_id=str(msg.user_id),
                nickname=msg.sender.nickname if msg.sender else "",
                card=msg.sender.card if msg.sender else "",
                content=msg.message or [],
                real_seq=msg.real_seq if msg.real_seq else ""
            )
        except Exception as e:
            print(f"Error converting message: {e}")
            return None
    
    @classmethod
    def create_text_message(cls, user_id: str, nickname: str, text: str, 
                           card: str = "", timestamp: Optional[datetime] = None) -> "Message":
        """创建文本消息"""
        if timestamp is None:
            timestamp = datetime.now()
        
        return cls(
            timestamp=timestamp,
            user_id=user_id,
            nickname=nickname,
            card=card,
            content=[TextMessageSegment(type="text", data=TextData(text=text))]
        )
    
    @classmethod
    def create_system_message(cls, text: str, timestamp: Optional[datetime] = None) -> "Message":
        """创建系统消息"""
        if timestamp is None:
            timestamp = datetime.now()
            
        return cls(
            timestamp=timestamp,
            user_id="0",  # 使用 "0" 作为系统消息的用户ID
            nickname="system",
            content=[TextMessageSegment(type="text", data=TextData(text=text))]
        )
    
    @classmethod
    def from_api_data(cls, msg_data: dict) -> "Message":
        """从 API 数据创建 Message 对象"""
        return cls(
            timestamp=datetime.fromtimestamp(
                msg_data.get('timestamp', datetime.now().timestamp())
            ),
            user_id=str(msg_data.get('user_id', 'unknown')),
            nickname=msg_data.get('nickname', ''),
            card=msg_data.get('card', ''),
            content=[
                TextMessageSegment(
                    type="text",
                    data=TextData(text=msg_data.get('content', ''))
                )
            ]
        )

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
            elif isinstance(segment, FileMessageSegment):
                text_parts.append("[文件]")
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


