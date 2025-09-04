from typing import List, Literal, Union, Annotated, Optional
from dataclasses import dataclass, field
from datetime import datetime


from pydantic import BaseModel, Field

from config import ENABLE_VISION


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

    def get_formatted_text(self, image_count_start: int = 0, vision_enabled: bool = True) -> tuple[str, int]:
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
        if not ENABLE_VISION:
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


