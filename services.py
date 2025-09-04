import httpx
import google.genai as genai
from google.genai import types
import aiohttp
import json
from typing import List, Deque, Optional
import os
import time
import random
import string
from io import BytesIO
from PIL import Image
from config import (
    GEMINI_API_KEY,
    SEND_MESSAGE_URL,
    EVENT_STREAM_URL,
    GET_GROUP_MSG_HISTORY_URL,
    ENABLE_VISION,
    MAX_MESSAGES_HISTORY,
)
from models import Message, GroupMessageHistoryResponse, Sender


class ImageService:
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self.image_dir = "data/img"
        os.makedirs(self.image_dir, exist_ok=True)

    def generate_random_string(self, length: int = 4) -> str:
        return "".join(random.choices(string.ascii_letters + string.digits, k=length))

    async def download_image(
        self, url: str, filename:str
    ) -> Optional[str]:
        if not ENABLE_VISION:
            return

        filepath = os.path.join(self.image_dir, filename)

        if os.path.exists(filepath):
            return

        try:
            async with self.session.get(url) as response:
                response.raise_for_status()
                image_data = await response.read()

                if len(image_data) > 1 * 1024 * 8:
                    img = Image.open(BytesIO(image_data))

                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    quality = 95
                    while len(image_data) > 1 * 1024 * 8 and quality > 10:
                        buffer = BytesIO()
                        img.save(buffer, format="JPEG", quality=quality)
                        image_data = buffer.getvalue()
                        quality -= 5

                with open(filepath, "wb") as f:
                    f.write(image_data)
        except aiohttp.ClientError as e:
            print(f"Error downloading image: {e}")
            return


class GeminiService:
    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        #self.model_name = "gemma-3-27b-it"
        self.model_name = "gemini-2.5-pro"
        self.max_messages_history = MAX_MESSAGES_HISTORY


    def _build_chat_prompt(self, messages: List[Message]) -> str:
        latest_msg = messages[-1]
        other_msgs = messages[:-1]
        
        pre_msg_lines = []
        image_count = 0
        for msg in other_msgs:
            formatted_text, image_count = msg.get_formatted_text(image_count, vision_enabled=ENABLE_VISION)
            pre_msg_lines.append(
                f"({msg.timestamp.strftime('%m-%d %H:%M')}) {msg.user_id}: {formatted_text}"
            )
        pre_msgs_text = "\n".join(pre_msg_lines)

        latest_msg_text, _ = latest_msg.get_formatted_text(image_count, vision_enabled=ENABLE_VISION)
        latest_msg_text = f"({latest_msg.timestamp.strftime('%m-%d %H:%M')}) {latest_msg.user_id}: {latest_msg_text}"
        senders: dict[str, Sender] = {}
        for msg in messages:
            if msg.user_id not in senders:
                senders[msg.user_id] = Sender(user_id=msg.user_id, nickname=msg.nickname, card=msg.card)
        senders_text = "\n".join(
            [
                f"ID: {sender.user_id} 账号名: {sender.nickname}{f' 群昵称: {sender.card}' if sender.card else ''}"
                for sender in senders.values()
            ]
        )
        
        
       
        return (
            f"你是一个群聊机器人 Lain. id 是 3236753583. 需要和群里的人交流\n"
            f"不要说违反中国法律的话, 不要太强调你的机器人身份, 就像一个普通人一样\n"
            f"如果有群昵称, 在需要称呼群员的时候, 优先用群昵称称呼群员,可以不用说完整的群昵称,根据群昵称想些能辨认的简称也行 其次是账号名, 尽量不要提及群员id\n"
            f"这次涉及到的群员有:\n{senders_text}\n"
            f"聊天记录格式是 (发送时间)群员id: 内容\n"
            f"时间格式是 %m-%d %H:%M\n"
            f"{'图片格式是 [n], 只要被 [] 包裹就是图片, n 是一个数字, 表示是第几张图片,' if ENABLE_VISION else ''}\n"
            f"下面是最近的聊天记录\n\n"
            f"{pre_msgs_text}\n\n"
            f"最近聊天记录只是参考, 主要是回复给你发送的消息, 你的回答不支持表情, 不能用 @ 符号来 mention 群员, 也不支持图片\n"
            f"给你发送的消息是\n\n{latest_msg_text}\n"
        )


    def generate_content(
        self, messages: Deque[Message]
    ) -> str:
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                if not messages:
                    return "I have no messages to process."

                recent_messages = list(messages)[-self.max_messages_history:]
                prompt = self._build_chat_prompt(recent_messages)
                content_parts = [prompt]
                if ENABLE_VISION:
                    imgs = []
                    for msg in recent_messages:
                        imgs.extend(msg.get_images())
                    
                    if imgs:
                        image_dir = "data/img"
                        for filename in imgs:
                            image_path = os.path.join(image_dir, filename)
                            if os.path.exists(image_path):
                                try:
                                    with open(image_path, "rb") as f:
                                        image_bytes = f.read()
                                        content_parts.append(
                                            types.Part.from_bytes(
                                                data=image_bytes,
                                                mime_type='image/jpeg',
                                            ),
                                        )
                                except Exception as e:
                                    print(f"Could not open image {image_path}: {e}")

                imgs = []
                if ENABLE_VISION:
                    for msg in recent_messages:
                        imgs.extend(msg.get_images())

                # response = await model.generate_content(content_parts)
                response = self.client.models.generate_content(
                model=self.model_name,
                    contents=content_parts
                )
                return response.text
            except Exception as e:
                if ("503" in str(e) or "overloaded" in str(e).lower()) and attempt < max_retries:
                    print(e)
                    print(f"Model is overloaded. Retrying ... (Attempt {attempt + 1}/{max_retries})")
                else:
                    print(f"Error generating content with Gemini: {e}")
                    return "Sorry, I had a problem generating a response."
        return "Sorry, I had a problem generating a response after multiple retries."


class ChatService:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def send_group_message(self, group_id: int, message: str, reply_id: int|None = None):
        max_retries = 2
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                if reply_id is None:
                    payload = {"group_id": group_id, "message": [{"type": "text", "data": {"text": message}}]}
                else:
                    payload = {"group_id": group_id,"message": [{"type": "reply","data": {"id": reply_id}},{"type": "text", "data": {"text": message}}],}
                
                response = await self.client.post(SEND_MESSAGE_URL, json=payload)
                response.raise_for_status()
                print(f"Sent message to group {group_id}")
                return  # 成功发送，退出重试循环
                
            except httpx.HTTPStatusError as e:
                print(f"Error sending message (attempt {retry_count + 1}/{max_retries + 1}): {e.response.status_code}")
                if retry_count == max_retries:
                    print(f"Failed to send message after {max_retries + 1} attempts")
                    break
                retry_count += 1
                
            except httpx.RequestError as e:
                print(f"Error sending message (attempt {retry_count + 1}/{max_retries + 1}): {e}")
                if retry_count == max_retries:
                    print(f"Failed to send message after {max_retries + 1} attempts")
                    break
                retry_count += 1

    async def get_group_msg_history(
        self, group_id: int, message_seq: int = 0, count: int = 1000
    ) -> Optional[GroupMessageHistoryResponse]:
        payload = {
            "group_id": group_id,
            "message_seq": message_seq,
            "count": count,
            "reverseOrder": False,
        }
        try:
            response = await self.client.post(GET_GROUP_MSG_HISTORY_URL, json=payload, timeout=60.0)
            response.raise_for_status()
            return GroupMessageHistoryResponse.model_validate(response.json())
        except httpx.HTTPStatusError as e:
            print(f"Error getting group message history: {e.response.status_code}")
        except httpx.RequestError as e:
            print(f"Error getting group message history: {e}")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from response: {response.text}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        return None


class EventService:
    def __init__(self):
        timeout = aiohttp.ClientTimeout(total=None)
        self.session = aiohttp.ClientSession(timeout=timeout)

    async def listen(self):
        try:
            async with self.session.get(
                EVENT_STREAM_URL, headers={"Accept": "text/event-stream"}
            ) as resp:
                while True:
                    line = await resp.content.readline()
                    if not line:
                        break
                    line = line.decode("utf-8").strip()
                    if line.startswith("data:"):
                        event_data = json.loads(line[5:])
                        if (
                            event_data.get("message_type") == "group"
                            and event_data.get("post_type") == "message"
                        ):
                            yield event_data
        except aiohttp.ClientError as e:
            print(f"Error connecting to event stream: {e}")
        finally:
            await self.session.close()


