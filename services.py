import config
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


async def retry_http_request(url: str, payload: dict, max_retry_count: int = 2, timeout: float = 10, client: Optional[httpx.AsyncClient] = None,method: str = "POST"):
    """
    通用的HTTP请求重试工具函数
    
    Args:
        url: 请求的URL
        payload: 请求的JSON载荷
        max_retry_count: 最大重试次数，默认2次
        timeout: 请求超时时间，默认60秒
        client: httpx.AsyncClient 实例，如果为None则创建临时客户端
    
    Returns:
        httpx.Response: 成功的响应对象
    
    Raises:
        httpx.HTTPStatusError: HTTP状态错误
        httpx.RequestError: 请求错误
    """
    retry_count = 0
    should_close_client = False
    
    if client is None:
        client = httpx.AsyncClient()
        should_close_client = True
    
    try:
        while retry_count <= max_retry_count:
            try:
                # response = await client.post(url, json=payload, timeout=timeout)
                response = await client.request(method, url, json=payload, timeout=timeout)
                response.raise_for_status()
                return response
                
            except httpx.HTTPStatusError as e:
                print(f"HTTP status error (attempt {retry_count + 1}/{max_retry_count + 1}): {e.response.status_code}")
                if retry_count == max_retry_count:
                    print(f"Failed to complete request after {max_retry_count + 1} attempts")
                    raise
                retry_count += 1
                
            except httpx.RequestError as e:
                print(f"Request error (attempt {retry_count + 1}/{max_retry_count + 1}): {e}")
                if retry_count == max_retry_count:
                    print(f"Failed to complete request after {max_retry_count + 1} attempts")
                    raise
                retry_count += 1
    finally:
        if should_close_client:
            await client.aclose()


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
        self.small_model_name = "gemma-3-27b-it"
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
            f"你是一个群聊机器人{config.BOT_NAME} . id 是 {config.BOT_QQ_ID}. 需要和群里的人交流\n"
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
                        
                grounding_tool = types.Tool(
                    google_search=types.GoogleSearch()
                )

                # response = await model.generate_content(content_parts)
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=content_parts,
                    config=types.GenerateContentConfig(
                        tools=[grounding_tool]
                        )
                )
                text_response = response.text
                

                return text_response
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

    async def send_group_message(self, group_id: int, message: str, reply_id: Optional[int] = None):
        if reply_id is None:
            payload = {"group_id": group_id, "message": [{"type": "text", "data": {"text": message}}]}
        else:
            payload = {"group_id": group_id,"message": [{"type": "reply","data": {"id": reply_id}},{"type": "text", "data": {"text": message}}],}
        
        try:
            await retry_http_request(SEND_MESSAGE_URL, payload, max_retry_count=2, client=self.client)
            print(f"Sent message to group {group_id}: {message}")
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            print(f"Failed to send message to group {group_id}: {message}. error: {e}")

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
            response = await retry_http_request(GET_GROUP_MSG_HISTORY_URL, payload, max_retry_count=2, client=self.client, method="POST")
            return GroupMessageHistoryResponse.model_validate(response.json())
        except httpx.HTTPStatusError as e:
            print(f"Error getting group message history: {e.response.status_code}")
        except httpx.RequestError as e:
            print(f"Error getting group message history: {e}")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from response: {e}")
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


def add_citations(response: types.GenerateContentResponse):
    text = response.text
    supports = response.candidates[0].grounding_metadata.grounding_supports
    chunks = response.candidates[0].grounding_metadata.grounding_chunks

    # Sort supports by end_index in descending order to avoid shifting issues when inserting.
    sorted_supports = sorted(supports, key=lambda s: s.segment.end_index, reverse=True)

    for support in sorted_supports:
        end_index = support.segment.end_index
        if support.grounding_chunk_indices:
            # Create citation string like [1](link1)[2](link2)
            citation_links = []
            for i in support.grounding_chunk_indices:
                if i < len(chunks):
                    uri = chunks[i].web.uri
                    citation_links.append(f"[{i + 1}]({uri})")

            citation_string = ", ".join(citation_links)
            text = text[:end_index] + "\n[参考]"+ citation_string + text[end_index:]

    return text