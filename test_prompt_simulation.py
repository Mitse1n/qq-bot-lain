import asyncio
import json
import os
import sys
from collections import deque
from datetime import datetime
from typing import List, Deque, Optional

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

try:
    from qqbot.config_loader import settings
    from qqbot.services import GeminiService
    from qqbot.models import Message, GroupMessageEvent, Sender
    import google.genai.types as types
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure you are running this script from the project root.")
    sys.exit(1)

def load_history(file_path: str = "history.json") -> List[dict]:
    """Load the history JSON file."""
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return []
    
    print(f"Loading {file_path}...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Handle different JSON structures if necessary
            if "data" in data and "messages" in data["data"]:
                return data["data"]["messages"]
            elif isinstance(data, list):
                return data
            else:
                print("Unknown history format.")
                return []
    except Exception as e:
        print(f"Error loading history: {e}")
        return []

def convert_event_to_message(event_data: dict) -> Optional[Message]:
    """Convert a raw history event dict to a Message object."""
    try:
        # Validate with Pydantic model first to handle parsing logic
        event = GroupMessageEvent.model_validate(event_data)
        
        return Message(
            timestamp=datetime.fromtimestamp(event.time),
            user_id=str(event.user_id),
            nickname=event.sender.nickname,
            card=event.sender.card,
            content=event.message
        )
    except Exception as e:
        # Silently skip invalid messages or print debug
        # print(f"Skipping message due to validation error: {e}")
        return None

async def main():
    # 1. Initialize Service
    print("Initializing GeminiService...")
    # We pass None for ImageService for now, as we focus on prompt text.
    # If images are required, the service will try to load them from ../data/img
    gemini_service = GeminiService(image_service=None)
    
    # 2. Load History
    raw_messages = load_history()
    if not raw_messages:
        return


    # while True:
    print("\n" + "="*50)
    # user_input = input("Enter real_seq to simulate (or 'q' to quit): ").strip()
    
    # if user_input.lower() == 'q':
    #     break
        
    # if not user_input:
    #     continue
    # # 3. Find target message and build context
    # target_index = -1
    target_real_seq = "665034"
    
    # Try to match string or int real_seq
    for i, msg in enumerate(raw_messages):
        # Check both string and int representation
        msg_seq = str(msg.get("real_seq", ""))
        if msg_seq == target_real_seq:
            target_index = i
            break
    
    if target_index == -1:
        print(f"Message with real_seq {target_real_seq} not found.")
    print(f"Found message at index {target_index}.")
    
    # 4. Reconstruct Context
    # GeminiService uses the last max_messages_history.
    # We collect all messages up to the target (inclusive) and let the service/slicing handle it,
    # OR we slice it here. 
    # The bot typically maintains a deque of recent messages.
    # We will reconstruct the timeline up to this point.
    
    context_messages: Deque[Message] = deque()
    
    # Optimization: Don't convert ALL messages, just enough to fill the window + some buffer
    # But to be accurate to "what the bot saw", we should just take the slice ending at target.
    max_history = 200
    
    # Actually, let's grab a bit more to be safe, then convert
    slice_start = max(0, target_index - 500) 
    subset = raw_messages[slice_start : target_index + 1]
    
    for msg_data in subset:
        msg_obj = convert_event_to_message(msg_data)
        if msg_obj:
            context_messages.append(msg_obj)
    
    print(f"Context constructed with {len(context_messages)} messages (ending at target).")
    
    # 5. Test _build_chat_parts (Prompt Preview)
    print("\n--- Generating Prompt Preview ---")
    try:
        # We need to manually slice for _build_chat_parts if we want to see exactly what's sent,
        # as generate_content_stream does this slicing internally.
        recent_msgs = list(context_messages)[-max_history:]
        
        parts = gemini_service._build_chat_parts(recent_msgs)
        
        print("\n[PROMPT CONTENT]:")
        for part in parts:
            if isinstance(part, str):
                print(part)
            else:
                print(f"<Non-text part: {type(part)}>")
    except Exception as e:
        print(f"Error building chat parts: {e}")
        import traceback
        traceback.print_exc()
    # 6. Test generate_content_stream (LLM Response)
    # print("\n--- Calling LLM (Stream) ---")
    # confirm = input("Send request to Gemini? (y/n): ").strip().lower()
    # if confirm == 'y':
    #     try:
    #         print("Thinking...", end="", flush=True)
    #         async for chunk in gemini_service.generate_content_stream(context_messages):
    #             if hasattr(chunk, 'text'):
    #                 print(chunk.text, end="", flush=True)
    #             # Handle thinking chunks if applicable in newer models
    #             # elif hasattr(chunk, 'parts'): ... 
    #         print("\n[Done]")
    #     except Exception as e:
    #         print(f"\nError from LLM: {e}")

if __name__ == "__main__":
    asyncio.run(main())

