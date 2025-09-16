#!/usr/bin/env python3
"""
Test script for the memory API endpoint.

This script demonstrates how to use the POST /memory/:group_id endpoint
to initialize or update group memory.
"""

import httpx
import asyncio
import json
from datetime import datetime


async def test_memory_api():
    """Test the memory API endpoints."""
    base_url = "http://localhost:8000"
    group_id = 12345  # Test group ID
    
    async with httpx.AsyncClient() as client:
        print("Testing Memory API Endpoints")
        print("=" * 50)
        
        # Test 1: Health check
        print("1. Testing health check...")
        try:
            response = await client.get(f"{base_url}/health")
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.json()}")
        except Exception as e:
            print(f"   Error: {e}")
        
        print()
        
        # Test 2: Get memory (should not exist initially)
        print(f"2. Getting memory for group {group_id} (should not exist)...")
        try:
            response = await client.get(f"{base_url}/memory/{group_id}")
            print(f"   Status: {response.status_code}")
            print(f"   Response: {json.dumps(response.json(), indent=2)}")
        except Exception as e:
            print(f"   Error: {e}")
        
        print()
        
        # Test 3: Initialize memory with custom messages
        print(f"3. Initializing memory for group {group_id} with custom messages...")
        
        # Create sample messages
        sample_messages = []
        base_timestamp = datetime.now().timestamp()
        
        # Generate enough messages to trigger memory initialization
        users = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
        conversations = [
            "Hello everyone!",
            "How's everyone doing today?",
            "I'm doing great, thanks for asking!",
            "What are we planning for the weekend?",
            "Maybe we could go to the movies?",
            "That sounds like a great idea!",
            "What movie should we watch?",
            "How about the new action movie that just came out?",
            "I love action movies! Count me in.",
            "Should we meet at the mall at 7 PM?",
            "Perfect, see you all there!",
            "Don't forget to buy tickets in advance.",
            "Good point, I'll buy them online now.",
            "Great! Looking forward to it.",
            "This is going to be so much fun!"
        ]
        
        # Create 450 messages to ensure we exceed the 400 message threshold
        for i in range(450):
            user_idx = i % len(users)
            conv_idx = i % len(conversations)
            
            message = {
                "user_id": f"user_{user_idx + 1}",
                "nickname": users[user_idx],
                "card": f"{users[user_idx]}_card",
                "content": f"{conversations[conv_idx]} (message {i+1})",
                "timestamp": base_timestamp + (i * 60)  # 1 minute apart
            }
            sample_messages.append(message)
        
        request_data = {
            "messages": sample_messages,
            "force_reinitialize": False
        }
        
        try:
            response = await client.post(
                f"{base_url}/memory/{group_id}",
                json=request_data
            )
            print(f"   Status: {response.status_code}")
            response_data = response.json()
            print(f"   Success: {response_data.get('success')}")
            print(f"   Message: {response_data.get('message')}")
            print(f"   Memory Initialized: {response_data.get('memory_initialized')}")
            if response_data.get('memory_content'):
                print(f"   Memory Content (first 200 chars): {response_data.get('memory_content')[:200]}...")
        except Exception as e:
            print(f"   Error: {e}")
        
        print()
        
        # Test 4: Get memory again (should exist now)
        print(f"4. Getting memory for group {group_id} (should exist now)...")
        try:
            response = await client.get(f"{base_url}/memory/{group_id}")
            print(f"   Status: {response.status_code}")
            response_data = response.json()
            print(f"   Memory Initialized: {response_data.get('memory_initialized')}")
            print(f"   Is Accumulating: {response_data.get('is_accumulating')}")
            if response_data.get('memory_content'):
                print(f"   Memory Content (first 200 chars): {response_data.get('memory_content')[:200]}...")
        except Exception as e:
            print(f"   Error: {e}")
        
        print()
        
        # Test 5: Update memory with new messages
        print(f"5. Updating memory for group {group_id} with new messages...")
        
        new_messages = [
            {
                "user_id": "user_1",
                "nickname": "Alice",
                "content": "Hey everyone, the movie was amazing!",
                "timestamp": base_timestamp + (500 * 60)
            },
            {
                "user_id": "user_2", 
                "nickname": "Bob",
                "content": "I agree! The action scenes were incredible.",
                "timestamp": base_timestamp + (501 * 60)
            },
            {
                "user_id": "user_3",
                "nickname": "Charlie", 
                "content": "We should definitely do this again next weekend.",
                "timestamp": base_timestamp + (502 * 60)
            }
        ]
        
        update_request = {
            "messages": new_messages,
            "force_reinitialize": False
        }
        
        try:
            response = await client.post(
                f"{base_url}/memory/{group_id}",
                json=update_request
            )
            print(f"   Status: {response.status_code}")
            response_data = response.json()
            print(f"   Success: {response_data.get('success')}")
            print(f"   Message: {response_data.get('message')}")
        except Exception as e:
            print(f"   Error: {e}")
        
        print()
        
        # Test 6: Test with insufficient messages (new group)
        test_group_id = 67890
        print(f"6. Testing with insufficient messages for group {test_group_id}...")
        
        few_messages = [
            {
                "user_id": "user_1",
                "nickname": "Alice",
                "content": "Hi there!",
                "timestamp": datetime.now().timestamp()
            },
            {
                "user_id": "user_2",
                "nickname": "Bob", 
                "content": "Hello!",
                "timestamp": datetime.now().timestamp() + 60
            }
        ]
        
        request_data = {
            "messages": few_messages,
            "force_reinitialize": False
        }
        
        try:
            response = await client.post(
                f"{base_url}/memory/{test_group_id}",
                json=request_data
            )
            print(f"   Status: {response.status_code}")
            response_data = response.json()
            print(f"   Success: {response_data.get('success')}")
            print(f"   Message: {response_data.get('message')}")
            print(f"   Memory Initialized: {response_data.get('memory_initialized')}")
        except Exception as e:
            print(f"   Error: {e}")
        
        print()
        print("Test completed!")


if __name__ == "__main__":
    print("Make sure the QQ Bot is running with the API server enabled.")
    print("Starting memory API tests in 3 seconds...")
    
    import time
    time.sleep(3)
    
    asyncio.run(test_memory_api())
