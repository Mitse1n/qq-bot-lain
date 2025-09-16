"""
API module for exposing HTTP endpoints for the QQ bot.
"""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import asyncio
from datetime import datetime

from qqbot.memory_service import GroupMemoryService
from qqbot.services import GeminiService, ChatService
from qqbot.models import Message, TextMessageSegment, TextData


class MemoryUpdateRequest(BaseModel):
    """Request model for memory update/initialization."""
    messages: Optional[List[Dict[str, Any]]] = Field(
        None, 
        description="Optional list of messages to initialize memory with. If not provided, will fetch from chat history."
    )
    force_reinitialize: bool = Field(
        False, 
        description="Whether to force reinitialize memory even if it already exists."
    )


class MemoryUpdateResponse(BaseModel):
    """Response model for memory update/initialization."""
    success: bool
    message: str
    group_id: int
    memory_initialized: bool
    memory_content: Optional[str] = None


class APIServer:
    """FastAPI server for exposing bot management endpoints."""
    
    def __init__(self):
        self.app = FastAPI(
            title="QQ Bot API",
            description="API endpoints for managing QQ bot functionality",
            version="1.0.0"
        )
        self.memory_service: Optional[GroupMemoryService] = None
        self._setup_routes()
    
    def set_memory_service(self, memory_service: GroupMemoryService):
        """Set the memory service instance."""
        self.memory_service = memory_service
    
    def _get_memory_service(self) -> GroupMemoryService:
        """Get the memory service instance or raise error if not set."""
        if self.memory_service is None:
            raise HTTPException(
                status_code=500, 
                detail="Memory service not initialized"
            )
        return self.memory_service
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.post("/memory/{group_id}", response_model=MemoryUpdateResponse)
        async def update_or_initialize_memory(
            group_id: int,
            request: MemoryUpdateRequest,
            memory_service: GroupMemoryService = Depends(self._get_memory_service)
        ):
            """
            Initialize or update memory for a specific group.
            
            Args:
                group_id: The ID of the group to update memory for
                request: Request payload containing optional messages and configuration
                
            Returns:
                MemoryUpdateResponse with operation result
            """
            try:
                # Validate group_id
                if group_id <= 0:
                    raise HTTPException(
                        status_code=400,
                        detail="Group ID must be a positive integer"
                    )
                
                # Check if memory already exists and if we should force reinitialize
                memory_exists = memory_service.is_memory_initialized(group_id)
                
                if memory_exists and not request.force_reinitialize:
                    # Memory already exists, try to update it
                    if request.messages:
                        # Convert provided messages to Message objects
                        message_objects = []
                        for msg_data in request.messages:
                            try:
                                # Create a Message object from the provided data
                                message = Message.from_api_data(msg_data)
                                message_objects.append(message)
                            except Exception as e:
                                print(f"Error processing message: {e}")
                                continue
                        
                        # Update memory with provided messages
                        if message_objects:
                            success = await memory_service.update_group_memory(group_id, message_objects)
                            if success:
                                memory_content = memory_service.get_cached_memory(group_id)
                                return MemoryUpdateResponse(
                                    success=True,
                                    message=f"Successfully updated memory for group {group_id}"
                                            f" with {len(message_objects)} messages",
                                    group_id=group_id,
                                    memory_initialized=True,
                                    memory_content=memory_content
                                )
                            else:
                                raise HTTPException(
                                    status_code=500,
                                    detail=f"Failed to update memory for group {group_id}"
                                )
                        else:
                            raise HTTPException(
                                status_code=400,
                                detail="No valid messages provided for memory update"
                            )
                    else:
                        # No new messages provided, just return current memory status
                        memory_content = memory_service.get_cached_memory(group_id)
                        return MemoryUpdateResponse(
                            success=True,
                            message=f"Memory already exists for group {group_id}."
                                    f" Use force_reinitialize=true to reinitialize.",
                            group_id=group_id,
                            memory_initialized=True,
                            memory_content=memory_content
                        )
                
                # Initialize or reinitialize memory
                if request.force_reinitialize and memory_exists:
                    # Clear existing memory state
                    memory_service.group_memory_initialized[group_id] = False
                    memory_service.group_memories.pop(group_id, None)
                    memory_service.group_message_counts[group_id] = 0
                
                if request.messages:
                    # Initialize with provided messages
                    message_objects = []
                    for msg_data in request.messages:
                        try:
                            message = Message.from_api_data(msg_data)
                            message_objects.append(message)
                        except Exception as e:
                            print(f"Error processing message: {e}")
                            continue
                    
                    if len(message_objects) < memory_service.MINIMUM_MESSAGES_FOR_MEMORY:
                        # Not enough messages to generate memory
                        return MemoryUpdateResponse(
                            success=True,
                            message=f"Received {len(message_objects)} messages. "
                                    f"Need {memory_service.MINIMUM_MESSAGES_FOR_MEMORY - len(message_objects)}"
                                    f" more messages to initialize memory.",
                            group_id=group_id,
                            memory_initialized=False,
                            memory_content=None
                        )
                    else:
                        # Enough messages to initialize memory directly
                        # Use the new unified memory generation method
                        
                        # Trigger memory generation
                        success = await memory_service._generate_initial_memory_from_messages(
                            group_id, 
                            message_objects
                        )
                        
                        if success:
                            memory_content = memory_service.get_cached_memory(group_id)
                            return MemoryUpdateResponse(
                                success=True,
                                message=f"Successfully initialized memory for group {group_id}"
                                        f" with {len(message_objects)} messages",
                                group_id=group_id,
                                memory_initialized=True,
                                memory_content=memory_content
                            )
                        else:
                            raise HTTPException(
                                status_code=500,
                                detail=f"Failed to initialize memory for group {group_id}"
                            )
                else:
                    # Initialize with chat history
                    success = await memory_service.initialize_group_memory(group_id)
                    
                    if success:
                        memory_content = memory_service.get_cached_memory(group_id)
                        return MemoryUpdateResponse(
                            success=True,
                            message=f"Successfully initialized memory for group {group_id} from chat history",
                            group_id=group_id,
                            memory_initialized=True,
                            memory_content=memory_content
                        )
                    else:
                        # Check if it's in accumulation mode
                        if memory_service.is_accumulating_messages(group_id):
                            accumulated_count = memory_service.get_accumulated_message_count(group_id)
                            return MemoryUpdateResponse(
                                success=True,
                                message=f"Group {group_id} has insufficient messages for memory initialization."
                                        f" Currently accumulated: "
                                        f"{accumulated_count}/{memory_service.MINIMUM_MESSAGES_FOR_MEMORY}",
                                group_id=group_id,
                                memory_initialized=False,
                                memory_content=None
                            )
                        else:
                            raise HTTPException(
                                status_code=500,
                                detail=f"Failed to initialize memory for group {group_id}"
                            )
                            
            except HTTPException:
                raise
            except Exception as e:
                print(f"Unexpected error in memory endpoint: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Internal server error: {str(e)}"
                )
        
        @self.app.get("/memory/{group_id}")
        async def get_memory(
            group_id: int,
            memory_service: GroupMemoryService = Depends(self._get_memory_service)
        ):
            """
            Get current memory for a specific group.
            
            Args:
                group_id: The ID of the group to get memory for
                
            Returns:
                Current memory content and status
            """
            try:
                if group_id <= 0:
                    raise HTTPException(
                        status_code=400,
                        detail="Group ID must be a positive integer"
                    )
                
                memory_content = memory_service.get_cached_memory(group_id)
                is_initialized = memory_service.is_memory_initialized(group_id)
                is_accumulating = memory_service.is_accumulating_messages(group_id)
                accumulated_count = memory_service.get_accumulated_message_count(group_id) if is_accumulating else 0
                
                return {
                    "group_id": group_id,
                    "memory_initialized": is_initialized,
                    "is_accumulating": is_accumulating,
                    "accumulated_message_count": accumulated_count,
                    "required_messages": memory_service.MINIMUM_MESSAGES_FOR_MEMORY,
                    "memory_content": memory_content
                }
                
            except Exception as e:
                print(f"Error getting memory for group {group_id}: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to get memory for group {group_id}: {str(e)}"
                )
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "service": "QQ Bot API"
            }


# Global API server instance
api_server = APIServer()
app = api_server.app
