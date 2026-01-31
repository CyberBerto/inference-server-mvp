"""
Pydantic models for OpenAI-compatible API.
Matches OpenRouter's expected request/response format.
"""

from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field


# ============ Request Models ============

class Message(BaseModel):
    """Chat message in OpenAI format."""
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ToolFunction(BaseModel):
    """Function definition for tool use."""
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class Tool(BaseModel):
    """Tool definition."""
    type: Literal["function"] = "function"
    function: ToolFunction


class ResponseFormat(BaseModel):
    """Response format specification."""
    type: Literal["text", "json_object"] = "text"


class ChatCompletionRequest(BaseModel):
    """
    OpenAI-compatible chat completion request.
    Supports all standard parameters plus extensions.
    """
    model: str
    messages: List[Message]

    # Generation parameters
    max_tokens: Optional[int] = Field(default=4096, ge=1, le=131072)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=1)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    repetition_penalty: Optional[float] = Field(default=1.0, ge=0.0)

    # Control
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False

    # Tools
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

    # Response format
    response_format: Optional[ResponseFormat] = None

    # Metadata
    user: Optional[str] = None

    # vLLM-specific
    best_of: Optional[int] = None
    use_beam_search: Optional[bool] = False
    skip_special_tokens: Optional[bool] = True


# ============ Response Models ============

class UsageInfo(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    """Single completion choice."""
    index: int
    message: Message
    finish_reason: Optional[str] = None
    logprobs: Optional[Dict[str, Any]] = None


class ChatCompletionResponse(BaseModel):
    """
    OpenAI-compatible chat completion response.
    """
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[UsageInfo] = None
    system_fingerprint: Optional[str] = None


# ============ Streaming Response Models ============

class DeltaMessage(BaseModel):
    """Delta content for streaming."""
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class StreamChoice(BaseModel):
    """Streaming choice with delta."""
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """Single chunk in streaming response."""
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamChoice]


# ============ Model Info ============

class ModelPricing(BaseModel):
    """Pricing information for OpenRouter."""
    prompt: str
    completion: str


class ModelInfo(BaseModel):
    """
    Model metadata for OpenRouter discovery.
    Returned by /api/v1/models endpoint.
    """
    id: str
    object: str = "model"
    created: int
    owned_by: str
    name: str
    context_length: int
    pricing: ModelPricing
    quantization: str
    supported_features: List[str]
