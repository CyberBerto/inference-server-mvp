"""
Pydantic models for the OpenAI-compatible Chat Completions API.

This module defines all request and response models matching OpenAI's API schema,
with extensions for OpenRouter compatibility and vLLM-specific features.

Model Categories:
    1. Request Models - Incoming API request validation
    2. Response Models - Non-streaming completion responses
    3. Streaming Models - Server-Sent Events (SSE) chunk format
    4. Model Info - OpenRouter model discovery metadata

OpenAI API Reference:
    https://platform.openai.com/docs/api-reference/chat/create

OpenRouter Extensions:
    - pricing: Token costs for billing
    - supported_features: Capability flags
    - quantization: Model precision info

vLLM Extensions:
    - best_of: Return best of n completions
    - use_beam_search: Enable beam search decoding
    - skip_special_tokens: Filter special tokens from output

Version: 0.3.0
License: MIT
"""

from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field


# =============================================================================
# Request Models
# =============================================================================

class Message(BaseModel):
    """
    A single message in a chat conversation.

    Follows OpenAI's message format with support for all role types
    including tool/function calling.

    Attributes:
        role: The speaker role - system, user, assistant, or tool
        content: The text content of the message (None for tool calls)
        name: Optional name for the participant (used with tools)
        tool_calls: List of tool/function calls made by assistant
        tool_call_id: ID linking tool response to original call

    Examples:
        System message:
            {"role": "system", "content": "You are a helpful assistant."}

        User message:
            {"role": "user", "content": "What's the weather?"}

        Assistant with tool call:
            {"role": "assistant", "tool_calls": [{"id": "call_1", ...}]}

        Tool response:
            {"role": "tool", "tool_call_id": "call_1", "content": "72°F"}
    """
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ToolFunction(BaseModel):
    """
    Function definition for tool/function calling.

    Describes a function the model can call, including its parameters
    as a JSON Schema object.

    Attributes:
        name: Function identifier (must match pattern ^[a-zA-Z0-9_-]+$)
        description: Human-readable description of what the function does
        parameters: JSON Schema describing the function's parameters

    Example:
        {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    """
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class Tool(BaseModel):
    """
    Tool definition wrapper.

    Currently only "function" type is supported, matching OpenAI's API.
    Future types may include "code_interpreter", "retrieval", etc.

    Attributes:
        type: Tool type, currently always "function"
        function: The function definition
    """
    type: Literal["function"] = "function"
    function: ToolFunction


class ResponseFormat(BaseModel):
    """
    Response format specification for structured output.

    Controls the format of the model's response. Use "json_object" to
    guarantee valid JSON output (requires prompting for JSON as well).

    Attributes:
        type: "text" for normal output, "json_object" for JSON mode

    Note:
        When using json_object, you MUST also instruct the model to
        produce JSON in your system/user message, or it may loop.
    """
    type: Literal["text", "json_object"] = "text"


class ChatCompletionRequest(BaseModel):
    """
    OpenAI-compatible chat completion request.

    Supports all standard OpenAI parameters plus vLLM-specific extensions.
    All optional parameters have sensible defaults.

    Required Fields:
        model: The model identifier to use for completion
        messages: List of messages in the conversation

    Generation Parameters:
        max_tokens: Maximum tokens to generate (1-131072, default: 4096)
        temperature: Sampling temperature (0.0-2.0, default: 0.7)
            - 0.0: Deterministic, always picks most likely token
            - 0.7: Balanced creativity (recommended)
            - 2.0: Maximum randomness
        top_p: Nucleus sampling threshold (0.0-1.0, default: 1.0)
            - 1.0: Consider all tokens
            - 0.9: Consider tokens comprising top 90% probability
        top_k: Top-k sampling (optional, vLLM extension)
        frequency_penalty: Penalize repeated tokens (-2.0 to 2.0)
        presence_penalty: Penalize tokens that appeared at all (-2.0 to 2.0)
        repetition_penalty: vLLM's repetition penalty (>= 0.0, default: 1.0)

    Control Parameters:
        stop: Stop sequences - generation stops when these are produced
        stream: Enable SSE streaming (default: False)

    Tool/Function Calling:
        tools: List of available tools the model can call
        tool_choice: Control tool selection ("auto", "none", or specific)

    Response Format:
        response_format: Request JSON output format

    Metadata:
        user: Unique end-user identifier for abuse monitoring

    vLLM Extensions:
        best_of: Generate n completions, return best (expensive!)
        use_beam_search: Use beam search instead of sampling
        skip_special_tokens: Remove special tokens from output

    Example:
        {
            "model": "your-org/your-model",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"}
            ],
            "max_tokens": 1000,
            "temperature": 0.7,
            "stream": false
        }
    """
    # Required fields
    model: str
    messages: List[Message]

    # Generation parameters with validation
    max_tokens: Optional[int] = Field(
        default=4096,
        ge=1,
        le=131072,
        description="Maximum tokens to generate"
    )
    temperature: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0=deterministic, 2=max random)"
    )
    top_p: Optional[float] = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability threshold"
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        description="Top-k sampling (vLLM extension)"
    )
    frequency_penalty: Optional[float] = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Penalize tokens by their frequency"
    )
    presence_penalty: Optional[float] = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Penalize tokens that have appeared"
    )
    repetition_penalty: Optional[float] = Field(
        default=1.0,
        ge=0.0,
        description="vLLM repetition penalty (1.0 = no penalty)"
    )

    # Control parameters
    stop: Optional[Union[str, List[str]]] = None
    """Stop sequences - can be a single string or list of strings."""

    stream: Optional[bool] = False
    """Enable Server-Sent Events streaming response."""

    # Tool/function calling
    tools: Optional[List[Tool]] = None
    """List of tools available for the model to call."""

    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    """
    Control tool selection behavior:
    - "auto": Model decides whether to call tools
    - "none": Never call tools
    - {"type": "function", "function": {"name": "..."}}: Force specific tool
    """

    # Response format
    response_format: Optional[ResponseFormat] = None
    """Request structured JSON output."""

    # Metadata
    user: Optional[str] = None
    """Unique identifier for end-user, used for abuse monitoring."""

    # vLLM-specific extensions
    best_of: Optional[int] = None
    """Generate n completions and return the best one (expensive!)."""

    use_beam_search: Optional[bool] = False
    """Use beam search decoding instead of sampling."""

    skip_special_tokens: Optional[bool] = True
    """Remove special tokens from output."""


# =============================================================================
# Response Models
# =============================================================================

class UsageInfo(BaseModel):
    """
    Token usage statistics for billing and monitoring.

    Attributes:
        prompt_tokens: Tokens in the input/prompt
        completion_tokens: Tokens in the generated response
        total_tokens: Sum of prompt + completion tokens

    Note:
        Token counts are model-specific. The same text may have different
        token counts across different models/tokenizers.
    """
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    """
    A single completion choice in non-streaming response.

    For standard requests, there's typically one choice. When using
    `n` or `best_of` parameters, multiple choices may be returned.

    Attributes:
        index: Zero-based index of this choice
        message: The generated message
        finish_reason: Why generation stopped:
            - "stop": Natural stop or stop sequence hit
            - "length": Max tokens reached
            - "tool_calls": Model wants to call a tool
            - "content_filter": Content was filtered
        logprobs: Token log probabilities (if requested)
    """
    index: int
    message: Message
    finish_reason: Optional[str] = None
    logprobs: Optional[Dict[str, Any]] = None


class ChatCompletionResponse(BaseModel):
    """
    OpenAI-compatible chat completion response.

    Returned for non-streaming requests after generation completes.

    Attributes:
        id: Unique identifier for this completion (format: chatcmpl-{uuid})
        object: Always "chat.completion" for this response type
        created: Unix timestamp when the completion was created
        model: The model used for generation
        choices: List of generated completions
        usage: Token usage statistics
        system_fingerprint: Backend system identifier (optional)

    Example:
        {
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "created": 1699000000,
            "model": "your-org/your-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
    """
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[UsageInfo] = None
    system_fingerprint: Optional[str] = None


# =============================================================================
# Streaming Response Models
# =============================================================================

class DeltaMessage(BaseModel):
    """
    Incremental message content for streaming responses.

    In streaming mode, content arrives in chunks. The delta contains
    only the new content since the last chunk.

    Attributes:
        role: Only present in the first chunk (establishes role)
        content: New content fragment (may be empty string)
        tool_calls: Incremental tool call data

    Example stream:
        Chunk 1: {"role": "assistant", "content": ""}
        Chunk 2: {"content": "Hello"}
        Chunk 3: {"content": " there"}
        Chunk 4: {"content": "!"}
    """
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class StreamChoice(BaseModel):
    """
    A single choice in a streaming chunk.

    Similar to Choice but uses delta instead of complete message.

    Attributes:
        index: Zero-based index of this choice
        delta: Incremental content update
        finish_reason: Only present in final chunk when generation completes
    """
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """
    Single chunk in a streaming response.

    Sent via Server-Sent Events (SSE) prefixed with "data: ".

    Attributes:
        id: Same ID for all chunks in a response
        object: Always "chat.completion.chunk"
        created: Unix timestamp
        model: The model generating the response
        choices: List of choice deltas (usually one)

    SSE Format:
        data: {"id": "chatcmpl-...", "object": "chat.completion.chunk", ...}
        data: {"id": "chatcmpl-...", "object": "chat.completion.chunk", ...}
        data: [DONE]
    """
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamChoice]


# =============================================================================
# Model Discovery Models (OpenRouter)
# =============================================================================

class ModelPricing(BaseModel):
    """
    Token pricing information for OpenRouter billing.

    Prices are strings to preserve decimal precision for micropayments.

    Attributes:
        prompt: Cost per input token in USD (e.g., "0.000008")
        completion: Cost per output token in USD (e.g., "0.000024")

    Note:
        Completion tokens are typically more expensive than prompt tokens
        because generation is more compute-intensive than processing input.
    """
    prompt: str
    completion: str


class ModelInfo(BaseModel):
    """
    Model metadata for OpenRouter discovery.

    Returned by the /api/v1/models endpoint to advertise model capabilities,
    pricing, and other metadata to OpenRouter and compatible routers.

    Attributes:
        id: Unique model identifier in org/model format
        object: Always "model"
        created: Unix timestamp when model was added
        owned_by: Organization identifier
        name: Human-readable display name
        context_length: Maximum context window in tokens
        pricing: Token pricing for billing
        quantization: Model precision (fp16, fp8, int8, etc.)
        supported_features: List of capabilities

    OpenRouter Discovery:
        OpenRouter periodically polls /api/v1/models to discover available
        models and update their catalog. This enables:
        - Automatic model listing in their UI
        - Accurate pricing display
        - Feature-based routing decisions

    Example:
        {
            "id": "your-org/llama-3.1-8b",
            "object": "model",
            "created": 1699000000,
            "owned_by": "your-org",
            "name": "Llama 3.1 8B Instruct",
            "context_length": 131072,
            "pricing": {"prompt": "0.000008", "completion": "0.000024"},
            "quantization": "fp16",
            "supported_features": ["tools", "json_mode", "streaming"]
        }
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
