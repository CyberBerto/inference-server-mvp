"""
Tests for Pydantic models validation and serialization.

These tests verify that the request/response models properly validate
input and serialize output according to OpenAI API specifications.
"""

import pytest
from pydantic import ValidationError

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models import (
    Message,
    ToolFunction,
    Tool,
    ResponseFormat,
    ChatCompletionRequest,
    UsageInfo,
    Choice,
    ChatCompletionResponse,
    DeltaMessage,
    StreamChoice,
    ChatCompletionChunk,
    ModelPricing,
    ModelInfo,
)


class TestMessageModel:
    """Tests for Message model validation."""

    def test_valid_user_message(self):
        """User message with content should be valid."""
        msg = Message(role="user", content="Hello!")
        assert msg.role == "user"
        assert msg.content == "Hello!"

    def test_valid_system_message(self):
        """System message should be valid."""
        msg = Message(role="system", content="You are helpful.")
        assert msg.role == "system"

    def test_valid_assistant_message(self):
        """Assistant message should be valid."""
        msg = Message(role="assistant", content="Hi there!")
        assert msg.role == "assistant"

    def test_valid_tool_message(self):
        """Tool message with tool_call_id should be valid."""
        msg = Message(role="tool", content="72°F", tool_call_id="call_123")
        assert msg.role == "tool"
        assert msg.tool_call_id == "call_123"

    def test_invalid_role_rejected(self):
        """Invalid role should raise ValidationError."""
        with pytest.raises(ValidationError):
            Message(role="invalid", content="test")

    def test_message_with_name(self):
        """Message with optional name field should be valid."""
        msg = Message(role="user", content="Hello", name="John")
        assert msg.name == "John"

    def test_message_with_tool_calls(self):
        """Assistant message with tool_calls should be valid."""
        msg = Message(
            role="assistant",
            tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "test"}}]
        )
        assert len(msg.tool_calls) == 1

    def test_message_content_optional(self):
        """Content can be None (for tool call messages)."""
        msg = Message(role="assistant", content=None)
        assert msg.content is None


class TestToolModels:
    """Tests for Tool and ToolFunction models."""

    def test_valid_tool_function(self):
        """Tool function with all fields should be valid."""
        func = ToolFunction(
            name="get_weather",
            description="Get weather for a location",
            parameters={"type": "object", "properties": {"location": {"type": "string"}}}
        )
        assert func.name == "get_weather"
        assert func.description is not None
        assert func.parameters is not None

    def test_minimal_tool_function(self):
        """Tool function with only name should be valid."""
        func = ToolFunction(name="my_function")
        assert func.name == "my_function"
        assert func.description is None
        assert func.parameters is None

    def test_valid_tool(self):
        """Tool with function should be valid."""
        tool = Tool(
            type="function",
            function=ToolFunction(name="test_func")
        )
        assert tool.type == "function"
        assert tool.function.name == "test_func"

    def test_tool_type_defaults_to_function(self):
        """Tool type should default to 'function'."""
        tool = Tool(function=ToolFunction(name="test"))
        assert tool.type == "function"


class TestResponseFormat:
    """Tests for ResponseFormat model."""

    def test_default_type_is_text(self):
        """Response format should default to text."""
        fmt = ResponseFormat()
        assert fmt.type == "text"

    def test_json_object_type(self):
        """JSON object type should be valid."""
        fmt = ResponseFormat(type="json_object")
        assert fmt.type == "json_object"

    def test_invalid_type_rejected(self):
        """Invalid format type should raise ValidationError."""
        with pytest.raises(ValidationError):
            ResponseFormat(type="xml")


class TestChatCompletionRequest:
    """Tests for ChatCompletionRequest validation."""

    def test_minimal_valid_request(self):
        """Request with model and messages should be valid."""
        req = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")]
        )
        assert req.model == "test-model"
        assert len(req.messages) == 1

    def test_default_values(self):
        """Default values should be set correctly."""
        req = ChatCompletionRequest(
            model="test",
            messages=[Message(role="user", content="test")]
        )
        assert req.max_tokens == 4096
        assert req.temperature == 0.7
        assert req.top_p == 1.0
        assert req.stream is False

    def test_temperature_valid_range(self):
        """Temperature within 0-2 should be valid."""
        req = ChatCompletionRequest(
            model="test",
            messages=[Message(role="user", content="test")],
            temperature=0.0
        )
        assert req.temperature == 0.0

        req = ChatCompletionRequest(
            model="test",
            messages=[Message(role="user", content="test")],
            temperature=2.0
        )
        assert req.temperature == 2.0

    def test_temperature_below_range_rejected(self):
        """Temperature below 0 should raise ValidationError."""
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="test",
                messages=[Message(role="user", content="test")],
                temperature=-0.1
            )

    def test_temperature_above_range_rejected(self):
        """Temperature above 2 should raise ValidationError."""
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="test",
                messages=[Message(role="user", content="test")],
                temperature=2.1
            )

    def test_max_tokens_valid_range(self):
        """max_tokens within range should be valid."""
        req = ChatCompletionRequest(
            model="test",
            messages=[Message(role="user", content="test")],
            max_tokens=1
        )
        assert req.max_tokens == 1

        req = ChatCompletionRequest(
            model="test",
            messages=[Message(role="user", content="test")],
            max_tokens=131072
        )
        assert req.max_tokens == 131072

    def test_max_tokens_zero_rejected(self):
        """max_tokens of 0 should raise ValidationError."""
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="test",
                messages=[Message(role="user", content="test")],
                max_tokens=0
            )

    def test_max_tokens_exceeds_limit_rejected(self):
        """max_tokens exceeding 131072 should raise ValidationError."""
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="test",
                messages=[Message(role="user", content="test")],
                max_tokens=131073
            )

    def test_top_p_valid_range(self):
        """top_p within 0-1 should be valid."""
        req = ChatCompletionRequest(
            model="test",
            messages=[Message(role="user", content="test")],
            top_p=0.0
        )
        assert req.top_p == 0.0

        req = ChatCompletionRequest(
            model="test",
            messages=[Message(role="user", content="test")],
            top_p=1.0
        )
        assert req.top_p == 1.0

    def test_top_p_out_of_range_rejected(self):
        """top_p outside 0-1 should raise ValidationError."""
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="test",
                messages=[Message(role="user", content="test")],
                top_p=1.1
            )

    def test_stop_as_string(self):
        """stop can be a single string."""
        req = ChatCompletionRequest(
            model="test",
            messages=[Message(role="user", content="test")],
            stop="END"
        )
        assert req.stop == "END"

    def test_stop_as_list(self):
        """stop can be a list of strings."""
        req = ChatCompletionRequest(
            model="test",
            messages=[Message(role="user", content="test")],
            stop=["END", "STOP"]
        )
        assert req.stop == ["END", "STOP"]

    def test_with_tools(self):
        """Request with tools should be valid."""
        req = ChatCompletionRequest(
            model="test",
            messages=[Message(role="user", content="test")],
            tools=[Tool(function=ToolFunction(name="my_func"))]
        )
        assert len(req.tools) == 1

    def test_with_response_format(self):
        """Request with response_format should be valid."""
        req = ChatCompletionRequest(
            model="test",
            messages=[Message(role="user", content="test")],
            response_format=ResponseFormat(type="json_object")
        )
        assert req.response_format.type == "json_object"

    def test_vllm_extensions(self):
        """vLLM-specific parameters should be accepted."""
        req = ChatCompletionRequest(
            model="test",
            messages=[Message(role="user", content="test")],
            best_of=3,
            use_beam_search=True,
            skip_special_tokens=False
        )
        assert req.best_of == 3
        assert req.use_beam_search is True
        assert req.skip_special_tokens is False


class TestResponseModels:
    """Tests for response model construction."""

    def test_usage_info(self):
        """UsageInfo should accept token counts."""
        usage = UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30

    def test_choice(self):
        """Choice should contain message and metadata."""
        choice = Choice(
            index=0,
            message=Message(role="assistant", content="Hello!"),
            finish_reason="stop"
        )
        assert choice.index == 0
        assert choice.message.content == "Hello!"
        assert choice.finish_reason == "stop"

    def test_chat_completion_response(self):
        """ChatCompletionResponse should be constructable."""
        response = ChatCompletionResponse(
            id="chatcmpl-123",
            created=1699000000,
            model="test-model",
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content="Hi!"),
                    finish_reason="stop"
                )
            ],
            usage=UsageInfo(prompt_tokens=5, completion_tokens=2, total_tokens=7)
        )
        assert response.id == "chatcmpl-123"
        assert response.object == "chat.completion"
        assert len(response.choices) == 1


class TestStreamingModels:
    """Tests for streaming response models."""

    def test_delta_message(self):
        """DeltaMessage should accept partial content."""
        delta = DeltaMessage(content="Hello")
        assert delta.content == "Hello"
        assert delta.role is None

    def test_delta_message_with_role(self):
        """First delta typically includes role."""
        delta = DeltaMessage(role="assistant", content="")
        assert delta.role == "assistant"

    def test_stream_choice(self):
        """StreamChoice should contain delta."""
        choice = StreamChoice(
            index=0,
            delta=DeltaMessage(content="Hi"),
            finish_reason=None
        )
        assert choice.delta.content == "Hi"
        assert choice.finish_reason is None

    def test_chat_completion_chunk(self):
        """ChatCompletionChunk should be constructable."""
        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            created=1699000000,
            model="test-model",
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaMessage(content="Hello"),
                    finish_reason=None
                )
            ]
        )
        assert chunk.object == "chat.completion.chunk"
        assert chunk.choices[0].delta.content == "Hello"


class TestModelInfoModels:
    """Tests for OpenRouter model discovery models."""

    def test_model_pricing(self):
        """ModelPricing should accept string prices."""
        pricing = ModelPricing(prompt="0.000008", completion="0.000024")
        assert pricing.prompt == "0.000008"
        assert pricing.completion == "0.000024"

    def test_model_info(self):
        """ModelInfo should contain all required fields."""
        info = ModelInfo(
            id="org/model",
            created=1699000000,
            owned_by="org",
            name="Test Model",
            context_length=128000,
            pricing=ModelPricing(prompt="0.000008", completion="0.000024"),
            quantization="fp16",
            supported_features=["streaming", "tools"]
        )
        assert info.id == "org/model"
        assert info.object == "model"
        assert info.context_length == 128000
        assert "streaming" in info.supported_features


class TestModelSerialization:
    """Tests for model serialization to dict/JSON."""

    def test_message_to_dict(self):
        """Message should serialize correctly."""
        msg = Message(role="user", content="Hello")
        data = msg.model_dump()
        assert data["role"] == "user"
        assert data["content"] == "Hello"

    def test_request_to_dict(self):
        """ChatCompletionRequest should serialize correctly."""
        req = ChatCompletionRequest(
            model="test",
            messages=[Message(role="user", content="test")],
            temperature=0.5
        )
        data = req.model_dump()
        assert data["model"] == "test"
        assert data["temperature"] == 0.5
        assert len(data["messages"]) == 1

    def test_response_to_dict(self):
        """ChatCompletionResponse should serialize correctly."""
        response = ChatCompletionResponse(
            id="test-id",
            created=1699000000,
            model="test-model",
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content="Hi"),
                    finish_reason="stop"
                )
            ]
        )
        data = response.model_dump()
        assert data["id"] == "test-id"
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "Hi"
