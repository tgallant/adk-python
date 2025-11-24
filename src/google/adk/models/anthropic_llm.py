# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Anthropic integration for Claude models."""

from __future__ import annotations

import base64
from functools import cached_property
import logging
import os
from typing import Any
from typing import AsyncGenerator
from typing import Iterable
from typing import Literal
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union

from anthropic import AnthropicVertex
from anthropic import AsyncAnthropicVertex
from anthropic import NOT_GIVEN
from anthropic import types as anthropic_types
from google.genai import types
from pydantic import BaseModel
from typing_extensions import override

from .base_llm import BaseLlm
from .llm_response import LlmResponse

if TYPE_CHECKING:
  from .llm_request import LlmRequest

__all__ = ["Claude"]

logger = logging.getLogger("google_adk." + __name__)


class ClaudeRequest(BaseModel):
  system_instruction: str
  messages: Iterable[anthropic_types.MessageParam]
  tools: list[anthropic_types.ToolParam]


def to_claude_role(role: Optional[str]) -> Literal["user", "assistant"]:
  if role in ["model", "assistant"]:
    return "assistant"
  return "user"


def to_google_genai_finish_reason(
    anthropic_stop_reason: Optional[str],
) -> types.FinishReason:
  if anthropic_stop_reason in ["end_turn", "stop_sequence", "tool_use"]:
    return "STOP"
  if anthropic_stop_reason == "max_tokens":
    return "MAX_TOKENS"
  return "FINISH_REASON_UNSPECIFIED"


def _is_image_part(part: types.Part) -> bool:
  return (
      part.inline_data
      and part.inline_data.mime_type
      and part.inline_data.mime_type.startswith("image")
  )


def part_to_message_block(
    part: types.Part,
) -> Union[
    anthropic_types.TextBlockParam,
    anthropic_types.ImageBlockParam,
    anthropic_types.ToolUseBlockParam,
    anthropic_types.ToolResultBlockParam,
    dict,  # For thinking blocks
]:
  # Handle thinking blocks (must check thought=True BEFORE text)
  # Thinking is stored as Part(text=..., thought=True, thought_signature=...)
  if part.text and hasattr(part, 'thought') and part.thought:
    thinking_block = {"type": "thinking", "thinking": part.text}
    if hasattr(part, 'thought_signature') and part.thought_signature:
      # thought_signature is stored as bytes in Part, but API expects base64 string
      thinking_block["signature"] = base64.b64encode(part.thought_signature).decode('utf-8')
      logger.debug(f"Including signature with thinking block")
    else:
      logger.warning(f"No signature found for thinking block - this may cause API errors")
    return thinking_block
  elif part.text:
    return anthropic_types.TextBlockParam(text=part.text, type="text")
  elif part.function_call:
    assert part.function_call.name

    return anthropic_types.ToolUseBlockParam(
        id=part.function_call.id or "",
        name=part.function_call.name,
        input=part.function_call.args,
        type="tool_use",
    )
  elif part.function_response:
    content = ""
    response_data = part.function_response.response

    # Handle response with content array
    if "content" in response_data and response_data["content"]:
      content_items = []
      for item in response_data["content"]:
        if isinstance(item, dict):
          # Handle text content blocks
          if item.get("type") == "text" and "text" in item:
            content_items.append(item["text"])
          else:
            # Handle other structured content
            content_items.append(str(item))
        else:
          content_items.append(str(item))
      content = "\n".join(content_items) if content_items else ""
    # Handle traditional result format
    elif "result" in response_data and response_data["result"]:
      # Transformation is required because the content is a list of dict.
      # ToolResultBlockParam content doesn't support list of dict. Converting
      # to str to prevent anthropic.BadRequestError from being thrown.
      content = str(response_data["result"])

    return anthropic_types.ToolResultBlockParam(
        tool_use_id=part.function_response.id or "",
        type="tool_result",
        content=content,
        is_error=False,
    )
  elif _is_image_part(part):
    data = base64.b64encode(part.inline_data.data).decode()
    return anthropic_types.ImageBlockParam(
        type="image",
        source=dict(
            type="base64", media_type=part.inline_data.mime_type, data=data
        ),
    )
  elif part.executable_code:
    return anthropic_types.TextBlockParam(
        type="text",
        text="Code:```python\n" + part.executable_code.code + "\n```",
    )
  elif part.code_execution_result:
    return anthropic_types.TextBlockParam(
        text="Execution Result:```code_output\n"
        + part.code_execution_result.output
        + "\n```",
        type="text",
    )

  raise NotImplementedError(f"Not supported yet: {part}")


def content_to_message_param(
    content: types.Content,
) -> anthropic_types.MessageParam:
  thinking_blocks = []
  other_blocks = []

  for part in content.parts or []:
    # Image data is not supported in Claude for model turns.
    if _is_image_part(part):
      logger.warning("Image data is not supported in Claude for model turns.")
      continue

    block = part_to_message_block(part)

    # Separate thinking blocks from other blocks
    # Anthropic requires thinking blocks to come FIRST in assistant messages
    if isinstance(block, dict) and block.get("type") == "thinking":
      thinking_blocks.append(block)
    else:
      other_blocks.append(block)

  # Thinking blocks MUST come first (Anthropic API requirement)
  message_block = thinking_blocks + other_blocks

  return {
      "role": to_claude_role(content.role),
      "content": message_block,
  }


def content_block_to_part(
    content_block: anthropic_types.ContentBlock,
) -> types.Part:
  if isinstance(content_block, anthropic_types.TextBlock):
    return types.Part.from_text(text=content_block.text)
  if isinstance(content_block, anthropic_types.ToolUseBlock):
    assert isinstance(content_block.input, dict)
    part = types.Part.from_function_call(
        name=content_block.name, args=content_block.input
    )
    part.function_call.id = content_block.id
    return part

  # Handle thinking blocks from Anthropic extended thinking feature
  # Thinking blocks have a 'thinking' attribute containing the reasoning text
  if hasattr(content_block, "thinking"):
    thinking_text = content_block.thinking
    signature = getattr(content_block, 'signature', None)
    logger.info(f"Received thinking block ({len(thinking_text)} chars, signature={'present' if signature else 'missing'})")
    # Return as Part with thought=True and preserve signature (standard GenAI format)
    return types.Part(text=thinking_text, thought=True, thought_signature=signature)

  # Alternative check: some versions may use type attribute
  if (
      hasattr(content_block, "type")
      and getattr(content_block, "type", None) == "thinking"
  ):
    thinking_text = str(content_block)
    signature = getattr(content_block, 'signature', None)
    logger.info(
        f"Received thinking block via type check ({len(thinking_text)} chars, signature={'present' if signature else 'missing'})"
    )
    # Return as Part with thought=True and preserve signature (standard GenAI format)
    return types.Part(text=thinking_text, thought=True, thought_signature=signature)

  raise NotImplementedError(
      f"Not supported yet: {type(content_block).__name__}"
  )


def streaming_event_to_llm_response(
    event: anthropic_types.MessageStreamEvent,
) -> Optional[LlmResponse]:
  """Convert Anthropic streaming events to ADK LlmResponse format.

  Args:
    event: Anthropic streaming event

  Returns:
    LlmResponse or None if event should be skipped
  """
  # Handle content block deltas
  if event.type == "content_block_delta":
    delta = event.delta

    # Text delta
    if delta.type == "text_delta":
      return LlmResponse(
          content=types.Content(
              role="model",
              parts=[types.Part.from_text(text=delta.text)],
          ),
          partial=True,
      )

    # Thinking delta
    elif delta.type == "thinking_delta":
      return LlmResponse(
          content=types.Content(
              role="model",
              parts=[types.Part(text=delta.thinking, thought=True)],
          ),
          partial=True,
      )

  # Handle message deltas (usage updates)
  elif event.type == "message_delta":
    if hasattr(event, "usage"):
      input_tokens = getattr(event.usage, "input_tokens", 0) or 0
      output_tokens = getattr(event.usage, "output_tokens", 0) or 0
      return LlmResponse(
          usage_metadata=types.GenerateContentResponseUsageMetadata(
              prompt_token_count=input_tokens,
              candidates_token_count=output_tokens,
              total_token_count=input_tokens + output_tokens,
          ),
      )

  # Ignore start/stop events
  return None


def message_to_generate_content_response(
    message: anthropic_types.Message,
) -> LlmResponse:
  logger.info("Received response from Claude.")
  logger.debug(
      "Claude response: %s",
      message.model_dump_json(indent=2, exclude_none=True),
  )

  return LlmResponse(
      content=types.Content(
          role="model",
          parts=[content_block_to_part(cb) for cb in message.content],
      ),
      usage_metadata=types.GenerateContentResponseUsageMetadata(
          prompt_token_count=message.usage.input_tokens,
          candidates_token_count=message.usage.output_tokens,
          total_token_count=(
              message.usage.input_tokens + message.usage.output_tokens
          ),
      ),
      # TODO: Deal with these later.
      # finish_reason=to_google_genai_finish_reason(message.stop_reason),
  )


def _update_type_string(value_dict: dict[str, Any]):
  """Updates 'type' field to expected JSON schema format."""
  if "type" in value_dict:
    value_dict["type"] = value_dict["type"].lower()

  if "items" in value_dict:
    # 'type' field could exist for items as well, this would be the case if
    # items represent primitive types.
    _update_type_string(value_dict["items"])

    if "properties" in value_dict["items"]:
      # There could be properties as well on the items, especially if the items
      # are complex object themselves. We recursively traverse each individual
      # property as well and fix the "type" value.
      for _, value in value_dict["items"]["properties"].items():
        _update_type_string(value)


def function_declaration_to_tool_param(
    function_declaration: types.FunctionDeclaration,
) -> anthropic_types.ToolParam:
  """Converts a function declaration to an Anthropic tool param."""
  assert function_declaration.name

  # Use parameters_json_schema if available, otherwise convert from parameters
  if function_declaration.parameters_json_schema:
    input_schema = function_declaration.parameters_json_schema
  else:
    properties = {}
    required_params = []
    if function_declaration.parameters:
      if function_declaration.parameters.properties:
        for key, value in function_declaration.parameters.properties.items():
          value_dict = value.model_dump(exclude_none=True)
          _update_type_string(value_dict)
          properties[key] = value_dict
      if function_declaration.parameters.required:
        required_params = function_declaration.parameters.required

    input_schema = {
        "type": "object",
        "properties": properties,
    }
    if required_params:
      input_schema["required"] = required_params

  return anthropic_types.ToolParam(
      name=function_declaration.name,
      description=function_declaration.description or "",
      input_schema=input_schema,
  )


class Claude(BaseLlm):
  """Integration with Claude models served from Vertex AI.

  Attributes:
    model: The name of the Claude model.
    max_tokens: The maximum number of tokens to generate.
    extra_headers: Optional extra headers to pass to the Anthropic API.
  """

  model: str = "claude-3-5-sonnet-v2@20241022"
  max_tokens: int = 8192
  extra_headers: Optional[dict[str, str]] = None

  @classmethod
  @override
  def supported_models(cls) -> list[str]:
    return [r"claude-3-.*", r"claude-.*-4.*"]

  @override
  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    messages = [
        content_to_message_param(content)
        for content in llm_request.contents or []
    ]
    tools = NOT_GIVEN
    if (
        llm_request.config
        and llm_request.config.tools
        and llm_request.config.tools[0].function_declarations
    ):
      tools = [
          function_declaration_to_tool_param(tool)
          for tool in llm_request.config.tools[0].function_declarations
      ]
    tool_choice = (
        anthropic_types.ToolChoiceAutoParam(type="auto")
        if llm_request.tools_dict
        else NOT_GIVEN
    )

    # Extract and convert thinking config from ADK to Anthropic format
    thinking = NOT_GIVEN

    if llm_request.config and llm_request.config.thinking_config:
      budget = llm_request.config.thinking_config.thinking_budget
      if budget:
        if budget == -1:
          raise ValueError(
              "Unlimited thinking budget (-1) is not supported with Claude."
          )
        elif budget > 0:

          thinking = {"type": "enabled", "budget_tokens": budget}
          logger.info(
              f"Extended thinking enabled (budget: {budget} tokens)"
          )
      else:
        logger.warning(f"Budget not given! budget={budget}")
    else:
      logger.warning(f"No thinking_config found in llm_request.config")

    # Use extra headers if provided
    extra_headers = self.extra_headers or NOT_GIVEN

    if stream:
      # Use streaming mode
      logger.info(
          f"Using streaming mode (stream={stream}, "
          f"has_thinking={thinking != NOT_GIVEN}, "
          f"large_max_tokens={self.max_tokens >= 8192})"
      )

      # Accumulators for text and thinking
      accumulated_text = ""
      accumulated_thinking = ""

      async with self._anthropic_client.messages.stream(
          model=llm_request.model,
          system=llm_request.config.system_instruction,
          messages=messages,
          tools=tools,
          tool_choice=tool_choice,
          max_tokens=self.max_tokens,
          thinking=thinking,
          extra_headers=extra_headers,
      ) as anthropic_stream:
        # Process streaming events
        async for event in anthropic_stream:
          # Convert Anthropic event to LlmResponse
          if llm_response := streaming_event_to_llm_response(event):
            # Track accumulated content
            is_thought = False
            if llm_response.content and llm_response.content.parts:
              for part in llm_response.content.parts:
                if part.text:
                  if hasattr(part, "thought") and part.thought:
                    accumulated_thinking += part.text
                    is_thought = True
                  else:
                    accumulated_text += part.text

            # If we have accumulated thinking and now getting text,
            # yield the accumulated thinking first
            # NOTE: This partial response is for UI display only
            # The final response with signature will be yielded after the stream ends
            if accumulated_thinking and accumulated_text and not is_thought:
              yield LlmResponse(
                  content=types.Content(
                      role="model",
                      parts=[
                          types.Part(text=accumulated_thinking, thought=True)
                      ],
                  ),
                  partial=True,
              )
              accumulated_thinking = ""  # Reset after yielding

            # Yield partial response (but skip individual thought deltas)
            if not is_thought:
              yield llm_response

        # Get final message with complete content blocks (includes signatures)
        final_message = await anthropic_stream.get_final_message()

        # Build final response from complete content blocks to preserve thinking signatures
        # IMPORTANT: Use final_message.content instead of accumulated strings
        # because accumulated strings don't have signatures
        if final_message.content:
          parts = [content_block_to_part(cb) for cb in final_message.content]
          input_tokens = final_message.usage.input_tokens
          output_tokens = final_message.usage.output_tokens
          yield LlmResponse(
              content=types.Content(role="model", parts=parts),
              usage_metadata=types.GenerateContentResponseUsageMetadata(
                  prompt_token_count=input_tokens,
                  candidates_token_count=output_tokens,
                  total_token_count=input_tokens + output_tokens,
              ),
          )

    else:
      # Non-streaming mode
      logger.info("Using non-streaming mode")
      message = await self._anthropic_client.messages.create(
          model=llm_request.model,
          system=llm_request.config.system_instruction,
          messages=messages,
          tools=tools,
          tool_choice=tool_choice,
          max_tokens=self.max_tokens,
          thinking=thinking,
          extra_headers=extra_headers,
      )
      yield message_to_generate_content_response(message)

  @cached_property
  def _anthropic_client(self) -> AsyncAnthropicVertex:
    if (
        "GOOGLE_CLOUD_PROJECT" not in os.environ
        or "GOOGLE_CLOUD_LOCATION" not in os.environ
    ):
      raise ValueError(
          "GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION must be set for using"
          " Anthropic on Vertex."
      )

    return AsyncAnthropicVertex(
        project_id=os.environ["GOOGLE_CLOUD_PROJECT"],
        region=os.environ["GOOGLE_CLOUD_LOCATION"],
    )
