# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import asyncio
import json
import logging
import os
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    AsyncLLMServerManager,
    DictConfigWrap,
    register,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import (
    initialize_interactions_from_config,
)
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# Canonical metrics structure to ensure consistency across all workers
# Matches the structure in pipeline_tool.py
CANONICAL_METRICS = {
    # Performance metrics (main results)
    "perf/recall@10": 0.0,
    "perf/mrr": 0.0,
    "perf/recall@10_weighted": 0.0,
    "perf/mrr_weighted": 0.0,
    "recall@10": 0.0,
    "mrr": 0.0,
    # Plan structure metrics
    "plan/depth": 0,
    "plan/num_nodes": 0,
    "plan/breadth": 0,
    "plan/avg_complexity": 0.0,
    "plan_depth": 0,
    "num_nodes": 0,
    # Quality/format metrics
    "quality/passed_validation": 0.0,
    "quality/failure_modes": 0,
    "quality/score": 0.0,
    # Decision metrics
    "decision/keep_rate": 0.0,
    "decision/revise_rate": 0.0,
    "decision/conclude_rate": 0.0,
    # Latency metrics
    "latency/server_ms": 0.0,
    "latency/client_ms": 0.0,
    "latency/total_ms": 0.0,
    "latency": 0.0,
    "tool_client_latency": 0.0,
    # Reward metrics
    "tool_reward": 0.0,
    "tool/reward/base": 0.0,
    # System/error metrics
    "system/execution_errors": 0.0,
    "system/server_used": "none",
    "tool_execution_error": 0.0,
    "tool_error": "",  # Empty string for no error
    "server_used": "none",
    # Tool-specific performance
    "tool/performance/recall@10": 0.0,
    "tool/performance/mrr": 0.0,
    # Tool-specific format
    "tool/format/plan_depth": 0,
    "tool/format/num_nodes": 0,
    "tool/format/passed_hard_gate": 0.0,
    "tool/format/num_failure_modes": 0,
    "tool/format/quality": 0.0,
    # Tool-specific decision
    "tool/decision/is_keep": 0.0,
    "tool/decision/is_revise": 0.0,
    "tool/decision/is_conclude": 0.0,
    # Tool-specific latency
    "tool/latency/server": 0.0,
    "tool/latency/client": 0.0,
    "tool/latency/total": 0.0,
}


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"
    INTERACTING = "interacting"


class AgentData:
    """Encapsulates all state variables for the agent loop. AgentData is passed to tool calling in case that
    tool may need to access full history state. User can store any tool session data in `extra_fields`.
    """

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: list[Image.Image],
        video_data: list[tuple[torch.Tensor, dict[str, Any]]],
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        interaction: Optional[BaseInteraction] = None,
        interaction_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.messages = messages
        self.image_data = image_data
        self.video_data = video_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []

        # Extra fields for dynamic addition, e.g., tool session data
        self.extra_fields: dict[str, Any] = {}


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        **kwargs,
    ):
        super().__init__(trainer_config, server_manager, tokenizer, processor, **kwargs)
        config = trainer_config.config

        # Initialize tools from config file
        self.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        self.max_assistant_turns = (
            config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        )
        self.max_parallel_calls = (
            config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        )
        self.max_tool_response_length = (
            config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        )
        self.tool_response_truncate_side = (
            config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        )
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = (
            initialize_tools_from_config(tool_config_path) if tool_config_path else []
        )
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_schemas = [
            tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True)
            for tool in tool_list
        ]
        self.tool_parser = ToolParser.get_tool_parser(
            config.actor_rollout_ref.rollout.multi_turn.format, self.tokenizer
        )
        self.tool_parser_name = config.actor_rollout_ref.rollout.multi_turn.format

        self.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        self.response_length = config.actor_rollout_ref.rollout.response_length

        # Initialize interactions from config file
        self.interaction_config_file = (
            config.actor_rollout_ref.rollout.multi_turn.interaction_config_path
        )
        if self.interaction_config_file:
            self.interaction_map: dict[str, BaseInteraction] = (
                self._initialize_interactions(self.interaction_config_file)
            )

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(
            kwargs["raw_prompt"]
        )  # [{"role": "user", "content": "..." }, ...]

        # extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})
        extra_info = kwargs.get("extra_info", {})
        is_validation = kwargs.get("is_validation", False)

        # Initialize interaction if needed
        interaction = None
        interaction_kwargs = {}
        if self.interaction_config_file:
            interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]
            if "name" not in interaction_kwargs:
                raise ValueError("'name' key is required in interaction_kwargs")
            interaction_name = interaction_kwargs["name"]
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                    f"{list(self.interaction_map.keys())}"
                )
            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(request_id, **interaction_kwargs)
        # Create AgentData instance to encapsulate all state
        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
        )
        if isinstance(extra_info, dict):
            agent_data.extra_fields.update(extra_info)
            # Log to verify extra_info contains original_data with golden doc IDs
            original_data = extra_info.get("original_data", {})
            if original_data:
                logger.warning(
                    f"[AGENT_LOOP] request_id={request_id}: Set extra_fields with original_data: "
                    f"qid={original_data.get('qid')}, golden_docs={len(original_data.get('gold_doc_ids', []))} IDs"
                )
        
        # Add validation flag to extra_fields for tool access
        agent_data.extra_fields["is_validation"] = is_validation
        if is_validation:
            logger.warning(
                f"[AGENT_LOOP] request_id={request_id}: VALIDATION MODE - is_validation={is_validation}"
            )

        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED
            logger.warning(
                f"[AGENT_LOOP_STATUS] request_id={request_id}: entering state={state.name}"
            )

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[
            : len(agent_data.prompt_ids) - len(agent_data.response_mask)
        ]
        multi_modal_data = {}
        if agent_data.image_data is not None:
            multi_modal_data["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            multi_modal_data["videos"] = agent_data.video_data
        extra_fields = dict(agent_data.extra_fields)
        extra_fields.update(
            {
                "turn_scores": agent_data.turn_scores,
                "tool_rewards": agent_data.tool_rewards,
                "assistant_turns": agent_data.assistant_turns,
                "user_turns": agent_data.user_turns,
            }
        )

        logger.warning(
            f"[AGENT_LOOP_STATUS] request_id={request_id}: finalize_output "
            f"assistant_turns={agent_data.assistant_turns} user_turns={agent_data.user_turns} "
            f"response_tokens={len(response_ids)}"
        )
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=(
                agent_data.response_logprobs[: self.response_length]
                if agent_data.response_logprobs
                else None
            ),
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            extra_fields=extra_fields,
        )
        return output

    async def _handle_pending_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any]
    ) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        logger.warning(
            f"[AGENT_LOOP_STATUS] request_id={agent_data.request_id}: building prompt"
        )
        prompt_ids = await self.apply_chat_template(
            agent_data.messages,
            tools=self.tool_schemas,
            images=agent_data.image_data,
            videos=agent_data.video_data,
        )
        agent_data.prompt_ids = prompt_ids
        logger.warning(
            f"[AGENT_LOOP_STATUS] request_id={agent_data.request_id}: prompt_ready tokens={len(prompt_ids)}"
        )
        return AgentState.GENERATING

    async def _handle_generating_state(
        self,
        agent_data: AgentData,
        sampling_params: dict[str, Any],
        ignore_termination: bool = False,
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        add_messages: list[dict[str, Any]] = []

        logger.warning(
            f"[AGENT_LOOP_STATUS] request_id={agent_data.request_id}: generation_start"
        )
        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
            )

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts
        if output.finish_reason is not None:
            agent_data.extra_fields["finish_reason"] = output.finish_reason
        elif output.stop_reason is not None:
            agent_data.extra_fields["finish_reason"] = output.stop_reason

        logger.warning(
            f"[AGENT_LOOP_STATUS] request_id={agent_data.request_id}: generation_done "
            f"response_tokens={len(agent_data.response_ids)} finish_reason={agent_data.extra_fields.get('finish_reason')}"
        )

        # Check termination conditions
        if (
            not ignore_termination
            and len(agent_data.response_mask) >= self.response_length
        ):
            self._mark_pre_tool_termination_error(
                agent_data, "response_length_limit_reached"
            )
            logger.warning(
                f"[AGENT_LOOP_STATUS] request_id={agent_data.request_id}: terminate_reason=response_length"
            )
            return AgentState.TERMINATED
        if (
            self.max_assistant_turns
            and agent_data.assistant_turns >= self.max_assistant_turns
        ):
            self._mark_pre_tool_termination_error(
                agent_data, "max_assistant_turns_reached"
            )
            logger.warning(
                f"[AGENT_LOOP_STATUS] request_id={agent_data.request_id}: terminate_reason=max_assistant_turns"
            )
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            self._mark_pre_tool_termination_error(agent_data, "max_user_turns_reached")
            logger.warning(
                f"[AGENT_LOOP_STATUS] request_id={agent_data.request_id}: terminate_reason=max_user_turns"
            )
            return AgentState.TERMINATED

        # Extract tool calls
        self._update_tool_parser_context(agent_data)
        response_text, agent_data.tool_calls = (
            await self.tool_parser.extract_tool_calls(agent_data.response_ids)
        )
        parse_error = self._get_tool_parser_error()
        if parse_error:
            logger.warning(
                f"[AGENT_LOOP_STATUS] request_id={agent_data.request_id}: tool_parse_failed error={parse_error}"
            )
            qid, query = self._resolve_tool_context(agent_data)
            logger.warning(
                "Tool parsing failed: %s qid=%s query=%s finish_reason=%s parse_comment=%s response=%s",
                parse_error,
                qid,
                query,
                agent_data.extra_fields.get("finish_reason"),
                parse_error,
                response_text,
            )
            # Use canonical metrics with error-specific overrides for consistency
            metrics_update = CANONICAL_METRICS.copy()
            metrics_update.update(
                {
                    "tool_error": parse_error,
                    "tool_execution_error": 1.0,  # Indicates execution failure
                    "tool_reward": -1.0,  # Error penalty
                    "tool/reward/base": -1.0,  # Error penalty
                    "system/execution_errors": 1.0,  # Error occurred
                }
            )
            agent_data.metrics.update(metrics_update)
            agent_data.extra_fields["tool_error"] = parse_error
            agent_data.extra_fields["final_reward"] = -1.0
            return AgentState.TERMINATED

        logger.warning(
            f"[AGENT_LOOP_STATUS] request_id={agent_data.request_id}: tool_parse_ok "
            f"tool_calls={len(agent_data.tool_calls)}"
        )

        # Handle interaction if needed
        if self.interaction_config_file:
            assistant_message = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(
                    agent_data.response_ids, skip_special_tokens=True
                ),
            )
            add_messages.append({"role": "assistant", "content": assistant_message})
            agent_data.messages.extend(add_messages)

        # Determine next state
        if agent_data.tool_calls:
            logger.warning(
                f"[AGENT_LOOP_STATUS] request_id={agent_data.request_id}: next_state=PROCESSING_TOOLS"
            )
            return AgentState.PROCESSING_TOOLS
        elif self.interaction_config_file:
            logger.warning(
                f"[AGENT_LOOP_STATUS] request_id={agent_data.request_id}: next_state=INTERACTING"
            )
            return AgentState.INTERACTING
        else:
            self._mark_pre_tool_termination_error(agent_data, "no_tool_calls")
            logger.warning(
                f"[AGENT_LOOP_STATUS] request_id={agent_data.request_id}: next_state=TERMINATED(no_tools)"
            )
            return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = (
            []
        )  # Local variable instead of agent_data attribute

        tasks = []
        tool_call_names = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(
                self._call_tool(tool_call, agent_data.tools_kwargs, agent_data)
            )
            tool_call_names.append(tool_call.name)

        logger.warning(
            f"[AGENT_LOOP_STATUS] request_id={agent_data.request_id}: tool_calls_start "
            f"count={len(tasks)} names={tool_call_names}"
        )
        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)
        logger.warning(
            f"[AGENT_LOOP_STATUS] request_id={agent_data.request_id}: tool_calls_done responses={len(responses)}"
        )

        # Process tool responses and update multi_modal_data
        # Removed: agent_data.new_images_this_turn = []
        should_terminate = False
        for tool_response, tool_reward, tool_meta in responses:
            # Create message from tool response
            if tool_response.image or tool_response.video:
                # Multi-modal content with structured format
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                if tool_response.image:
                    content.append({"type": "image"})
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                # Text-only content
                message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)

            # Handle image data
            if tool_response.image:
                # Add new image data
                if isinstance(tool_response.image, list):
                    # Ensure all elements in the list are valid image objects
                    for img in tool_response.image:
                        if (
                            img is not None
                        ):  # Add a check to ensure the image is not None
                            new_images_this_turn.append(img)  # Using local variable
                else:
                    # Ensure the image is not None
                    if tool_response.image is not None:
                        new_images_this_turn.append(
                            tool_response.image
                        )  # Using local variable

            # Handle video data
            if tool_response.video:
                # Currently not supported, raise informative error
                logger.warning(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)
            if isinstance(tool_meta, dict) and tool_meta.get("tool_is_final"):
                should_terminate = True

        agent_data.messages.extend(add_messages)

        response_ids = await self._encode_tool_messages(
            add_messages=add_messages,
            tool_call_names=tool_call_names,
            new_images_this_turn=new_images_this_turn,
        )
        total_after_tools = len(agent_data.response_mask) + len(response_ids)
        logger.warning(
            f"[AGENT_LOOP_STATUS] request_id={agent_data.request_id}: tool_response_tokens={len(response_ids)} "
            f"current_response_tokens={len(agent_data.response_mask)} total_after_tools={total_after_tools} "
            f"response_budget={self.response_length}"
        )

        if total_after_tools >= self.response_length:
            logger.warning(
                f"[AGENT_LOOP_STATUS] request_id={agent_data.request_id}: terminate_reason=response_length_after_tools"
            )
            return AgentState.TERMINATED
        # Update prompt_ids and response_mask

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            for img in new_images_this_turn:
                agent_data.image_data.append(img)

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        if should_terminate:
            logger.warning(
                f"[AGENT_LOOP_STATUS] request_id={agent_data.request_id}: next_state=TERMINATED(after_tools)"
            )
            return AgentState.TERMINATED
        logger.warning(
            f"[AGENT_LOOP_STATUS] request_id={agent_data.request_id}: next_state=GENERATING(after_tools)"
        )
        return AgentState.GENERATING

    async def _encode_tool_messages(
        self,
        *,
        add_messages: list[dict[str, Any]],
        tool_call_names: list[str],
        new_images_this_turn: list[Any],
    ) -> list[int]:
        if self.tool_parser_name == "gpt-oss":
            logger.info("manually format tool responses for gpt-oss")
            tool_response_text = build_gpt_oss_tool_response_text(
                add_messages, tool_call_names
            )
            return await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.encode(
                    tool_response_text, add_special_tokens=False
                ),
            )

        return await self.apply_chat_template(
            add_messages,
            images=new_images_this_turn,
            videos=None,
            remove_system_prompt=True,
        )

    async def _handle_interacting_state(self, agent_data: AgentData) -> AgentState:
        """Handle the interacting state: get user input from interaction."""
        logger.warning(
            f"[AGENT_LOOP_STATUS] request_id={agent_data.request_id}: interacting_start"
        )
        (
            should_terminate_sequence,
            interaction_responses,
            reward,
            metrics,
        ) = await agent_data.interaction.generate_response(
            agent_data.request_id, agent_data.messages, **agent_data.interaction_kwargs
        )
        agent_data.user_turns += 1

        add_messages: list[dict[str, Any]] = [
            {"role": "user", "content": interaction_responses}
        ]
        agent_data.messages.extend(add_messages)

        if reward is not None:
            agent_data.turn_scores.append(reward)

        # Update prompt with user responses (similar to _handle_processing_tools_state)
        response_ids = await self.apply_chat_template(
            add_messages,
            remove_system_prompt=True,
        )

        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

        # double check prompt
        # Check termination condition
        if should_terminate_sequence:
            logger.warning(
                f"[AGENT_LOOP_STATUS] request_id={agent_data.request_id}: next_state=TERMINATED(after_interaction)"
            )
            return AgentState.TERMINATED
        else:
            logger.warning(
                f"[AGENT_LOOP_STATUS] request_id={agent_data.request_id}: next_state=GENERATING(after_interaction)"
            )
            return AgentState.GENERATING

    async def _call_tool(
        self,
        tool_call: FunctionCall,
        tools_kwargs: dict[str, Any],
        agent_data: AgentData,
    ) -> tuple[ToolResponse, float, dict]:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(
                create_kwargs=kwargs.get("create_kwargs", {})
            )
            tool_execution_response, tool_reward, res = await tool.execute(
                instance_id, tool_args, agent_data=agent_data
            )
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            return (
                ToolResponse(
                    text=f"Error when executing tool: {e}",
                ),
                0.0,
                {},
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if (
            tool_response_text
            and len(tool_response_text) > self.max_tool_response_length
        ):
            if self.tool_response_truncate_side == "left":
                tool_response_text = (
                    tool_response_text[: self.max_tool_response_length]
                    + "...(truncated)"
                )
            elif self.tool_response_truncate_side == "right":
                tool_response_text = (
                    "(truncated)..."
                    + tool_response_text[-self.max_tool_response_length :]
                )
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = (
                    tool_response_text[:length]
                    + "...(truncated)..."
                    + tool_response_text[-length:]
                )

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs), tool_reward, res

    def _update_tool_parser_context(self, agent_data: AgentData) -> None:
        set_context = getattr(self.tool_parser, "set_context", None)
        if not callable(set_context):
            return

        qid, query = self._resolve_tool_context(agent_data)
        finish_reason = agent_data.extra_fields.get("finish_reason")
        try:
            set_context(qid=qid, query=query, finish_reason=finish_reason)
        except TypeError:
            set_context(qid=qid, query=query)

    def _resolve_tool_context(self, agent_data: AgentData) -> tuple[str, Optional[str]]:
        qid = None
        query = None
        original_data = agent_data.extra_fields.get("original_data", {})
        if isinstance(original_data, dict):
            qid = original_data.get("qid")
            query = original_data.get("question")

        if qid is None:
            qid = agent_data.request_id
        if query is None:
            query = self._get_last_user_message(agent_data.messages)

        return qid, query

    def _get_last_user_message(self, messages: list[dict[str, Any]]) -> Optional[str]:
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    return content
        return None

    def _get_tool_parser_error(self) -> Optional[str]:
        get_error = getattr(self.tool_parser, "get_last_parse_error", None)
        if callable(get_error):
            return get_error()
        return None

    def _mark_pre_tool_termination_error(
        self, agent_data: AgentData, reason: str
    ) -> None:
        """Record a canonical tool error when terminating before any tool metrics exist."""
        if "performance_metrics" in agent_data.extra_fields:
            return
        if "tool_error" in agent_data.extra_fields:
            return

        error_msg = f"terminated_before_tool_execution:{reason}"
        agent_data.extra_fields["tool_error"] = error_msg
        agent_data.extra_fields["final_reward"] = -1.0
        agent_data.extra_fields.setdefault(
            "final_metrics",
            {
                "recall@10": 0.0,
                "mrr": 0.0,
                "plan_depth": 0,
                "num_nodes": 0,
                "plan_breadth": 0,
                "avg_node_complexity": 0.0,
            },
        )

        metrics_update = CANONICAL_METRICS.copy()
        metrics_update.update(
            {
                "tool_error": error_msg,
                "tool_execution_error": 1.0,
                "tool_reward": -1.0,
                "tool/reward/base": -1.0,
                "system/execution_errors": 1.0,
            }
        )
        agent_data.metrics.update(metrics_update)

    def _initialize_interactions(self, interaction_config_file):
        """Initialize interactions from configuration.
        Returns:
            dict[str, BaseInteraction]: A dictionary mapping interaction names to interaction instances.
        """
        if interaction_config_file is None:
            return {}

        interaction_map = initialize_interactions_from_config(interaction_config_file)
        return interaction_map
