# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config.reasoning import ReasoningConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.utils import (
    check_stop,
    compute_thinking_answer_split,
    update_thinking_answer_phase,
)
from vllm.v1.request import Request, RequestStatus


def _make_reasoning_config(
    *,
    end_token_ids: list[int],
    end_str: str = "</think>",
) -> ReasoningConfig:
    config = ReasoningConfig(
        reasoning_parser="qwen3",
        reasoning_end_str=end_str,
    )
    config._reasoning_start_token_ids = [100]
    config._reasoning_end_token_ids = end_token_ids
    config._enabled = True
    return config


def _make_request(max_tokens: int = 100) -> Request:
    sampling_params = SamplingParams(max_tokens=max_tokens, min_tokens=0)
    sampling_params.update_from_generation_config({}, eos_token_id=2)
    request = Request(
        request_id="req-1",
        prompt_token_ids=[1, 2, 3],
        sampling_params=sampling_params,
        pooling_params=None,
    )
    return request


def test_compute_thinking_answer_split_without_reasoning():
    thinking, answer, completed = compute_thinking_answer_split([10, 11, 12], None)
    assert thinking == 0
    assert answer == 3
    assert completed is True


def test_compute_thinking_answer_split_with_end_marker():
    config = _make_reasoning_config(end_token_ids=[99])
    output = [10, 11, 12, 99, 20, 21]
    thinking, answer, completed = compute_thinking_answer_split(output, config)
    assert thinking == 3
    assert answer == 2
    assert completed is True


def test_compute_thinking_answer_split_truncated_thinking():
    config = _make_reasoning_config(end_token_ids=[99, 100])
    output = [10, 11, 12, 13]
    thinking, answer, completed = compute_thinking_answer_split(output, config)
    assert thinking == 4
    assert answer == 0
    assert completed is False


def test_update_thinking_answer_phase_detects_end_marker():
    request = _make_request()
    config = _make_reasoning_config(end_token_ids=[99])
    for token_id in [10, 11, 99, 20]:
        request.append_output_token_ids(token_id)
        update_thinking_answer_phase(request, config)

    assert request.thinking_token_count == 2
    assert request.answer_token_count == 1
    assert request.thinking_phase_completed is True


def test_check_stop_updates_thinking_stats_on_length_cap():
    request = _make_request(max_tokens=4)
    config = _make_reasoning_config(end_token_ids=[99])
    for token_id in [10, 11, 12, 13]:
        request.append_output_token_ids(token_id)
        stopped = check_stop(request, max_model_len=100, reasoning_config=config)

    assert stopped is True
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert request.thinking_token_count == 4
    assert request.answer_token_count == 0
    assert request.thinking_phase_completed is False
