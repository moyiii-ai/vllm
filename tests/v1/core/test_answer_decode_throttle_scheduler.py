# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time

import torch

from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.config.reasoning import ReasoningConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager


def _make_reasoning_config() -> ReasoningConfig:
    config = ReasoningConfig(
        reasoning_parser="qwen3",
        reasoning_end_str="</think>",
    )
    config._reasoning_start_token_ids = [100]
    config._reasoning_end_token_ids = [99]
    config._enabled = True
    return config


def _create_throttle_scheduler(
    max_num_seqs: int = 3,
    throttle_ms: int = 60,
    max_num_scheduled_tokens: int | None = None,
) -> Scheduler:
    model_config = ModelConfig(
        model="facebook/opt-125m",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=8192,
        max_num_scheduled_tokens=max_num_scheduled_tokens or 1,
        max_model_len=8192,
        enable_chunked_prefill=True,
        answer_decode_throttle_ms=throttle_ms,
        async_scheduling=False,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=False,
    )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(),
        reasoning_config=_make_reasoning_config(),
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=10000,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    cache_config.num_gpu_blocks = 10000
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=16,
        log_stats=False,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


def _make_request(request_id: str, prompt_len: int = 3) -> Request:
    init_none_hash(sha256)
    block_hasher = get_request_block_hasher(16, sha256)
    sampling_params = SamplingParams(max_tokens=50, min_tokens=0)
    sampling_params.update_from_generation_config({}, eos_token_id=2)
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=block_hasher,
    )


def _register_running_decode_request(scheduler: Scheduler, request: Request) -> None:
    scheduler.requests[request.request_id] = request
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = request.num_prompt_tokens
    scheduler.kv_cache_manager.allocate_slots(request, request.num_prompt_tokens)


def test_throttled_answer_scheduled_before_running():
    scheduler = _create_throttle_scheduler()
    req_thinking = _make_request("thinking")
    req_answer = _make_request("answer")

    _register_running_decode_request(scheduler, req_thinking)
    _register_running_decode_request(scheduler, req_answer)
    for req in (req_thinking, req_answer):
        req.append_output_token_ids(10)

    req_thinking.thinking_phase_completed = False
    req_answer.thinking_phase_completed = True
    scheduler.running = [req_thinking]
    scheduler.answer_running = [req_answer]
    scheduler.answer_last_output_time[req_answer.request_id] = 0.0

    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {req_answer.request_id: 1}
    assert req_thinking.request_id not in output.num_scheduled_tokens


def test_all_eligible_answers_scheduled_when_budget_allows():
    scheduler = _create_throttle_scheduler(
        max_num_seqs=3,
        max_num_scheduled_tokens=2,
    )
    req_answer1 = _make_request("answer1")
    req_answer2 = _make_request("answer2")

    for req in (req_answer1, req_answer2):
        _register_running_decode_request(scheduler, req)
        req.append_output_token_ids(10)
        req.thinking_phase_completed = True

    scheduler.answer_running = [req_answer1, req_answer2]
    scheduler.answer_last_output_time[req_answer1.request_id] = 0.0
    scheduler.answer_last_output_time[req_answer2.request_id] = 0.0

    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {
        req_answer1.request_id: 1,
        req_answer2.request_id: 1,
    }


def test_eligible_answers_limited_by_token_budget():
    scheduler = _create_throttle_scheduler(
        max_num_seqs=3,
        max_num_scheduled_tokens=1,
    )
    req_answer1 = _make_request("answer1")
    req_answer2 = _make_request("answer2")

    for req in (req_answer1, req_answer2):
        _register_running_decode_request(scheduler, req)
        req.append_output_token_ids(10)
        req.thinking_phase_completed = True

    scheduler.answer_running = [req_answer1, req_answer2]
    scheduler.answer_last_output_time[req_answer1.request_id] = 0.0
    scheduler.answer_last_output_time[req_answer2.request_id] = 0.0

    output = scheduler.schedule()
    assert len(output.num_scheduled_tokens) == 1
    assert output.num_scheduled_tokens.keys() == {"answer1"}


def test_first_answer_decode_not_throttled_after_thinking_end():
    scheduler = _create_throttle_scheduler(throttle_ms=60)
    req_answer = _make_request("answer")
    _register_running_decode_request(scheduler, req_answer)
    req_answer.append_output_token_ids(10)
    req_answer.thinking_phase_completed = True
    scheduler.answer_running = [req_answer]

    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {req_answer.request_id: 1}


def test_all_eligible_past_interval_scheduled_not_fcfs_head_only():
    scheduler = _create_throttle_scheduler(
        max_num_seqs=3,
        max_num_scheduled_tokens=2,
        throttle_ms=50,
    )
    req_answer1 = _make_request("answer1")
    req_answer2 = _make_request("answer2")

    for req in (req_answer1, req_answer2):
        _register_running_decode_request(scheduler, req)
        req.append_output_token_ids(10)
        req.thinking_phase_completed = True
        req.answer_token_count = 1

    # answer2 is at the front of the queue but not yet eligible.
    scheduler.answer_running = [req_answer2, req_answer1]
    scheduler.answer_last_output_time[req_answer1.request_id] = 0.0
    scheduler.answer_last_output_time[req_answer2.request_id] = time.monotonic()

    eligible = scheduler._get_throttled_answer_requests()
    assert eligible == [req_answer1]

    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {req_answer1.request_id: 1}


def test_no_timestamp_is_immediately_eligible():
    scheduler = _create_throttle_scheduler(throttle_ms=60)
    req_answer = _make_request("answer")
    _register_running_decode_request(scheduler, req_answer)
    req_answer.append_output_token_ids(10)
    req_answer.thinking_phase_completed = True
    scheduler.answer_running = [req_answer]

    assert scheduler._get_throttled_answer_requests() == [req_answer]


def test_answer_not_scheduled_until_throttle_interval_elapses():
    scheduler = _create_throttle_scheduler(throttle_ms=60)
    req_answer = _make_request("answer")
    _register_running_decode_request(scheduler, req_answer)
    req_answer.append_output_token_ids(10)
    req_answer.thinking_phase_completed = True
    req_answer.answer_token_count = 1
    scheduler.answer_running = [req_answer]
    scheduler.answer_last_output_time[req_answer.request_id] = time.monotonic()

    output = scheduler.schedule()
    assert req_answer.request_id not in output.num_scheduled_tokens

    scheduler.answer_last_output_time[req_answer.request_id] = time.monotonic() - 0.1
    output2 = scheduler.schedule()
    assert output2.num_scheduled_tokens == {req_answer.request_id: 1}
