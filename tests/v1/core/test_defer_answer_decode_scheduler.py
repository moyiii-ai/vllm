# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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


def _create_defer_scheduler(max_num_seqs: int = 2) -> Scheduler:
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
        max_num_scheduled_tokens=1,
        max_model_len=8192,
        enable_chunked_prefill=True,
        defer_answer_decode=True,
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


def test_answer_decode_deferred_while_thinking_decode_pending():
    scheduler = _create_defer_scheduler(max_num_seqs=3)
    req_thinking1 = _make_request("thinking1")
    req_thinking2 = _make_request("thinking2")
    req_answer = _make_request("answer")

    for req in (req_thinking1, req_thinking2, req_answer):
        _register_running_decode_request(scheduler, req)
        req.append_output_token_ids(10)

    req_thinking1.thinking_phase_completed = False
    req_thinking2.thinking_phase_completed = False
    req_answer.thinking_phase_completed = True

    scheduler.running = [req_thinking1, req_thinking2]
    scheduler.answer_running = [req_answer]

    output = scheduler.schedule()
    scheduled_ids = set(output.num_scheduled_tokens)
    assert req_answer.request_id not in scheduled_ids
    assert scheduled_ids == {req_thinking1.request_id}

    output2 = scheduler.schedule()
    assert req_answer.request_id not in output2.num_scheduled_tokens
    assert output2.num_scheduled_tokens == {req_thinking2.request_id: 1}

    output3 = scheduler.schedule()
    assert output3.num_scheduled_tokens == {req_answer.request_id: 1}


def test_request_moves_to_answer_running_when_phase_completes():
    scheduler = _create_defer_scheduler(max_num_seqs=1)
    request = _make_request("req-0")
    _register_running_decode_request(scheduler, request)
    request.append_output_token_ids(10)
    scheduler.running = [request]

    scheduler._update_request_with_output(request, [11, 99])

    assert request.thinking_phase_completed is True
    assert request not in scheduler.running
    assert request in scheduler.answer_running
