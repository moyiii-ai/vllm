# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

from vllm.benchmarks.lib.endpoint_request_func import (
    RequestFuncOutput,
    compute_phase_tpot,
    compute_request_phase_tpots,
    count_phase_tokens,
)


class _FakeTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return type("Enc", (), {"input_ids": list(range(len(text.split())))})()


def test_count_phase_tokens_prefers_tokenizer():
    tokenizer = _FakeTokenizer()
    assert count_phase_tokens("one two three", tokenizer, 1) == 3
    assert count_phase_tokens("", tokenizer, 4) == 4
    assert count_phase_tokens("", None, 0) == 0


def test_compute_phase_tpot_matches_standard_formula():
    start_time = 10.0
    phase_ttft = 1.0
    last_phase_time = start_time + phase_ttft + 0.6
    assert math.isclose(
        compute_phase_tpot(phase_ttft, last_phase_time, start_time, 4), 0.2
    )


def test_compute_phase_tpot_single_token_is_zero():
    assert compute_phase_tpot(0.5, 1.0, 0.0, 1) == 0.0


def test_compute_phase_tpot_unset_when_phase_missing():
    assert compute_phase_tpot(-1.0, 1.0, 0.0, 3) == -1.0
    assert compute_phase_tpot(0.5, -1.0, 0.0, 3) == -1.0


def test_compute_request_phase_tpots_for_thinking_and_answer():
    output = RequestFuncOutput(
        start_time=0.0,
        thinking_ttft=1.0,
        last_thinking_time=2.0,
        reasoning_text="a b c",
        answer_ttft=3.0,
        last_answer_time=4.5,
        content_text="x y",
    )
    thinking_tpot, answer_tpot = compute_request_phase_tpots(output, _FakeTokenizer())
    assert math.isclose(thinking_tpot, 0.5)
    assert math.isclose(answer_tpot, 1.5)


def test_compute_request_phase_tpots_answer_only():
    output = RequestFuncOutput(
        start_time=0.0,
        answer_ttft=1.0,
        last_answer_time=2.5,
        content_text="a b c",
    )
    thinking_tpot, answer_tpot = compute_request_phase_tpots(output, _FakeTokenizer())
    assert thinking_tpot == -1.0
    assert math.isclose(answer_tpot, 0.75)
