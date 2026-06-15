# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.sampling_params import RepetitionDetectionParams
from vllm.v1.request import Request, RequestStatus

if TYPE_CHECKING:
    from vllm.config.reasoning import ReasoningConfig

logger = init_logger(__name__)


def _has_repeating_pattern(
    token_ids: Sequence[int],
    pattern_len: int,
    repetition_min_count: int,
) -> bool:
    """Check if the tail of token_ids contains a repeating pattern.

    Compares the last pattern_len tokens against the preceding
    (repetition_min_count - 1) repetitions of the same length.
    """
    for n in range(1, pattern_len + 1):
        target_token = token_ids[-n]
        for m in range(1, repetition_min_count):
            if token_ids[-(pattern_len * m + n)] != target_token:
                return False
    return True


def check_sequence_repetition(
    token_ids: Sequence[int],
    params: RepetitionDetectionParams,
) -> bool:
    """Check if a sequence of token IDs has a repetition pattern.
    Args:
        token_ids: List of token IDs
        params: Repetition detection parameters.
    Returns:
        True if a repetition pattern is found, False otherwise.
    """
    max_pattern_size = params.max_pattern_size
    min_pattern_size = params.min_pattern_size
    min_count = params.min_count

    if min_pattern_size <= 0:
        min_pattern_size = 1

    if max_pattern_size <= 0 or min_count < 2 or min_pattern_size > max_pattern_size:
        return False

    for pattern_len in range(
        min_pattern_size,
        max_pattern_size + 1,
    ):
        if pattern_len * min_count > len(token_ids):
            return False

        if _has_repeating_pattern(token_ids, pattern_len, min_count):
            return True

    return False


def remove_all(lst: list, items_to_remove: set) -> list:
    """Remove all items from a list that are in the items_to_remove set.

    This method optimizes for the common case of removing a single item,
    falling back to list comprehension for multiple items.

    Args:
        lst: The list to remove items from
        items_to_remove: Set of items to remove

    Returns:
        Either the modified original list (for single item removal) or
        a new list (for multiple item removal). Callers should use the
        returned value.

    Note:
        For single item removal, this modifies the original list in-place
        and returns it. For multiple items, it creates and returns a new list.
    """
    if not items_to_remove:
        return lst

    if len(items_to_remove) == 1:
        # Fast path for single item removal (most common case)
        item = next(iter(items_to_remove))
        with contextlib.suppress(ValueError):
            lst.remove(item)
        return lst
    # For multiple items, use list comprehension
    return [item for item in lst if item not in items_to_remove]


def _find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> int | None:
    """Return the start index of ``needle`` in ``haystack``, or ``None``."""
    needle_len = len(needle)
    if needle_len == 0 or len(haystack) < needle_len:
        return None
    for start in range(len(haystack) - needle_len + 1):
        if list(haystack[start : start + needle_len]) == list(needle):
            return start
    return None


def compute_thinking_answer_split(
    output_token_ids: Sequence[int],
    reasoning_config: "ReasoningConfig | None",
) -> tuple[int, int, bool]:
    """Split generated tokens into thinking and answer counts.

    Returns:
        A tuple of (thinking_token_count, answer_token_count,
        thinking_phase_completed). When reasoning is disabled or unavailable,
        all output tokens are counted as answer tokens and
        ``thinking_phase_completed`` is ``True``.
    """
    num_output = len(output_token_ids)
    if num_output == 0:
        return 0, 0, True

    if reasoning_config is None or not reasoning_config.enabled:
        return 0, num_output, True

    end_token_ids = reasoning_config.reasoning_end_token_ids
    if not end_token_ids:
        return 0, num_output, True

    end_pos = _find_subsequence(output_token_ids, end_token_ids)
    if end_pos is not None:
        thinking_count = end_pos
        answer_count = num_output - end_pos - len(end_token_ids)
        return thinking_count, max(answer_count, 0), True

    return num_output, 0, False


def update_thinking_answer_phase(
    request: Request,
    reasoning_config: "ReasoningConfig | None",
) -> bool:
    """Update per-request thinking/answer stats after a new output token.

    Also checks whether the reasoning end token sequence (e.g.
    ``</think>`` for Qwen3) has been generated.

    Returns:
        ``True`` if the thinking phase end marker was just completed.
    """
    prev_completed = request.thinking_phase_completed
    thinking_count, answer_count, completed = compute_thinking_answer_split(
        request.output_token_ids,
        reasoning_config,
    )
    request.thinking_token_count = thinking_count
    request.answer_token_count = answer_count
    request.thinking_phase_completed = completed
    return completed and not prev_completed


def log_thinking_answer_stats(
    request: Request,
    reasoning_config: "ReasoningConfig | None",
) -> None:
    """Log thinking/answer token stats when a request finishes."""
    if reasoning_config is None or not reasoning_config.enabled:
        return

    thinking_count, answer_count, completed = compute_thinking_answer_split(
        request.output_token_ids,
        reasoning_config,
    )
    request.thinking_token_count = thinking_count
    request.answer_token_count = answer_count
    request.thinking_phase_completed = completed

    end_token_ids = reasoning_config.reasoning_end_token_ids or []
    truncation_note = (
        "thinking completed normally"
        if completed
        else (
            "thinking truncated before reasoning end token "
            f"{reasoning_config.reasoning_end_str!r}"
        )
    )
    logger.info(
        "Request %s thinking/answer stats: thinking_tokens=%d, "
        "answer_tokens=%d, total_output_tokens=%d, "
        "thinking_phase_completed=%s (%s), status=%s, stop_reason=%s, "
        "reasoning_end_token_ids=%s",
        request.request_id,
        thinking_count,
        answer_count,
        request.num_output_tokens,
        completed,
        truncation_note,
        request.status.name,
        request.stop_reason,
        end_token_ids,
    )


def check_stop(
    request: Request,
    max_model_len: int,
    reasoning_config: "ReasoningConfig | None" = None,
) -> bool:
    assert not request.pooling_params

    sampling_params = request.sampling_params
    assert sampling_params is not None

    if request.num_output_tokens < sampling_params.min_tokens:
        if reasoning_config is not None:
            update_thinking_answer_phase(request, reasoning_config)
        return False

    thinking_end_just_seen = False
    if reasoning_config is not None:
        thinking_end_just_seen = update_thinking_answer_phase(
            request, reasoning_config
        )
        if thinking_end_just_seen:
            logger.debug(
                "Request %s reached reasoning end token %r (token_ids=%s); "
                "entering answer phase",
                request.request_id,
                reasoning_config.reasoning_end_str,
                reasoning_config.reasoning_end_token_ids,
            )

    last_token_id = request.output_token_ids[-1]
    if last_token_id == sampling_params.eos_token_id:
        request.status = RequestStatus.FINISHED_STOPPED
        return True

    if last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = RequestStatus.FINISHED_STOPPED
        request.stop_reason = last_token_id
        return True
    if (
        request.num_tokens >= max_model_len
        or request.num_output_tokens >= request.max_tokens
    ):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True

    repetition_detection = sampling_params.repetition_detection
    if repetition_detection is not None and (
        check_sequence_repetition(
            request.output_token_ids,
            repetition_detection,
        )
    ):
        request.status = RequestStatus.FINISHED_REPETITION
        request.stop_reason = "repetition_detected"
        return True

    return False
