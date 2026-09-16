"""Executable contracts for episode-relative temporal window queries."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WindowCheck:
    accepted: bool
    issues: list[str] = field(default_factory=list)


def expected_window_query(
    absolute_index: int,
    delta: int,
    episode_start: int,
    episode_end: int,
) -> tuple[int, bool]:
    """Return the clamped absolute query index and its padding flag."""

    requested = absolute_index + delta
    is_pad = requested < episode_start or requested >= episode_end
    query_index = max(episode_start, min(episode_end - 1, requested))
    return query_index, is_pad


def check_window_query(
    *,
    absolute_index: int,
    delta: int,
    episode_start: int,
    episode_end: int,
    observed_query_index: int,
    observed_is_pad: bool,
) -> WindowCheck:
    """Check a loader's window mapping against absolute episode boundaries."""

    issues: list[str] = []
    if episode_end <= episode_start:
        return WindowCheck(False, ["episode boundary is empty or reversed"])
    if not episode_start <= absolute_index < episode_end:
        issues.append("current absolute index lies outside its declared episode")
    expected_index, expected_pad = expected_window_query(
        absolute_index, delta, episode_start, episode_end
    )
    if observed_query_index != expected_index:
        issues.append(
            f"query index {observed_query_index} != expected absolute index "
            f"{expected_index}"
        )
    if bool(observed_is_pad) != expected_pad:
        issues.append(
            f"padding flag {bool(observed_is_pad)} != expected {expected_pad}"
        )
    return WindowCheck(not issues, issues)


def upstream_2610_buggy_query(
    *, relative_index: int, delta: int, episode_start: int, episode_end: int
) -> tuple[int, bool]:
    """Minimal transcription of LeRobot 7f40b3b's faulty code path.

    The upstream method accepted ``idx`` from the filtered dataset and compared
    it directly with absolute ``dataset_from_index`` / ``dataset_to_index``.
    """

    requested = relative_index + delta
    query_index = max(episode_start, min(episode_end - 1, requested))
    is_pad = requested < episode_start or requested >= episode_end
    return query_index, is_pad

