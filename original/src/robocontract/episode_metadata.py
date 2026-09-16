"""Cross-row invariants for episode boundary metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EpisodeMetadataCheck:
    accepted: bool
    issues: list[str] = field(default_factory=list)


def check_episode_index_chain(rows: list[dict[str, Any]]) -> EpisodeMetadataCheck:
    """Require lengths and global episode offsets to form one contiguous chain."""

    issues: list[str] = []
    expected_start = 0
    for position, row in enumerate(rows):
        episode_index = int(row["episode_index"])
        start = int(row["dataset_from_index"])
        end = int(row["dataset_to_index"])
        length = int(row["length"])
        if episode_index != position:
            issues.append(
                f"row {position} declares episode_index={episode_index}"
            )
        if end - start != length:
            issues.append(
                f"episode {episode_index} span {end - start} != length {length}"
            )
        if start != expected_start:
            issues.append(
                f"episode {episode_index} starts at {start}, expected {expected_start}"
            )
        expected_start = end
    return EpisodeMetadataCheck(not issues, issues)


def simulate_v21_to_v30_offsets(
    *,
    episode_lengths: list[int],
    episode_sizes_bytes: list[int],
    target_file_size_mb: float,
    reset_global_frame_count: bool,
) -> list[dict[str, int]]:
    """Execute the indexing loop used by the historical v2.1→v3 converter.

    ``reset_global_frame_count=True`` transcribes the faulty line removed by
    merged LeRobot PR #2057.  File sizes use MiB, matching the upstream helper.
    """

    if len(episode_lengths) != len(episode_sizes_bytes):
        raise ValueError("episode length and file-size arrays differ")
    size_mb = 0.0
    num_frames = 0
    rows: list[dict[str, int]] = []
    for episode_index, (length, size_bytes) in enumerate(
        zip(episode_lengths, episode_sizes_bytes)
    ):
        episode_size_mb = size_bytes / (1024 * 1024)
        rows.append(
            {
                "episode_index": episode_index,
                "dataset_from_index": num_frames,
                "dataset_to_index": num_frames + length,
                "length": length,
            }
        )
        size_mb += episode_size_mb
        num_frames += length
        if size_mb < target_file_size_mb:
            continue

        # The current episode starts the next file.  Before PR #2057, the
        # converter incorrectly reset a dataset-global counter at this point.
        size_mb = episode_size_mb
        if reset_global_frame_count:
            num_frames = length
    return rows
