from __future__ import annotations

"""Utility dataclass and helpers for sound-event representations."""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

__all__ = ["Event", "events_to_frame_labels"]


@dataclass(slots=True)
class Event:
    """Simple container for a sound event.

    Parameters
    ----------
    onset : float
        Event onset time in seconds.
    offset : float
        Event offset time in seconds.  Must be > ``onset``.
    label : str
        Class label for the event (e.g. "cat", "dog" ).
    """

    onset: float
    offset: float
    label: str

    # ---------------------------------------------------------------------
    # Validation helpers
    # ---------------------------------------------------------------------
    def __post_init__(self) -> None:  # noqa: D401
        if self.offset <= self.onset:
            raise ValueError(
                "Event offset must be > onset (got %.3f ≤ %.3f)"
                % (self.offset, self.onset)
            )

    # Convenient tuple-like representation for metrics
    def as_interval(self) -> Tuple[float, float]:
        """Return *(onset, offset)* as a tuple."""
        return (self.onset, self.offset)


# -------------------------------------------------------------------------
# Helper: events -> frame-level multi-hot matrix
# -------------------------------------------------------------------------


def events_to_frame_labels(
    events: Sequence[Event],
    fps: float,
    duration: float,
    label_set: Sequence[str] | None = None,
) -> Tuple[np.ndarray, List[str]]:
    """Convert a list of ``Event`` objects to frame-level multi-hot labels.

    The output is a *binary* matrix of shape ``(num_frames, n_classes)`` where
    ``num_frames = ceil(duration × fps)`` and each column corresponds to one
    class in **label_set**.

    Parameters
    ----------
    events : Sequence[Event]
        List of event annotations.
    fps : float
        Frames-per-second rate used to discretise time.
    duration : float
        Total clip duration **in seconds**.  Frames beyond this duration are
        not generated.
    label_set : Sequence[str] | None, optional
        Explicit ordering of class labels (determines column order).  If
        *None*, the set of labels is inferred from ``events`` **sorted
        alphabetically**.

    Returns
    -------
    Tuple[np.ndarray, List[str]]
        ``(frame_labels, label_set)`` where *frame_labels* is a ``uint8`` array
        of shape ``(num_frames, n_classes)`` and *label_set* is the class
        ordering used.
    """

    if fps <= 0:
        raise ValueError("fps must be positive (got %s)" % fps)
    if duration <= 0:
        raise ValueError("duration must be positive (got %s)" % duration)

    # ------------------------------------------------------------------
    # 1. Determine class ordering
    # ------------------------------------------------------------------
    if label_set is None:
        label_set = sorted({e.label for e in events})
    label_list = list(label_set)
    n_classes = len(label_list)

    # Map label → column index
    idx_for_label = {lbl: i for i, lbl in enumerate(label_list)}

    # ------------------------------------------------------------------
    # 2. Initialise output matrix
    # ------------------------------------------------------------------
    num_frames = int(np.ceil(duration * fps))
    frame_labels = np.zeros((num_frames, n_classes), dtype=np.uint8)

    # ------------------------------------------------------------------
    # 3. Fill in active regions for each event
    # ------------------------------------------------------------------
    for ev in events:
        if ev.label not in idx_for_label:
            # Ignore labels not in label_set (rare).
            continue
        col = idx_for_label[ev.label]

        # Convert times → frame indices (inclusive start, exclusive end)
        start_idx = int(np.floor(ev.onset * fps))
        end_idx = int(np.ceil(ev.offset * fps))

        start_idx = max(start_idx, 0)
        end_idx = min(end_idx, num_frames)
        if end_idx <= start_idx:
            continue  # zero-length after rounding

        frame_labels[start_idx:end_idx, col] = 1

    return frame_labels, label_list
