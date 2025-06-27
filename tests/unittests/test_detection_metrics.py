import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from representation_learning.metrics.strong_detection.vox_strong_metrics import StrongDetectionF1Metric
from representation_learning.metrics.strong_detection.framewise_detection_metrics import StrongDetectionF1Tensor


def _write_selection_table(path: Path, events: list[tuple[float, float, str]]) -> None:
    """Helper to write a minimal Raven selection table TSV."""
    cols = [
        "Selection",
        "View",
        "Channel",
        "Begin Time (s)",
        "End Time (s)",
        "Low Freq (Hz)",
        "High Freq (Hz)",
        "Annotation",
        "Notes",
    ]
    rows = []
    for i, (on, off, ann) in enumerate(events, 1):
        rows.append([i, "spectrogram 1", 1, on, off, 0, 0, ann, ""])
    pd.DataFrame(rows, columns=cols).to_csv(path, sep="\t", index=False)


def test_strong_f1_tensor_matches_file_metric_binary():
    fps = 50.0

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # --------------- create synthetic events -------------------
        ref_events = [(0.0, 1.0, "bird"), (2.0, 3.0, "bird")]
        pred_events = [(0.1, 1.1, "bird"), (4.0, 5.0, "bird")]  # 1 TP, 1 FP, 1 FN

        # Raven selection tables
        ref_sel = tmp_path / "ref.tsv"
        pred_sel = tmp_path / "pred.tsv"
        _write_selection_table(ref_sel, ref_events)
        _write_selection_table(pred_sel, pred_events)

        # CSV wrappers expected by StrongDetectionF1Metric
        ref_csv = tmp_path / "ref.csv"
        pred_csv = tmp_path / "pred.csv"
        pd.DataFrame({
            "fn": ["clip1"],
            "audio_fp": [np.nan],
            "selection_table_fp": [ref_sel.name],
        }).to_csv(ref_csv, index=False)
        pd.DataFrame({
            "fn": ["clip1"],
            "audio_fp": [np.nan],
            "selection_table_fp": [pred_sel.name],
        }).to_csv(pred_csv, index=False)

        # ---------- original file-based metric ---------------------
        metric_file = StrongDetectionF1Metric(
            iou=0.5,
            base_path_reference=str(tmp_path),
            base_path_prediction=str(tmp_path),
        )
        # Use default_duration since we don't have audio files
        f1_file = metric_file(ref_csv, pred_csv, default_duration=6.0)

        # ---------- tensor-based metric ----------------------------
        T = int(6 * fps)  # 6 seconds long clip for safety
        gt_frames = np.zeros(T, dtype=int)
        pred_frames = np.zeros(T, dtype=int)
        # Ground truth events
        gt_frames[int(0 * fps) : int(1 * fps)] = 1
        gt_frames[int(2 * fps) : int(3 * fps)] = 1
        # Predictions
        pred_frames[int(0.1 * fps) : int(1.1 * fps)] = 1
        pred_frames[int(4 * fps) : int(5 * fps)] = 1

        logits = torch.from_numpy(pred_frames[None, :, None]).float() * 10  # (1,T,1)

        # Build event annotations list [[[events_class0]]]
        from representation_learning.metrics.strong_detection.framewise_detection_metrics import _frames_to_events
        true_events = _frames_to_events(gt_frames, fps)
        targets_events = [[true_events]]  # B=1, C=1

        metric_tensor = StrongDetectionF1Tensor(iou=0.5, fps=fps, prob_threshold=0.5)
        metric_tensor.update(logits, targets_events)
        f1_tensor = metric_tensor.get_metric()["f1"]

        # they should be identical
        assert abs(f1_tensor - f1_file) < 1e-6


def test_strong_f1_tensor_multi_class():
    fps = 20.0
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Two classes: cat and dog
        ref_events = [
            (0.0, 1.0, "cat"),
            (1.5, 2.5, "dog"),
        ]
        pred_events = [
            (0.1, 0.9, "cat"),  # TP
            (3.0, 4.0, "dog"),  # FP
        ]

        ref_sel = tmp_path / "ref2.tsv"
        pred_sel = tmp_path / "pred2.tsv"
        _write_selection_table(ref_sel, ref_events)
        _write_selection_table(pred_sel, pred_events)

        ref_csv = tmp_path / "ref2.csv"
        pred_csv = tmp_path / "pred2.csv"
        pd.DataFrame({
            "fn": ["clip2"],
            "audio_fp": [np.nan],
            "selection_table_fp": [ref_sel.name],
        }).to_csv(ref_csv, index=False)
        pd.DataFrame({
            "fn": ["clip2"],
            "audio_fp": [np.nan],
            "selection_table_fp": [pred_sel.name],
        }).to_csv(pred_csv, index=False)

        label_set = ["cat", "dog"]
        metric_file = StrongDetectionF1Metric(
            iou=0.5,
            label_set=label_set,
            base_path_reference=str(tmp_path),
            base_path_prediction=str(tmp_path),
        )
        # Use default_duration since we don't have audio files
        f1_file = metric_file(ref_csv, pred_csv, default_duration=5.0)

        # Build frame-level tensors
        duration = 5.0
        T = int(duration * fps)
        gt_frames = np.zeros((T, len(label_set)), dtype=int)
        pred_frames = np.zeros_like(gt_frames)

        # GT cat
        gt_frames[int(0 * fps) : int(1 * fps), 0] = 1
        # GT dog
        gt_frames[int(1.5 * fps) : int(2.5 * fps), 1] = 1

        # Pred cat TP
        pred_frames[int(0.1 * fps) : int(0.9 * fps), 0] = 1
        # Pred dog FP
        pred_frames[int(3.0 * fps) : int(4.0 * fps), 1] = 1

        logits = torch.from_numpy(pred_frames[None]).float() * 10  # (1, T, C)

        # Build event annotations list [[[events_class0]]]
        from representation_learning.metrics.strong_detection.framewise_detection_metrics import _frames_to_events
        true_events_cat = _frames_to_events(gt_frames[:, 0], fps)
        true_events_dog = _frames_to_events(gt_frames[:, 1], fps)
        targets_events = [[true_events_cat, true_events_dog]]

        metric_tensor = StrongDetectionF1Tensor(
            iou=0.5, fps=fps, prob_threshold=0.5, label_set=label_set
        )
        metric_tensor.update(logits, targets_events)
        f1_tensor = metric_tensor.get_metric()["f1"]

        assert abs(f1_tensor - f1_file) < 1e-6