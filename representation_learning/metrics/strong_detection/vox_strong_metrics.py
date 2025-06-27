import os
from pathlib import Path

import numpy as np
import pandas as pd

from representation_learning.metrics.strong_detection.vox_raven_helpers import Clip


class StrongDetectionF1Metric:
    def __init__(
        self,
        iou: float = 0.5,
        class_threshold: float = 0.5,
        label_set: list = None,
        unknown_label: str = None,
        view: str = None,
        label_mapping: dict = None,
        base_path_prediction="",
        base_path_reference="",
        merge_labels: bool = False,
    ):
        """
        Parameters:
            iou: Minimum Intersection over Union threshold for matching.
            class_threshold: Threshold below which predicted class probabilities are set to unknown.
            label_set: Optional list of labels. If provided, per-label metrics are computed.
            unknown_label: Label to assign for predictions below threshold.
            view: (Optional) Filter for a specific view in the selection table.
            label_mapping: (Optional) Dictionary for mapping labels.
        """
        super().__init__()
        self.iou = iou
        self.class_threshold = class_threshold
        self.label_set = label_set
        self.unknown_label = unknown_label
        self.view = view
        self.label_mapping = label_mapping
        self.base_path_reference = base_path_reference
        self.base_path_prediction = base_path_prediction
        self.merge_labels = merge_labels

    def __call__(
        self,
        reference_fp: Path,
        prediction_fp: Path,
        thresholding: bool = False,
        default_duration: float = 20,
    ) -> float:
        """
        Instead of directly reading a selection table, reference_fp and prediction_fp point
        to CSV files whose rows are of the format:
          fn,audio_fp,selection_table_fp
        We merge the reference and prediction CSVs on the 'fn' column and then, for each clip,
        load the audio and corresponding selection table files. The F1 score is computed for each
        clip and then averaged.
        """
        # Load CSV files.
        ref_df = pd.read_csv(reference_fp)
        pred_df = pd.read_csv(prediction_fp)

        # Merge on 'fn'; assume that the two CSV files share this column.
        merged = pd.merge(ref_df, pred_df, on="fn", suffixes=("_ref", "_pred"))
        scores = []

        for _, row in merged.iterrows():
            clip = Clip(
                label_set=self.label_set,
                unknown_label=self.unknown_label,
                merge_labels=self.merge_labels,
            )

            audio_file = row.get("audio_fp_ref", None)
            if audio_file is None or not isinstance(audio_file, str):
                clip.duration = default_duration
            else:
                clip.load_audio(os.path.join(self.base_path_reference, audio_file))

            # Load reference annotations and prediction annotations.
            selection_ref = row.get("selection_table_fp_ref")
            selection_pred = row.get("selection_table_fp_pred")
            if selection_ref is None or selection_pred is None:
                print(
                    f"Skipping clip {row.get('fn')} because one of the selection table paths is missing."
                )
                continue

            clip.load_annotations(
                os.path.join(self.base_path_reference, str(selection_ref)),
                view=self.view,
                label_mapping=self.label_mapping,
            )
            clip.load_predictions(
                os.path.join(self.base_path_prediction, str(selection_pred)),
                view=self.view,
                label_mapping=self.label_mapping,
            )

            # Apply thresholding if needed.
            if thresholding:
                clip.threshold_class_predictions(self.class_threshold)
            # Compute matching.
            clip.compute_matching(IoU_minimum=self.iou)
            # Evaluate the metrics.
            metrics = clip.evaluate()

            # Compute F1 score.
            if self.label_set is None:
                overall = metrics["all"]
                TP, FP, FN = overall["TP"], overall["FP"], overall["FN"]
                f1 = (2 * TP / (2 * TP + FP + FN)) if (2 * TP + FP + FN) > 0 else 0.0
            else:
                f1_scores = []
                for label, vals in metrics.items():
                    if label == "all":
                        continue
                    TP, FP, FN = vals["TP"], vals["FP"], vals["FN"]
                    if (2 * TP + FP + FN) > 0:
                        f1_scores.append(2 * TP / (2 * TP + FP + FN))
                f1 = np.mean(f1_scores) if f1_scores else 0.0

            scores.append(f1)

        return np.mean(scores) if scores else 0.0
