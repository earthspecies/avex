"""
Functions for evaluating model performance
"""

import argparse
import csv
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import tqdm
from scipy.signal import find_peaks, medfilt
from voxaboxen.evaluation.conf_mats import (
    get_confusion_matrix,
    plot_confusion_matrix,
    summarize_confusion_matrix,
)
from voxaboxen.evaluation.nms import nms, soft_nms
from voxaboxen.evaluation.raven_utils import Clip
from voxaboxen.model.model import rms_and_mixup


def f1_from_counts(tp: int, fp: int, fn: int) -> Dict:
    """
    Calculate precision, recall and F1 score from true positives,
    false positives and false negatives.

    Parameters
    ----------
    tp : int
        Number of true positives
    fp : int
        Number of false positives
    fn : int
        Number of false negatives

    Returns
    -------
    dict
        Dictionary containing precision, recall and F1 score
    """

    prec = 1 if tp == 0 else tp / (tp + fp)
    rec = 0 if tp == 0 else tp / (tp + fn)
    f1 = 0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return {"prec": prec, "rec": rec, "f1": f1}


def macro_micro_f1_metrics(
    summary: Dict, unknown_label: str = "Unknown"
) -> Tuple[Dict, Dict]:
    """
    Calculate macro and micro averaged  from per-class summary statistics.

    Parameters
    ----------
    summary : dict
        Dictionary of the sort output by summarize_metrics(), containing per-class
        metrics and counts
    unknown_label : str, optional
        Label to exclude from macro averaging, by default "Unknown"

    Returns
    -------
    tuple
        Tuple containing (macro_metrics, micro_metrics) dictionaries
    """

    metrics = ["f1", "precision", "recall", "f1_seg", "precision_seg", "recall_seg"]
    macro = {}

    for metric in metrics:
        e = []
        for labelname in summary:
            if labelname == unknown_label:
                continue
            m = summary[labelname][metric]
            e.append(m)
        macro[metric] = float(np.mean(e))

    tp = sum(v["TP"] for v in summary.values())
    fp = sum(v["FP"] for v in summary.values())
    fn = sum(v["FN"] for v in summary.values())
    micro = f1_from_counts(tp, fp, fn)
    tp = sum(v["TP_seg"] for v in summary.values())
    fp = sum(v["FP_seg"] for v in summary.values())
    fn = sum(v["FN_seg"] for v in summary.values())
    seg_micro = f1_from_counts(tp, fp, fn)
    seg_micro = {f"{k}_seg": v for k, v in seg_micro.items()}
    micro.update(seg_micro)

    return macro, micro


def get_metrics(
    predictions_fp: str,
    annotations_fp: str,
    iou: float,
    class_threshold: float,
    duration: float,
    label_mapping: Dict[str, str],
    label_set: list,
    unknown_label: str,
) -> Dict[str, Dict[str, int]]:
    """
    Compare the predicted boxes in `prediction_fp` to the true boxes as given in
    annotations_fp, and use graph matching to compute TP, FP and FN counts by
    class.

    Parameters
    ----------
    predictions_fp : str
        Filepath to predictions file
    annotations_fp : str
        Filepath to annotations file
    iou : float
        Intersection-over-Union threshold for matching
    class_threshold : float
        Confidence threshold for class predictions
    duration : float
        Duration of audio clip in seconds
    label_mapping : dict
        Mapping between annotation labels and model classes
    label_set: list
        List of labels used
    unknown_label : str
        Label to use for unknown classes

    Returns
    -------
    dict
        Nested dictionary, outer-keys are class names, inner-keys are TP, FP,
        FN, TP_seg, FP_seg, FN_seg, the latter three referring to the counts
        from segmentation-based matching.
    """

    c = Clip(label_set=label_set, unknown_label=unknown_label)
    c.duration = duration
    c.load_predictions(predictions_fp)
    c.threshold_class_predictions(class_threshold)
    assert not any(c.predictions["Annotation"] == "Unknown")
    c.load_annotations(annotations_fp, label_mapping=label_mapping)

    c.compute_matching(IoU_minimum=iou)
    tfpn_counts_by_class = c.evaluate()

    return tfpn_counts_by_class


def summarize_metrics(tpfn_counts: Dict) -> Dict:
    """
    Aggregate true/false positive/negative counts across files and compute
    precision, recall, and F1 scores.

    Parameters
    ----------
    tpfn_counts : dict
        Dictionary containing per-file metrics counts with structure:
        {
            filepath1: {
                class_label1: {
                    'TP': int,      # True positives for detection
                    'FP': int,      # False positives for detection
                    'FN': int,      # False negatives for detection
                    'TP_seg': int,  # True positives by segmentation-base evaluation
                    'FP_seg': int,  # False positives by segmentation-base evaluation
                    'FN_seg': int   # False negatives by segmentation-base evaluation
                },
                class_label2: {...},
                ...
            },
            filepath2: {...},
            ...
        }

    Returns
    -------
    dict
        Dictionary containing aggregated metrics per class with structure:
        {
            class_label1: {
                'TP': int,             # Sum of true positives
                'FP': int,             # Sum of false positives
                'FN': int,             # Sum of false negatives
                'TP_seg': int,         # Sum of segmentation-based true positives
                'FP_seg': int,         # Sum of segmentation-based false positives
                'FN_seg': int,         # Sum of segmentation-based false negatives
                'precision': float,     # Detection precision (TP/(TP+FP))
                'precision_seg': float, # Precision
                'recall': float,       # Detection recall (TP/(TP+FN))
                'recall_seg': float,    # Segmentation-based recall
                'f1': float,           # Detection F1 score
                'f1_seg': float        # Segmentation-based F1 score
            },
            class_label2: {...},
            ...
        }
    """

    fps = sorted(tpfn_counts.keys())
    class_labels = sorted(tpfn_counts[fps[0]].keys())

    overall = {
        labelname: {"TP": 0, "FP": 0, "FN": 0, "TP_seg": 0, "FP_seg": 0, "FN_seg": 0}
        for labelname in class_labels
    }

    for fp in fps:
        for labelname in class_labels:
            counts = tpfn_counts[fp][labelname]
            overall[labelname]["TP"] += counts["TP"]
            overall[labelname]["FP"] += counts["FP"]
            overall[labelname]["FN"] += counts["FN"]
            overall[labelname]["TP_seg"] += counts["TP_seg"]
            overall[labelname]["FP_seg"] += counts["FP_seg"]
            overall[labelname]["FN_seg"] += counts["FN_seg"]

    for labelname in class_labels:
        tp = overall[labelname]["TP"]
        fp = overall[labelname]["FP"]
        fn = overall[labelname]["FN"]
        tp_seg = overall[labelname]["TP_seg"]
        fp_seg = overall[labelname]["FP_seg"]
        fn_seg = overall[labelname]["FN_seg"]

        if tp + fp == 0:
            prec = 1
        else:
            prec = tp / (tp + fp)
        overall[labelname]["precision"] = prec

        if tp_seg + fp_seg == 0:
            prec_seg = 1
        else:
            prec_seg = tp_seg / (tp_seg + fp_seg)
        overall[labelname]["precision_seg"] = prec_seg

        if tp + fn == 0:
            rec = 1
        else:
            rec = tp / (tp + fn)
        overall[labelname]["recall"] = rec

        if tp_seg + fn_seg == 0:
            rec_seg = 1
        else:
            rec_seg = tp_seg / (tp_seg + fn_seg)
        overall[labelname]["recall_seg"] = rec_seg

        if prec + rec == 0:
            f1 = 0
        else:
            f1 = 2 * prec * rec / (prec + rec)
        overall[labelname]["f1"] = f1

        if prec_seg + rec_seg == 0:
            f1_seg = 0
        else:
            f1_seg = 2 * prec_seg * rec_seg / (prec_seg + rec_seg)
        overall[labelname]["f1_seg"] = f1_seg

    return overall


def generate_predictions(
    model: torch.nn.Module,
    single_clip_dataloader: torch.utils.data.DataLoader,
    args: argparse.Namespace,
    verbose: bool = True,
) -> Tuple[
    np.ndarray,  # assembled_dets
    np.ndarray,  # assembled_regs
    np.ndarray,  # assembled_classifs
    Optional[np.ndarray],  # assembled_rev_dets
    Optional[np.ndarray],  # assembled_rev_regs
    Optional[np.ndarray],  # assembled_rev_classifs
]:
    """
    Generate predictions for a single audio clip using the model.

    Parameters
    ----------
    model : torch.nn.Module
        Model to use for prediction
    single_clip_dataloader : torch.utils.data.DataLoader
        DataLoader for a single audio clip
    args : argparse.Namespace
        Configuration arguments
    verbose : bool, optional
        Whether to show progress bar, by default True

    Returns
    -------
    tuple
        Tuple containing forward and backward predictions
        (detections, regressions, classifications)
    """

    assert single_clip_dataloader.dataset.clip_hop == args.clip_duration / 2, (
        "For inference, clip hop is assumed to be equal to half clip duration"
    )

    device = next(iter(model.parameters())).device
    model.eval()

    all_detections = []
    all_regressions = []
    all_classifs = []
    all_rev_detections = []
    all_rev_regressions = []
    all_rev_classifs = []

    if verbose:
        iterator = tqdm.tqdm(
            enumerate(single_clip_dataloader), total=len(single_clip_dataloader)
        )
    else:
        iterator = enumerate(single_clip_dataloader)

    with torch.no_grad():
        for i, X in iterator:
            X = X.to(device=device, dtype=torch.float)
            X, _, _, _ = rms_and_mixup(X, None, None, None, False, args)

            model_outputs = model(X)
            assert isinstance(model_outputs, tuple)
            all_detections.append(model_outputs[0])
            all_regressions.append(model_outputs[1])
            if args.segmentation_based:
                classification = torch.nn.functional.sigmoid(model_outputs[2])
            else:
                classification = torch.nn.functional.softmax(model_outputs[2], dim=-1)
            all_classifs.append(classification)
            if model.is_bidirectional:
                assert all(x is not None for x in model_outputs)
                all_rev_detections.append(model_outputs[3])
                all_rev_regressions.append(model_outputs[4])
                all_rev_classifs.append(
                    model_outputs[5].softmax(-1)
                )  # segmentation-based is not used when bidirectional
            else:
                assert all(x is None for x in model_outputs[3:])

                if args.is_test and i == 15:
                    break

        all_detections = torch.cat(all_detections)
        all_regressions = torch.cat(all_regressions)
        all_classifs = torch.cat(all_classifs)
        if model.is_bidirectional:
            all_rev_detections = torch.cat(all_rev_detections)
            all_rev_regressions = torch.cat(all_rev_regressions)
            all_rev_classifs = torch.cat(all_rev_classifs)

        # Todo: Need better checking that preds are the correct dur
        assert all_detections.size(dim=1) % 2 == 0
        first_quarter_window_dur_samples = all_detections.size(dim=1) // 4
        last_quarter_window_dur_samples = (
            all_detections.size(dim=1) // 2
        ) - first_quarter_window_dur_samples

        def assemble(
            d: torch.Tensor, r: torch.Tensor, c: torch.Tensor
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            Assemble predictions from overlapping windows by discarding boundary
            predictions. We use half overlapping windows.
            See get_val_dataloader and get_test_dataloader in data.py

            Parameters
            ----------
            d : torch.Tensor
                Detection scores tensor of shape (windows, window_length)
            r : torch.Tensor
                Regression outputs tensor of shape (windows, window_length)
            c : torch.Tensor
                Classification logits of shape (windows, window_length, num_classes)

            Returns
            -------
            Tuple[np.ndarray, np.ndarray, np.ndarray]
                Tuple containing:
                - assembled detections (1D numpy array)
                - assembled regressions (1D numpy array)
                - assembled classifications (2D numpy array)
            """
            # assemble detections
            beginning_d_bit = d[0, :first_quarter_window_dur_samples]
            end_d_bit = d[-1, -last_quarter_window_dur_samples:]
            d_clipped = d[
                :, first_quarter_window_dur_samples:-last_quarter_window_dur_samples
            ]
            middle_d_bit = torch.reshape(d_clipped, (-1,))
            assembled_d = torch.cat([beginning_d_bit, middle_d_bit, end_d_bit])

            # assemble regressions
            beginning_r_bit = r[0, :first_quarter_window_dur_samples]
            end_r_bit = r[-1, -last_quarter_window_dur_samples:]
            r_clipped = r[
                :, first_quarter_window_dur_samples:-last_quarter_window_dur_samples
            ]
            middle_r_bit = torch.reshape(r_clipped, (-1,))
            assembled_r = torch.cat([beginning_r_bit, middle_r_bit, end_r_bit])

            # assemble classifs
            beginning_c_bit = c[0, :first_quarter_window_dur_samples, :]
            end_c_bit = c[-1, -last_quarter_window_dur_samples:, :]
            c_clipped = c[
                :, first_quarter_window_dur_samples:-last_quarter_window_dur_samples, :
            ]
            middle_c_bit = torch.reshape(c_clipped, (-1, c_clipped.size(-1)))
            assembled_c = torch.cat([beginning_c_bit, middle_c_bit, end_c_bit])
            return (
                assembled_d.detach().cpu().numpy(),
                assembled_r.detach().cpu().numpy(),
                assembled_c.detach().cpu().numpy(),
            )

        assembled_dets, assembled_regs, assembled_classifs = assemble(
            all_detections, all_regressions, all_classifs
        )
        if model.is_bidirectional:
            assembled_rev_dets, assembled_rev_regs, assembled_rev_classifs = assemble(
                all_rev_detections, all_rev_regressions, all_rev_classifs
            )
        else:
            assembled_rev_dets = assembled_rev_regs = assembled_rev_classifs = None

        return (
            assembled_dets,
            assembled_regs,
            assembled_classifs,
            assembled_rev_dets,
            assembled_rev_regs,
            assembled_rev_classifs,
        )


def pred2bbox(
    detection_peaks: Union[np.ndarray, Sequence[float]],
    detection_probs: Union[np.ndarray, Sequence[float]],
    durations: Union[np.ndarray, Sequence[float]],
    class_idxs: Union[np.ndarray, Sequence[int]],
    class_probs: Union[np.ndarray, Sequence[float]],
    pred_sr: int,
    is_rev: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert detection peaks and durations to bounding box format.

    Parameters
    ----------
    detection_peaks : array-like
        Array of detected peak positions
    detection_probs : array-like
        Array of detection probabilities
    durations : array-like
        Array of predicted durations
    class_idxs : array-like
        Array of predicted class indices
    class_probs : array-like
        Array of class probabilities
    pred_sr : int
        Prediction sampling rate in Hz
    is_rev : bool
        Whether predictions are from backward pass

    Returns
    -------
    tuple
        Tuple containing (bboxes, detection_probs, class_idxs, class_probs)
    """

    detection_peaks = detection_peaks / pred_sr
    bboxes = []
    detection_probs_sub = []
    class_idxs_sub = []
    class_probs_sub = []

    for i in range(len(detection_peaks)):
        duration = durations[i]
        if duration <= 0:
            continue

        if is_rev:
            end = detection_peaks[i]
            bbox = [end - duration, end]
        else:
            start = detection_peaks[i]
            bbox = [start, start + duration]

        bboxes.append(bbox)

        detection_probs_sub.append(detection_probs[i])
        class_idxs_sub.append(class_idxs[i])
        class_probs_sub.append(class_probs[i])

    return (
        np.array(bboxes),
        np.array(detection_probs_sub),
        np.array(class_idxs_sub),
        np.array(class_probs_sub),
    )


def bbox2raven(
    bboxes: Union[np.ndarray, Sequence[Sequence[float]], None],
    class_idxs: Union[np.ndarray, Sequence[int]],
    label_set: List[str],
    detection_probs: Union[np.ndarray, Sequence[float]],
    class_probs: Union[np.ndarray, Sequence[float]],
    unknown_label: str,
) -> List[List[Union[str, float]]]:
    """
    Convert bounding boxes to Raven selection table format.

    Parameters
    ----------
    bboxes : array-like
        Array of bounding boxes [start, end]
    class_idxs : array-like
        Array of class indices
    label_set : list
        List of class labels
    detection_probs : array-like
        Array of detection probabilities
    class_probs : array-like
        Array of class probabilities
    unknown_label : str
        Label to use for unknown classes

    Returns
    -------
    list
        List of rows for Raven selection table
    """

    if bboxes is None:
        return [
            [
                "Begin Time (s)",
                "End Time (s)",
                "Annotation",
                "Detection Prob",
                "Class Prob",
            ]
        ]

    columns = [
        "Begin Time (s)",
        "End Time (s)",
        "Annotation",
        "Detection Prob",
        "Class Prob",
    ]

    def label_idx_to_label(i: int) -> str:
        """
        Convert label index (integer) to label (string)

        Returns
        -----
        label : str
        """
        if i == -1:
            return unknown_label
        else:
            return label_set[i]

    out_data = [
        [bbox[0], bbox[1], label_idx_to_label(int(c)), dp, cp]
        for bbox, c, dp, cp in zip(
            bboxes, class_idxs, detection_probs, class_probs, strict=False
        )
    ]
    out_data = sorted(out_data, key=lambda x: x[:2])

    out = [columns] + out_data

    return out


def fill_holes(m: Union[np.ndarray, Sequence[bool]], min_hole: int) -> np.ndarray:
    """
    Fill small gaps in binary mask.

    Parameters
    ----------
    m : array-like
        Binary mask
    min_hole : int
        Minimum length of a sequence of False's that won't get filled in, all
        sequences of length less than this will get flipped to True's

    Returns
    -------
    numpy.ndarray
        Filled binary mask
    """

    stops = m[:-1] * ~m[1:]
    stops = np.where(stops)[0]

    for stop in stops:
        look_forward = m[stop + 1 : stop + 1 + min_hole]
        if np.any(look_forward):
            next_start = np.amin(np.where(look_forward)[0]) + stop + 1
            m[stop:next_start] = True

    return m


def delete_short(m: Union[np.ndarray, Sequence[bool]], min_pos: int) -> np.ndarray:
    """
    Delete short sequences of True's in binary mask.

    Parameters
    ----------
    m : array-like
        Binary mask
    min_pos : int
        Minimum length of a sequence of True's that won't get deleted, all
        sequences of length less than this will get flipped to False's

    Returns
    -------
    numpy.ndarray
        Filled binary mask
    """

    starts = m[1:] * ~m[:-1]

    starts = np.where(starts)[0] + 1

    clips = []

    for start in starts:
        look_forward = m[start:]
        ends = np.where(~look_forward)[0]
        if len(ends) > 0:
            clips.append((start, start + np.amin(ends)))
        else:
            clips.append((start, len(m) - 1))

    m = np.zeros_like(m).astype(bool)
    for clip in clips:
        if clip[1] - clip[0] >= min_pos:
            m[clip[0] : clip[1]] = True

    return m


def write_tsv(out_fp: str, data: List) -> None:
    """
    Write data to TSV file.

    Parameters
    ----------
    out_fp : str
        Output file path
    data : list
        Iterable of lists, which will be written as rows to `out_fp`
    """

    with open(out_fp, "w", newline="") as ff:
        tsv_output = csv.writer(ff, delimiter="\t")

        for row in data:
            tsv_output.writerow(row)


def select_from_neg_idxs(
    df: pd.DataFrame, neg_idxs: Union[Sequence[int], pd.Index]
) -> pd.DataFrame:
    """
    Select rows not in given indices from DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame
    neg_idxs : array-like
        Indices to exclude

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame
    """
    masked = df.loc[~df.index.isin(neg_idxs)]
    return masked


def combine_fwd_bck_preds(
    target_dir: str,
    fn: str,
    comb_discard_threshold: float,
    comb_iou_thresh: float,
    det_thresh: float,
) -> Tuple[str, str]:
    """
    Combine forward and backward predictions using graph matching.

    Parameters
    ----------
    target_dir : str
        Directory containing prediction files
    fn : str
        Base filename
    comb_discard_threshold : float
        Probability threshold for discarding predictions
    comb_iou_thresh : float
        IoU threshold for matching predictions, a pair of fwd and bck prediction
        that have IoU above this threshold are connected in the graph and the
        algorithm can choose to match them
    det_thresh : float
        Detection threshold

    Returns
    -------
    tuple
        Tuple containing filepaths to combined and matched predictions

    Notes
    -----
    This function returns both the combined predictions and the matched pred-
    ictions. The latter are the outputs from the graph matching. The former
    include the latter and also whatever forward and backward predictions were
    not a part of the matching but which have high probability.

    Graph matching is performed using the Hopcroft-Karp-Karzanov algorithm for
    maximum graph matching, which computes the matching with the maximum number
    of edges.
    """

    fwd_preds_fp = os.path.join(
        target_dir, f"peaks_pred_{fn}-detthresh{det_thresh}-fwd.txt"
    )
    bck_preds_fp = os.path.join(
        target_dir, f"peaks_pred_{fn}-detthresh{det_thresh}-bck.txt"
    )
    comb_preds_fp = os.path.join(
        target_dir, f"peaks_pred_{fn}-detthresh{det_thresh}-comb.txt"
    )
    match_preds_fp = os.path.join(
        target_dir, f"peaks_pred_{fn}-detthresh{det_thresh}-match.txt"
    )
    fwd_preds = pd.read_csv(fwd_preds_fp, sep="\t")
    bck_preds = pd.read_csv(bck_preds_fp, sep="\t")

    c = Clip()
    c.load_annotations(fwd_preds_fp)
    c.load_predictions(bck_preds_fp)
    os.makedirs(f"{target_dir}/tmp-cache", exist_ok=True)
    if (
        os.path.exists(
            match_cache_fp := (
                f"{target_dir}/tmp-cache/match_cache_{fn}"
                f"-detthresh{det_thresh}"
                f"-cit{comb_iou_thresh}.npy"
            )
        )
        and False
    ):
        matching = np.load(match_cache_fp)
    else:
        c.compute_matching(IoU_minimum=comb_iou_thresh)
        matching = np.array(c.matching)
        np.save(match_cache_fp, matching)

    if matching.shape == (0,):
        match_preds = pd.DataFrame([], columns=fwd_preds.columns)
    else:
        fwd_matches = fwd_preds.iloc[matching[:, 0]]
        bck_matches = bck_preds.iloc[matching[:, 1]]

        fbn = fwd_matches["Begin Time (s)"].to_numpy()
        fen = fwd_matches["End Time (s)"].to_numpy()
        bbn = bck_matches["Begin Time (s)"].to_numpy()
        ben = bck_matches["End Time (s)"].to_numpy()
        starts = (fbn + bbn) / 2
        ends = (fen + ben) / 2
        ious = ((fen + ben) - (fbn + bbn)) / (
            2 * (np.maximum(fen, ben) - np.minimum(fbn, bbn))
        )
        probs = (
            1
            - (1 - fwd_matches["Detection Prob"].to_numpy())
            * (1 - bck_matches["Detection Prob"].to_numpy())
        ) * ious
        match_preds = pd.DataFrame(
            {
                "Begin Time (s)": starts,
                "End Time (s)": ends,
                "Detection Prob": probs,
                "Annotation": fwd_matches["Annotation"],
                "Class Prob": 1.0,
            }
        )

    # Include the union of all predictions that weren't part of the matching
    fwd_matched_idxs = [m[0] for m in matching]
    bck_matched_idxs = [m[1] for m in matching]
    fwd_unmatched = select_from_neg_idxs(fwd_preds, fwd_matched_idxs)
    bck_unmatched = select_from_neg_idxs(bck_preds, bck_matched_idxs)
    to_concat = [
        x for x in [match_preds, fwd_unmatched, bck_unmatched] if x.shape[0] > 0
    ]
    comb_preds = pd.concat(to_concat) if len(to_concat) > 0 else fwd_preds
    assert len(comb_preds) == len(fwd_preds) + len(bck_preds) - len(matching)

    # Finally, keep only predictions above a threshold,
    # this will include almost all matches
    comb_preds = comb_preds.loc[comb_preds["Detection Prob"] > comb_discard_threshold]
    comb_preds.sort_values("Begin Time (s)")
    comb_preds.index = list(range(len(comb_preds)))

    comb_preds.to_csv(comb_preds_fp, sep="\t", index=False)
    match_preds.to_csv(match_preds_fp, sep="\t", index=False)
    return comb_preds_fp, match_preds_fp


def export_to_selection_table(
    detections: Union[np.ndarray, Sequence[float]],
    regressions: Union[np.ndarray, Sequence[float]],
    classifications: Union[np.ndarray, Sequence[Sequence[float]]],
    fn: str,
    args: argparse.Namespace,
    is_bck: bool,
    verbose: bool = True,
    target_dir: Optional[str] = None,
    detection_threshold: float = 0.5,
    classification_threshold: float = 0.0,
) -> str:
    """
    Convert a set of model outputs (detections, regressions, classifications)
    into a set of predicted bounding boxes, and write as a Raven-style selection
    table.

    Parameters
    ----------
    detections : array-like
        Detection scores (as probabilities)
    regressions : array-like
        Duration predictions
    classifications : array-like
        Classification scores (as logits)
    fn : str
        Output filename
    args : argparse.Namespace
        Configuration arguments
    is_bck : bool
        Whether predictions are from the backward version of the model
    verbose : bool, optional
        Whether to print progress, by default True
    target_dir : str, optional
        Output directory, by default None, in which case it is taken from `args`
    detection_threshold : float, optional
        Detection threshold, by default 0.5
    classification_threshold : float, optional
        Classification threshold, by default 0

    Returns
    -------
    str
        Path to saved selection table
    """

    if hasattr(args, "bidirectional") and args.bidirectional:
        if is_bck:
            fn += "-bck"
        else:
            fn += "-fwd"

    if target_dir is None:
        target_dir = args.experiment_output_dir

    if hasattr(args, "segmentation_based") and args.segmentation_based:
        pred_sr = args.sr // args.scale_factor
        bboxes = []
        det_probs = []
        class_idxs = []
        class_probs = []
        for c in range(np.shape(classifications)[1]):
            classifications_sub = classifications[:, c]

            if hasattr(args, "median_filter_width") and args.median_filter_width > 1:
                classifications_sub = medfilt(
                    classifications_sub, args.median_filter_width
                )

            classifications_sub_binary = classifications_sub >= detection_threshold
            classifications_sub_binary = fill_holes(
                classifications_sub_binary, int(args.fill_holes_dur_sec * pred_sr)
            )
            classifications_sub_binary = delete_short(
                classifications_sub_binary, int(args.delete_short_dur_sec * pred_sr)
            )

            starts = classifications_sub_binary[1:] * ~classifications_sub_binary[:-1]
            starts = np.where(starts)[0] + 1
            if classifications_sub_binary[0]:
                starts = np.append(starts, 0)

            for start in starts:
                look_forward = classifications_sub_binary[start:]
                ends = np.where(~look_forward)[0]
                if len(ends) > 0:
                    end = start + np.amin(ends)
                else:
                    end = len(classifications_sub_binary) - 1

                bbox = [start / pred_sr, end / pred_sr]
                bboxes.append(bbox)
                det_probs.append(classifications_sub[start:end].mean())
                class_idxs.append(c)
                class_probs.append(classifications_sub[start:end].mean())

        bboxes = np.array(bboxes)
        det_probs = np.array(det_probs)
        class_idxs = np.array(class_idxs)
        class_probs = np.array(class_probs)

    else:
        # peaks
        detection_peaks, properties = find_peaks(
            detections, height=detection_threshold, distance=args.peak_distance
        )
        det_probs = properties["peak_heights"]

        # regressions and classifications
        durations = []
        class_idxs = []
        class_probs = []

        for i in detection_peaks:
            dur = regressions[i]
            durations.append(dur)

            c = np.argmax(classifications[i, :])
            p = classifications[i, c]

            if p < classification_threshold:
                c = -1

            class_idxs.append(c)
            class_probs.append(p)

        durations = np.array(durations)
        class_idxs = np.array(class_idxs)
        class_probs = np.array(class_probs)

        pred_sr = args.sr // args.scale_factor

        bboxes, det_probs, class_idxs, class_probs = pred2bbox(
            detection_peaks,
            det_probs,
            durations,
            class_idxs,
            class_probs,
            pred_sr,
            is_bck,
        )

    if args.nms == "soft_nms":
        bboxes, det_probs, class_idxs, class_probs = soft_nms(
            bboxes,
            det_probs,
            class_idxs,
            class_probs,
            sigma=args.soft_nms_sigma,
            thresh=detection_threshold,
        )
    elif args.nms == "nms":
        bboxes, det_probs, class_idxs, class_probs = nms(
            bboxes, det_probs, class_idxs, class_probs, iou_thresh=args.nms_thresh
        )

    if verbose:
        print(f"Found {len(det_probs)} boxes")

    target_fp = os.path.join(target_dir, f"peaks_pred_{fn}.txt")

    st = bbox2raven(
        bboxes, class_idxs, args.label_set, det_probs, class_probs, args.unknown_label
    )
    write_tsv(target_fp, st)

    return target_fp


def predict_and_generate_manifest(
    model: torch.nn.Module,
    dataloader_dict: Dict[str, torch.utils.data.DataLoader],
    args: argparse.Namespace,
    detection_thresholds: Optional[List[float]] = None,
    verbose: bool = True,
) -> Dict[float, pd.DataFrame]:
    """
    Generate predictions for multiple files and create manifest of results.

    For each threshold in `detection_thresholds`, for each filename in the keys
    of `dataloader_dict`, create a set of predictions and save as a selection
    table. For each threshold, create a manifest specifying the forward and back-
    ward predictions for each file, and the files with the corresponding
    ground truth boxes. Return a dictionary of manifests, one for each
    threshold.

    Parameters
    ----------
    model : torch.nn.Module
        Model to use for prediction
    dataloader_dict : dict
        Dictionary mapping filenames to DataLoaders
    args : argparse.Namespace
        Configuration arguments
    detection_thresholds : list, optional
        List of detection thresholds to evaluate, by default None, in which case
        a single threshold of args.detection_threshold is used
    verbose : bool, optional
        Whether to show progress, by default True

    Returns
    -------
    dict
        Dictionary mapping thresholds to prediction manifests
    """

    if detection_thresholds is None:
        detection_thresholds = [args.detection_threshold]
    fns = []
    annotations_fps = []
    fwd_predictions_fps = []
    bck_predictions_fps = []
    durations = []
    for i, (fn_base, dloader) in enumerate(dataloader_dict.items()):
        if args.is_test and i == 3:
            break
        annotations_fp = dataloader_dict[fn_base].dataset.annot_fp
        fns.append(fn_base)
        annotations_fps.append(annotations_fp)
        (
            fwd_detections,
            fwd_regressions,
            fwd_classifications,
            bck_detections,
            bck_regressions,
            bck_classifications,
        ) = generate_predictions(model, dloader, args, verbose=verbose)
        durations.append(np.shape(fwd_detections)[0] * args.scale_factor / args.sr)

        fwd_predictions_fps_by_thresh = {}
        bck_predictions_fps_by_thresh = {}
        for det_thresh in detection_thresholds:
            fn = f"{fn_base}-detthresh{det_thresh}"
            fwd_predictions_fp = export_to_selection_table(
                fwd_detections,
                fwd_regressions,
                fwd_classifications,
                fn,
                args,
                is_bck=False,
                verbose=verbose,
                detection_threshold=det_thresh,
            )
            if model.is_bidirectional:
                assert all(
                    x is not None
                    for x in [bck_detections, bck_classifications, bck_regressions]
                )
                bck_predictions_fp = export_to_selection_table(
                    bck_detections,
                    bck_regressions,
                    bck_classifications,
                    fn,
                    args,
                    is_bck=True,
                    verbose=verbose,
                    detection_threshold=args.detection_threshold,
                )
            else:
                assert all(
                    x is None
                    for x in [bck_detections, bck_classifications, bck_regressions]
                )
                bck_predictions_fp = None

            fwd_predictions_fps_by_thresh[det_thresh] = fwd_predictions_fp
            bck_predictions_fps_by_thresh[det_thresh] = bck_predictions_fp

        fwd_predictions_fps.append(fwd_predictions_fps_by_thresh)
        bck_predictions_fps.append(bck_predictions_fps_by_thresh)

    manifests_by_thresh = {}
    for dt in detection_thresholds:
        fpfps = [x[dt] for x in fwd_predictions_fps]
        bpfps = [x[dt] for x in bck_predictions_fps]
        manifest = pd.DataFrame(
            {
                "filename": fns,
                "fwd_predictions_fp": fpfps,
                "bck_predictions_fp": bpfps,
                "annotations_fp": annotations_fps,
                "duration_sec": durations,
            }
        )
        manifests_by_thresh[dt] = manifest
    return manifests_by_thresh


def evaluate_based_on_manifest(
    manifest: pd.DataFrame,
    output_dir: str,
    iou: float,
    class_threshold: float,
    label_mapping: Dict[str, str],
    label_set: List,
    unknown_label: str,
    det_thresh: float,
    comb_discard_threshold: float = 0,
    comb_iou_thresh: float = 0.5,
    bidirectional: bool = False,
    pred_types: Optional[Union[Tuple[str, ...], Sequence[str]]] = None,
    make_confusion_matrix: bool = False,
    split: str = "",
) -> Tuple[Dict[str, Dict], None]:
    """
    Evaluate predictions using manifest file.

    Parameters
    ----------
    manifest : pandas.DataFrame
        Manifest containing prediction and annotation paths
    output_dir : str
        Directory to save results
    iou : float
        Intersection-over-Union threshold for matching during evaluation
    class_threshold : float
        Confidence threshold for class predictions
    label_mapping : dict
        Mapping between annotation labels and model classes
    label_set : List
        List of labels
    unknown_label : str
        Label to use for unknown classes
    det_thresh : float
        Detection threshold, detections with probability below `det_thresh` will
        be discarded
    comb_discard_threshold : float, optional
        Same as `det_thresh` but applied after combining forward and backward
        predictions, by default 0, only used if `bidirectional`=True
    comb_iou_thresh : float, optional
        IoU threshold for combining predictions, by default 0.5, only used if
        `bidirectional`=True
    bidirectional : bool, optional
        Whether model is bidirectional, by default False
    pred_types : tuple, optional
        Types of predictions to evaluate, by default None
    make_confusion_matrix: bool
        Whether to save confusion matrix
    split : str
        Name of split (ie test or val)

    Returns
    -------
    metrics : dict
        Nested dictionary containing scores for each metric for each pred type
    confusion_matrix: dict
        confusion matrices for different pred types and fp's
    """

    if pred_types is None:
        pred_types = ("fwd", "bck", "comb", "match") if bidirectional else ("fwd",)
    metrics = {p: {} for p in pred_types}

    if make_confusion_matrix:
        conf_mats = {p: {} for p in pred_types}
        conf_mat_labels = {}
    else:
        conf_mats = None
        conf_mat_labels = None

    for _i, row in manifest.iterrows():
        fn = row["filename"]
        annots_fp = row["annotations_fp"]
        duration = row["duration_sec"]
        if bidirectional:
            row["comb_predictions_fp"], row["match_predictions_fp"] = (
                combine_fwd_bck_preds(
                    output_dir,
                    fn,
                    comb_discard_threshold=comb_discard_threshold,
                    comb_iou_thresh=comb_iou_thresh,
                    det_thresh=det_thresh,
                )
            )

        for pred_type in pred_types:
            preds_fp = row[f"{pred_type}_predictions_fp"]
            metrics[pred_type][fn] = get_metrics(
                preds_fp,
                annots_fp,
                iou,
                class_threshold,
                duration,
                label_mapping,
                label_set,
                unknown_label,
            )
            if make_confusion_matrix:
                conf_mats[pred_type][fn], conf_mat_labels[pred_type] = (
                    get_confusion_matrix(
                        preds_fp,
                        annots_fp,
                        label_set,
                        label_mapping,
                        unknown_label,
                        iou,
                        class_threshold,
                    )
                )

    # summarize and save metrics
    conf_mat_summaries = {}
    for pred_type in pred_types:
        summary = summarize_metrics(metrics[pred_type])
        metrics[pred_type]["summary"] = summary
        macro, micro = macro_micro_f1_metrics(summary)
        metrics[pred_type]["macro"], metrics[pred_type]["micro"] = macro, micro
        if make_confusion_matrix:
            conf_mat_summaries[pred_type], confusion_matrix_labels = (
                summarize_confusion_matrix(
                    conf_mats[pred_type], conf_mat_labels[pred_type]
                )
            )
            plot_confusion_matrix(
                conf_mat_summaries[pred_type].astype(int),
                confusion_matrix_labels,
                output_dir,
                name=f"cm_{split}_iou_{iou}_class_threshold_{class_threshold}_{pred_type}",
            )

    return metrics, conf_mat_summaries


def mean_average_precision(
    manifests_by_thresh: Dict[float, pd.DataFrame],
    label_mapping: Dict[str, str],
    label_set: List,
    exp_dir: str,
    iou: float = 0.5,
    pred_type: str = "fwd",
    unknown_label: str = "Unknown",
    bidirectional: bool = False,
    comb_iou_thresh: float = 0.0,
    is_test: bool = False,
) -> Tuple[float, Dict[str, list], Dict[str, float]]:
    """
    Calculate mean average precision across detection thresholds.

    Parameters
    ----------
    manifests_by_thresh : dict
        Dictionary mapping thresholds to prediction manifests
    label_mapping : dict
        Mapping between annotation labels and model classes
    label_set: list
        List of labels used
    exp_dir : str
        Experiment directory
    iou : float, optional
        Intersection-over-Union threshold for matching during evaluation, by
        default 0.5
    pred_type : {'fwd', 'bck', 'comb', 'match'}, optional
        Type of predictions to evaluate, by default 'fwd'
    unknown_label : str, optional
        Label to use for unknown classes, by default 'Unknown'
    bidirectional : bool, optional
        Whether model is bidirectional, by default False
    comb_iou_thresh : float, optional
        IoU threshold for combining predictions, by default 0
    is_test : bool, optional
        Whether in test mode (reduced evaluation), by default False

    Returns
    -------
    mAP : float
        Mean average precision
    scores_by_class : dict
        Keys are class names, values are lists of all precision scores for that
        class across all thresholds
    ap_by_class:
        Keys are class names, values are floats for the mean precision for that
        class across all thresholds
    """

    # first loop through thresholds to gather all results
    scores_by_class = {c: [] for c in label_set}
    experiment_output_dir = os.path.join(exp_dir, "outputs")
    if bidirectional:
        comb_discard_threshes_to_sweep = [0.5] if is_test else np.linspace(0, 0.99, 30)
        comb_iou_threshes_to_sweep = [0.5] if is_test else np.linspace(0.2, 0.9, 10)
    else:
        comb_discard_threshes_to_sweep = [0]
        comb_iou_threshes_to_sweep = [0]
    for cdt in tqdm.tqdm(comb_discard_threshes_to_sweep):
        for cit in comb_iou_threshes_to_sweep:
            for det_thresh, test_manifest in manifests_by_thresh.items():
                test_metrics, test_conf_mats = evaluate_based_on_manifest(
                    test_manifest,
                    output_dir=experiment_output_dir,
                    iou=iou,
                    det_thresh=det_thresh,
                    class_threshold=0.0,
                    comb_discard_threshold=cdt,
                    comb_iou_thresh=cit,
                    label_mapping=label_mapping,
                    label_set=label_set,
                    unknown_label="Unknown",
                    bidirectional=bidirectional,
                    pred_types=(pred_type,),
                )
                for c, s in test_metrics[pred_type]["summary"].items():
                    scores_by_class[c].append(
                        dict(s, det_thresh=det_thresh, discard_thresh=cdt, ciou=cit)
                    )

    # now loop through classes to calculate APs
    ap_by_class = {}
    map_results = {c: {} for c in label_set}
    for c, sweep_ in scores_by_class.items():
        sweep = pd.DataFrame(sweep_)  # .sort_values('recall')
        # exclude cases where all TNs because they give weird f-scores
        sweep = sweep.loc[sweep["TP"] + sweep["FP"] + sweep["FN"] != 0]
        precs, recs = list(sweep["precision"]), list(sweep["recall"])
        precs = [0.0] + precs + [1.0]
        recs = [1.0] + recs + [0.0]
        prec_by_rec = {}
        for r, p in zip(recs, precs, strict=False):
            if r in prec_by_rec.keys():
                prec_by_rec[r] = max(p, prec_by_rec[r])
            else:
                prec_by_rec[r] = p
        recs, precs = list(prec_by_rec.keys()), list(prec_by_rec.values())

        # n-point AP computation: https://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf
        recs_np = np.array(recs)
        precs_np = np.array(precs)
        auc = 0
        recall_levels = np.linspace(0, 1, 1001)  # np.arange(0,11)/10
        for recall_level in recall_levels:
            p_interp_at_recall_level = np.amax(precs_np[recs_np >= recall_level])
            auc += p_interp_at_recall_level / len(recall_levels)

        ap_by_class[c] = auc
        map_results[c]["sweep"] = sweep_
        map_results[c]["precs"] = precs
        map_results[c]["recs"] = recs

    map_score = []
    for c in ap_by_class:
        if c != unknown_label:
            map_score.append(ap_by_class[c])
    map_score = float(np.array(map_score).mean())

    return map_score, scores_by_class, ap_by_class
