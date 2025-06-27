def voxaboxen_metrics(ref_events, pred_events, iou_thr=(0.5, 0.8)):
    """Return F1 and mAP at the requested IoU thresholds."""
    import numpy as np
    import sed_eval
    from sed_eval.util import event_list as el
    from sklearn.metrics import average_precision_score

    ref_list, pred_list = el.EventList(ref_events), el.EventList(pred_events)
    labels = sed_eval.util.event_list.unique_event_labels([ref_list, pred_list])

    # ---------- F1 ----------
    ev = sed_eval.sound_event.EventBasedMetrics(
        labels, t_collar=0.0, percentage_of_length=0.0, event_matching_type="optimal"
    )
    for f in el.unique_files(ref_list):
        ev.evaluate(
            el.filter_event_list(ref_list, f), el.filter_event_list(pred_list, f)
        )
    f1 = ev.results_overall_metrics()["f_measure"]["f_score"]

    # ---------- mAP ----------
    out = {"f1": f1}
    for thr in iou_thr:
        # assumes you already stored IoU and score for each pred
        y_true = np.array([p["iou"] >= thr for p in pred_events], dtype=int)
        y_score = np.array([p["confidence"] for p in pred_events])
        out[f"mAP@{thr}"] = average_precision_score(y_true, y_score)
    return out
