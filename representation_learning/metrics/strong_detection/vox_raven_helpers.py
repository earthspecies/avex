# Assuming match_events is available from your module:
import pandas as pd
import librosa
import numpy as np

from representation_learning.metrics.strong_detection.detection_metric_helpers import match_events

import pandas as pd

class Clip:
    def __init__(self, label_set=None, unknown_label=None, merge_labels: bool = False):
        self.sr = None
        self.samples = None
        self.duration = None
        self.annotations = None
        self.predictions = None
        self.matching = None
        self.matched_annotations = None
        self.matched_predictions = None
        self.label_set = label_set
        self.unknown_label = unknown_label
        self.merge_labels = merge_labels

    def load_selection_table(self, fp, view=None, label_mapping=None):
        annotations = pd.read_csv(fp, delimiter='\t')
        if view is None and 'View' in annotations.columns:
            views = annotations['View'].unique()
            if len(views) > 1:
                print(
                    f"Multiple views found in {fp}. Consider passing a 'view' parameter to avoid double counting."
                )
        if view is not None:
            annotations = annotations[annotations['View'].str.contains(view)].reset_index(drop=True)
        if label_mapping is not None:
            annotations['Annotation'] = annotations['Annotation'].map(label_mapping)
            annotations = annotations[~pd.isnull(annotations['Annotation'])]
        if self.merge_labels:
            annotations['Annotation'] = "BIO"
        return annotations


    def load_audio(self, fp):
        self.samples, self.sr = librosa.load(fp, sr=None)
        self.duration = len(self.samples) / self.sr

    def load_annotations(self, fp, view=None, label_mapping=None):
        self.annotations = self.load_selection_table(fp, view=view, label_mapping=label_mapping)
        self.annotations['index'] = self.annotations.index

    def load_predictions(self, fp, view=None, label_mapping=None):
        self.predictions = self.load_selection_table(fp, view=view, label_mapping=label_mapping)
        self.predictions['index'] = self.predictions.index

    def threshold_class_predictions(self, class_threshold):
        assert self.unknown_label is not None, "unknown_label must be provided if thresholding predictions."
        for i in self.predictions.index:
            if self.predictions.loc[i, 'Class Prob'] < class_threshold:
                self.predictions.at[i, 'Annotation'] = self.unknown_label

    def compute_matching(self, IoU_minimum=0.5):
        ref = np.array(self.annotations[['Begin Time (s)', 'End Time (s)']]).T
        est = np.array(self.predictions[['Begin Time (s)', 'End Time (s)']]).T
        self.matching = match_events(ref, est, min_iou=IoU_minimum, method="fast")
        self.matched_annotations = [p[0] for p in self.matching]
        self.matched_predictions = [p[1] for p in self.matching]

    def evaluate(self):
        eval_sr = 50  # Evaluation sampling rate (Hz)
        dur_samples = int(self.duration * eval_sr)

        if self.label_set is None:
            TP = len(self.matching)
            FP = len(self.predictions) - TP
            FN = len(self.annotations) - TP

            seg_annotations = np.zeros(dur_samples)
            seg_predictions = np.zeros(dur_samples)
            for i, row in self.annotations.iterrows():
                start_sample = int(row['Begin Time (s)'] * eval_sr)
                end_sample = min(int(row['End Time (s)'] * eval_sr), dur_samples)
                seg_annotations[start_sample:end_sample] = 1
            for i, row in self.predictions.iterrows():
                start_sample = int(row['Begin Time (s)'] * eval_sr)
                end_sample = min(int(row['End Time (s)'] * eval_sr), dur_samples)
                seg_predictions[start_sample:end_sample] = 1

            TP_seg = int((seg_predictions * seg_annotations).sum())
            FP_seg = int((seg_predictions * (1 - seg_annotations)).sum())
            FN_seg = int(((1 - seg_predictions) * seg_annotations).sum())
            return {'all': {'TP': TP, 'FP': FP, 'FN': FN,
                            'TP_seg': TP_seg, 'FP_seg': FP_seg, 'FN_seg': FN_seg}}
        else:
            out = {label: {'TP': 0, 'FP': 0, 'FN': 0, 'TP_seg': 0, 'FP_seg': 0, 'FN_seg': 0}
                   for label in self.label_set}
            pred_label = np.array(self.predictions['Annotation'])
            annot_label = np.array(self.annotations['Annotation'])
            for p in self.matching:
                annotation = annot_label[p[0]]
                prediction = pred_label[p[1]]
                if self.unknown_label is not None and prediction == self.unknown_label:
                    continue
                elif annotation == prediction:
                    out[annotation]['TP'] += 1
                elif self.unknown_label is not None and annotation == self.unknown_label:
                    out[prediction]['FP'] -= 1

            for label in self.label_set:
                n_annot = int((annot_label == label).sum())
                n_pred = int((pred_label == label).sum())
                out[label]['FP'] = out[label]['FP'] + n_pred - out[label]['TP']
                out[label]['FN'] = out[label]['FN'] + n_annot - out[label]['TP']

                seg_annotations = np.zeros(dur_samples)
                seg_predictions = np.zeros(dur_samples)
                annot_sub = self.annotations[self.annotations["Annotation"] == label]
                pred_sub = self.predictions[self.predictions["Annotation"] == label]
                for i, row in annot_sub.iterrows():
                    start_sample = int(row['Begin Time (s)'] * eval_sr)
                    end_sample = min(int(row['End Time (s)'] * eval_sr), dur_samples)
                    seg_annotations[start_sample:end_sample] = 1
                for i, row in pred_sub.iterrows():
                    start_sample = int(row['Begin Time (s)'] * eval_sr)
                    end_sample = min(int(row['End Time (s)'] * eval_sr), dur_samples)
                    seg_predictions[start_sample:end_sample] = 1
                TP_seg = int((seg_predictions * seg_annotations).sum())
                FP_seg = int((seg_predictions * (1 - seg_annotations)).sum())
                FN_seg = int(((1 - seg_predictions) * seg_annotations).sum())
                out[label]['TP_seg'] = TP_seg
                out[label]['FP_seg'] = FP_seg
                out[label]['FN_seg'] = FN_seg

            return out

    def confusion_matrix(self):
        if self.label_set is None:
            return None
        else:
            confusion_matrix_labels = self.label_set.copy()
            if self.unknown_label is not None:
                confusion_matrix_labels.append(self.unknown_label)
            confusion_matrix_labels.append('None')
            confusion_matrix_size = len(confusion_matrix_labels)

            cm = np.zeros((confusion_matrix_size, confusion_matrix_size))
            none_idx = confusion_matrix_labels.index('None')

            pred_label = np.array(self.predictions['Annotation'])
            annot_label = np.array(self.annotations['Annotation'])

            for p in self.matching:
                annotation = annot_label[p[0]]
                prediction = pred_label[p[1]]
                cm_annot_idx = confusion_matrix_labels.index(annotation)
                cm_pred_idx = confusion_matrix_labels.index(prediction)
                cm[cm_pred_idx, cm_annot_idx] += 1

            for label in confusion_matrix_labels:
                if label == 'None':
                    continue
                idx = confusion_matrix_labels.index(label)
                n_pred = int((pred_label == label).sum())
                n_positive_detections = cm.sum(axis=1)[idx]
                n_false = n_pred - n_positive_detections
                cm[idx, none_idx] = n_false

                n_annot = int((annot_label == label).sum())
                n_positive_detections_col = cm.sum(axis=0)[idx]
                n_missed = n_annot - n_positive_detections_col
                cm[none_idx, idx] = n_missed

            return cm, confusion_matrix_labels