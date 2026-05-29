"""Custom transforms for representation learning data processing."""

from typing import Any, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from esp_data.backends import PandasBackend
from esp_data.transforms import register_transform
from pydantic import BaseModel, Field


class TrainValSplitConfig(BaseModel):
    """Configuration for TrainValSplitTransform.

    This transform splits a dataset into train and validation subsplits
    and returns either the train or validation portion based on the
    subset parameter.
    """

    type: Literal["train_val_split"]
    subset: Literal["train", "validation"] = Field(
        default="train", description="Which subset to return after splitting"
    )
    train_size: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Proportion of the dataset to include in the train split",
    )
    random_state: Optional[int] = Field(default=42, description="Random state for reproducible splits")
    stratify_column: Optional[str] = Field(default=None, description="Column name to use for stratified splitting")


class TrainValSplitTransform:
    """Transform that splits a dataset into train and validation subsplits.

    Parameters
    ----------
    subset : Literal["train", "validation"], default="train"
        Which subset to return after splitting
    train_size : float, default=0.8
        Proportion of the dataset to include in the train split
    random_state : Optional[int], default=42
        Random state for reproducible splits
    stratify_column : Optional[str], default=None
        Column name to use for stratified splitting.
    """

    def __init__(
        self,
        subset: Literal["train", "validation"] = "train",
        train_size: float = 0.8,
        random_state: Optional[int] = 42,
        stratify_column: Optional[str] = None,
    ) -> None:
        if not 0.0 < train_size < 1.0:
            raise ValueError(f"train_size must be between 0 and 1, got {train_size}")
        self.subset = subset
        self.train_size = train_size
        self.random_state = random_state
        self.stratify_column = stratify_column

    @classmethod
    def from_config(cls, cfg: TrainValSplitConfig) -> "TrainValSplitTransform":
        return cls(**cfg.model_dump(exclude=("type",)))

    def __call__(self, data: Any) -> Tuple[Any, dict]:  # noqa: ANN401
        n = len(data)

        if n == 0:
            return data, {"subset": self.subset, "original_size": 0, "split_size": 0}

        rng = np.random.default_rng(self.random_state)
        is_dataframe = isinstance(data, pd.DataFrame)

        if self.stratify_column is not None:
            if self.stratify_column not in data.columns:
                raise ValueError(
                    f"Stratify column '{self.stratify_column}' not found in data. "
                    f"Available columns: {list(data.columns)}"
                )
            train_idxs: list[int] = []
            val_idxs: list[int] = []

            if is_dataframe:
                for class_val in data[self.stratify_column].unique():
                    group_pos = list(np.where(data[self.stratify_column] == class_val)[0])
                    perm = rng.permutation(len(group_pos)).tolist()
                    n_train = max(1, int(len(perm) * self.train_size)) if len(perm) > 1 else len(perm)
                    train_idxs.extend([group_pos[perm[i]] for i in range(n_train)])
                    val_idxs.extend([group_pos[perm[i]] for i in range(n_train, len(perm))])
            else:
                # Per-class proportional split via esp_data backend API
                indexed = data.add_column("__row_idx__", list(range(n)))
                for class_val in indexed.get_unique(self.stratify_column):
                    group = indexed.filter_isin(self.stratify_column, [class_val])
                    group_row_idxs = [row["__row_idx__"] for row in group]
                    perm = rng.permutation(len(group_row_idxs)).tolist()
                    n_train = max(1, int(len(perm) * self.train_size)) if len(perm) > 1 else len(perm)
                    train_idxs.extend([group_row_idxs[perm[i]] for i in range(n_train)])
                    val_idxs.extend([group_row_idxs[perm[i]] for i in range(n_train, len(perm))])

            selected = sorted(train_idxs) if self.subset == "train" else sorted(val_idxs)
        else:
            perm = rng.permutation(n).tolist()
            n_train = int(n * self.train_size)
            selected = sorted(perm[:n_train]) if self.subset == "train" else sorted(perm[n_train:])

        result = data.iloc[selected].reset_index(drop=True) if is_dataframe else data[selected]

        return result, {
            "subset": self.subset,
            "original_size": n,
            "split_size": len(result),
            "train_size": self.train_size,
            "random_state": self.random_state,
            "stratify_column": self.stratify_column,
        }


class RLSubsampleConfig(BaseModel):
    """Configuration for RLSubsampleTransform."""

    type: Literal["rl_subsample"]
    ratio: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Ratio of samples to keep from the dataset",
    )
    max_samples: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum total number of samples to keep",
    )
    random_state: Optional[int] = Field(default=42, description="Random state for reproducible sampling")


class RLSubsampleTransform:
    """Transform that performs simple random subsampling.

    Parameters
    ----------
    ratio : float, default=1.0
        Ratio of samples to keep from the dataset
    max_samples : Optional[int], default=None
        Maximum total number of samples to keep
    random_state : Optional[int], default=42
        Random state for reproducible sampling
    """

    def __init__(
        self,
        ratio: float = 1.0,
        max_samples: Optional[int] = None,
        random_state: Optional[int] = 42,
    ) -> None:
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"ratio must be between 0 and 1, got {ratio}")
        if max_samples is not None and max_samples < 1:
            raise ValueError(f"max_samples must be >= 1, got {max_samples}")
        self.ratio = ratio
        self.max_samples = max_samples
        self.random_state = random_state

    @classmethod
    def from_config(cls, cfg: RLSubsampleConfig) -> "RLSubsampleTransform":
        return cls(**cfg.model_dump(exclude=("type",)))

    def __call__(self, data: Any) -> Tuple[Any, dict]:  # noqa: ANN401
        n_total = len(data)

        if n_total == 0:
            return data, {
                "rl_subsample": {
                    "original_size": 0,
                    "sampled_size": 0,
                    "ratio": self.ratio,
                    "max_samples": self.max_samples,
                }
            }

        n = int(n_total * self.ratio)
        n = min(n, n_total)
        if self.max_samples is not None:
            n = min(n, self.max_samples)

        seed = self.random_state if self.random_state is not None else 42
        result = data[0:0] if n == 0 else data.sample_rows(n, seed=seed)

        return result, {
            "rl_subsample": {
                "original_size": n_total,
                "sampled_size": len(result),
                "ratio": self.ratio,
                "max_samples": self.max_samples,
            }
        }


def _is_empty_labels(val: Any) -> bool:  # noqa: ANN401
    """Return True when a labels_as_list cell carries no usable string labels.

    Returns
    -------
    bool
        True when the cell is None, NaN, an empty list, or a list of only None values.
    """
    if val is None:
        return True
    if isinstance(val, float):
        return True  # pandas NaN
    if isinstance(val, (list, np.ndarray)):
        return len(val) == 0 or all(x is None for x in val)
    return False


class FillLabelsFromAnswerConfig(BaseModel):
    """Configuration for FillLabelsFromAnswer transform."""

    type: Literal["fill_labels_from_answer"]
    source: str = "answer"
    target: str = "labels_as_list"
    ignore_answers: list[str] = Field(default_factory=lambda: ["None"])


class FillLabelsFromAnswer:
    """Fill a null/empty list-label column from a plain-string fallback column.

    Handles BEANS train splits where labels_as_list is [] or [null] but the
    label string is stored in the `answer` column instead.  Rows with a
    properly-populated target are left untouched (no-op for val/test).

    answer values listed in `ignore_answers` (default: ["None"]) are treated as
    "no label" and will not be used to fill.
    """

    def __init__(
        self,
        *,
        source: str = "answer",
        target: str = "labels_as_list",
        ignore_answers: list[str] | None = None,
    ) -> None:
        self.source = source
        self.target = target
        self.ignore_answers: set[str] = set(ignore_answers) if ignore_answers is not None else {"None"}

    @classmethod
    def from_config(cls, cfg: FillLabelsFromAnswerConfig) -> "FillLabelsFromAnswer":
        return cls(**cfg.model_dump(exclude=("type",)))

    def __call__(self, data: Any) -> Tuple[Any, dict]:  # noqa: ANN401
        # Unwrap to pandas for row-wise label fix
        if hasattr(data, "unwrap"):
            frame = data.unwrap
            if hasattr(frame, "collect"):
                frame = frame.collect()
            df = frame.to_pandas() if hasattr(frame, "to_pandas") else frame
        else:
            df = data

        if self.target not in df.columns:
            raise ValueError(f"Target column '{self.target}' not found in data.")
        if self.source not in df.columns:
            raise ValueError(f"Source column '{self.source}' not found in data.")

        filled = 0

        def _fix(row: pd.Series) -> list:
            nonlocal filled
            if _is_empty_labels(row[self.target]):
                answer = row[self.source]
                if answer and isinstance(answer, str) and answer not in self.ignore_answers:
                    filled += 1
                    # BEANS train often uses comma-separated names in answer (e.g. enabirds).
                    return [part.strip() for part in answer.split(",") if part.strip()]
                return []
            val = row[self.target]
            return list(val) if not isinstance(val, list) else val

        df = df.copy()
        df[self.target] = df.apply(_fix, axis=1)

        # Always return PandasBackend so downstream multilabel_from_features
        # uses the pandas path, avoiding the Polars replace_strict/List(Null) bug.
        return PandasBackend(df), {"fill_labels_from_answer": {"filled": filled}}


# Register the transforms
register_transform(TrainValSplitConfig, TrainValSplitTransform)
register_transform(RLSubsampleConfig, RLSubsampleTransform)
register_transform(FillLabelsFromAnswerConfig, FillLabelsFromAnswer)
