from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Self

import cloudpathlib
import numpy as np
import pandas as pd

# Third-party I/O
import soundfile as sf
from google.cloud.storage.client import Client

from .config import DataConfig
from .transformations import (
    DataTransform,
    Filter,
    FilterConfig,
    Subsample,
    SubsampleConfig,
    TransformCfg,
)

if TYPE_CHECKING:
    from cloudpathlib import CloudPath
ANIMALSPEAK_PATH = "gs://animalspeak2/splits/v1/animalspeak_train_v1.3.csv"
ANIMALSPEAK_PATH_EVAL = "gs://animalspeak2/splits/v1/animalspeak_eval_v1.3.csv"


# -----------------------------------------------------------------------------
# gcloud helpers
# -----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_client() -> cloudpathlib.GSClient:  # pragma: no cover
    return cloudpathlib.GSClient(storage_client=Client(), file_cache_mode="close_file")


default_client = _get_client()  # module‑level singleton


class GSPath(cloudpathlib.GSPath):
    """Wrapper that injects a default GSClient so callers don't need env vars."""

    def __init__(
        self,
        client_path: str | Self | "CloudPath",
        *,
        client: cloudpathlib.GSClient = default_client,
    ) -> None:  # type: ignore[override]
        super().__init__(client_path, client=client)


# -----------------------------------------------------------------------------
# Core dataset
# -----------------------------------------------------------------------------


class AudioDataset:
    """
    Reads metadata from a CSV, loads audio, and yields a sample dict.

    Expected columns in the CSV:
    * 'filepath'  : str - path to the audio file on disk or a gs:// path.
    * <label_col> : str - value used for the target (e.g. species name).
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        data_config: DataConfig,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        preprocessor: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
        postprocessors: Optional[
            List[Callable[[Dict[str, Any]], Dict[str, Any]]]
        ] = None,
    ) -> None:
        super().__init__()
        # Ensure label column exists before dropping NAs
        if data_config.label_column not in metadata_df.columns:
            raise ValueError(
                f"Label column '{data_config.label_column}' not found in metadata."
            )
        self.metadata = metadata_df.reset_index(drop=True).dropna(
            subset=[data_config.label_column]
        )
        self.data_config = data_config
        self.preprocessor = preprocessor
        self.postprocessors = postprocessors or []

        self.audio_path_col = "gs_path"  # modify if your CSV uses a different name
        self.label_col = data_config.label_column

        # Build a label → index mapping for numeric targets
        unique_labels = sorted(self.metadata[self.label_col].unique())
        self.label2idx: Dict[str, int] = {lbl: i for i, lbl in enumerate(unique_labels)}

    def __len__(self) -> int:
        return len(self.metadata)

    # TODO (milad) we mostly care about iteration so define __iter__

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.metadata.iloc[idx]
        path_str: str = row[self.audio_path_col]

        # Use GSPath for gs:// paths if available, otherwise use the local Path.
        if path_str.startswith("gs://"):
            if GSPath is None:
                raise ImportError("cloudpathlib is required to handle gs:// paths.")
            audio_path = GSPath(path_str)
        else:
            audio_path = Path(path_str)

        # Open the audio file.
        with audio_path.open("rb") as f:
            audio, sr = sf.read(f)
        if audio.ndim == 2:  # stereo → mono
            audio = audio.mean(axis=1)

        item = {
            "raw_wav": audio.astype(np.float32),  # Keep as NumPy array initially
            "text_label": row[self.label_col],
            "label": self.label2idx[row[self.label_col]],
            "path": str(audio_path),
            "sample_rate": sr,
        }

        # ------------------------------------------------------------------
        # Optional post-processing (augmentations etc.) supplied by caller
        # ------------------------------------------------------------------
        for proc in self.postprocessors:
            item = proc(item)

        return item


# -----------------------------------------------------------------------------
# Helper builders (unchanged apart from target_len plumbing)
# -----------------------------------------------------------------------------


def _build_transforms(transform_configs: List[TransformCfg]) -> List[DataTransform]:
    transforms: List[DataTransform] = []
    for cfg in transform_configs:
        if isinstance(cfg, FilterConfig):
            transforms.append(Filter(cfg))
        elif isinstance(cfg, SubsampleConfig):
            transforms.append(Subsample(cfg))
        else:
            raise TypeError(
                "build_transforms() received an unexpected config type: "
                f"{type(cfg).__name__}"
            )
    return transforms


def _get_dataset_from_name(name: str, *, validation: bool = False) -> pd.DataFrame:  # noqa: D401
    name = name.lower().strip()

    if name == "animalspeak":
        animalspeak_path = ANIMALSPEAK_PATH_EVAL if validation else ANIMALSPEAK_PATH
        csv_path: Path | GSPath
        if animalspeak_path.startswith("gs://"):
            csv_path = GSPath(animalspeak_path)
        else:
            csv_path = Path(animalspeak_path)

        csv_text = csv_path.read_text(encoding="utf-8")
        df = pd.read_csv(StringIO(csv_text))
        df["gs_path"] = df["local_path"].apply(
            lambda x: (
                "/home/milad_earthspecies_org/data-migration/marius-highmem/mnt/foundation-model-data/audio_16k/"
                + x
            )
            # lambda x: "gs://foundation-model-data/audio_16k/" + x
        )
        return df

    raise NotImplementedError("Only AnimalSpeak dataset supported right now")


def get_dataset_dummy(
    data_config: DataConfig,
    *,
    transform: Optional[Callable] = None,  # kept for backward compatibility (unused)
    preprocessor: Optional[Callable] = None,
    validation: bool = False,
    postprocessors: Optional[List[Callable[[Dict[str, Any]], Dict[str, Any]]]] = None,
) -> AudioDataset:
    """Entry point that returns an ``AudioDataset`` with optional transforms.

    Returns:
        AudioDataset: The constructed audio dataset with optional transforms applied.
    """

    df = _get_dataset_from_name(data_config.dataset_name, validation=validation)

    if getattr(data_config, "transformations", None):
        transforms = _build_transforms(data_config.transformations)
        for transform in transforms:
            df = transform(df)

    return AudioDataset(
        metadata_df=df,
        data_config=data_config,
        preprocessor=preprocessor,
        postprocessors=postprocessors,
    )
