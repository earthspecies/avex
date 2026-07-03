"""Temporary BirdSet dataset with pre-refresh train/validation splits.

alp-data v0.1.0 (PR #243) dropped train splits from ``birdset``. This module
registers ``birdset_train`` with the split paths from pre-refresh alp-data
(``f2d0f0df``) so existing benchmark configs can keep using ``POW-train``,
``PER-train``, etc. Remove once train splits are restored upstream.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import librosa
import numpy as np
from alp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from alp_data.backends import BackendType
from alp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

_LEGACY_SPLIT_PATHS: dict[str, str] = {
    "HSN-train": "gs://foundation-model-data/data/birdset-train/HSN/HSN_taxonomic.jsonl",
    "HSN-validation": "gs://foundation-model-data/data/birdset-train/HSN/HSN_taxonomic.jsonl",
    "HSN-test": "gs://foundation-model-data/data/birdset-test/HSN/HSN_taxonomic.jsonl",
    "NBP-train": "gs://foundation-model-data/data/birdset-train/NBP/NBP_taxonomic.jsonl",
    "NBP-validation": "gs://foundation-model-data/data/birdset-train/NBP/NBP_taxonomic.jsonl",
    "NBP-test": "gs://foundation-model-data/data/birdset-test/NBP/NBP_taxonomic.jsonl",
    "NES-train": "gs://foundation-model-data/data/birdset-train/NES/NES_taxonomic.jsonl",
    "NES-validation": "gs://foundation-model-data/data/birdset-train/NES/NES_taxonomic.jsonl",
    "NES-test": "gs://foundation-model-data/data/birdset-test/NES/NES_taxonomic.jsonl",
    "PER-train": "gs://foundation-model-data/data/birdset-train/PER/PER_taxonomic.jsonl",
    "PER-validation": "gs://foundation-model-data/data/birdset-train/PER/PER_taxonomic.jsonl",
    "PER-test": "gs://foundation-model-data/data/birdset-test/PER/PER_taxonomic.jsonl",
    "POW-train": "gs://foundation-model-data/data/birdset-train/POW/POW_taxonomic.jsonl",
    "POW-validation": "gs://foundation-model-data/data/birdset-train/POW/POW_taxonomic.jsonl",
    "POW-test": "gs://foundation-model-data/data/birdset-test/POW/POW_taxonomic.jsonl",
    "UHH-train": "gs://foundation-model-data/data/birdset-train/UHH/UHH_taxonomic.jsonl",
    "UHH-validation": "gs://foundation-model-data/data/birdset-train/UHH/UHH_taxonomic.jsonl",
    "UHH-test": "gs://foundation-model-data/data/birdset-test/UHH/UHH_taxonomic.jsonl",
    "SSW-train": "gs://foundation-model-data/data/birdset-train/SSW/SSW_taxonomic.jsonl",
    "SSW-validation": "gs://foundation-model-data/data/birdset-train/SSW/SSW_taxonomic.jsonl",
    "SSW-test": "gs://foundation-model-data/data/birdset-test/SSW/SSW_taxonomic.jsonl",
    "SNE-train": "gs://foundation-model-data/data/birdset-train/SNE/SNE_taxonomic.jsonl",
    "SNE-validation": "gs://foundation-model-data/data/birdset-train/SNE/SNE_taxonomic.jsonl",
    "SNE-test": "gs://foundation-model-data/data/birdset-test/SNE/SNE_taxonomic.jsonl",
    "XCM": "gs://foundation-model-data/data/birdset-train/XCM/XCM_taxonomic.jsonl",
}


@register_dataset
class BirdSetTrainSplits(Dataset):
    """BirdSet with legacy train/validation/test split paths."""

    info = DatasetInfo(
        name="birdset_train",
        owner="marius; gagan",
        split_paths=_LEGACY_SPLIT_PATHS,
        version="0.0.1",
        description=(
            "Temporary BirdSet wrapper restoring pre-refresh train splits "
            "(see alp-data PR #243). Uses JSONL manifests and local/GCS audio "
            "via the ``path`` column."
        ),
        sources=["HSN", "NBP", "NES", "PER", "POW", "SSW", "SNE", "UHH", "XCM"],
        license="CC-BY-4.0, CC0",
    )

    def __init__(
        self,
        split: str = "HSN-train",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = None,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "pandas",
        streaming: bool = False,
    ) -> None:
        """Initialize BirdSet with legacy split paths.

        Parameters
        ----------
        split
            Split name, e.g. ``POW-train``.
        output_take_and_give
            Optional column rename/filter map.
        sample_rate
            Target audio sample rate; resamples when set.
        data_root
            Root prepended to each row's ``path`` column.
        backend
            DataFrame backend (``pandas`` or ``polars``).
        streaming
            Whether to stream JSONL/CSV from remote storage.
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self._data: Any = None
        self._load()
        self.sample_rate = sample_rate

        if data_root is None:
            self.data_root = anypath("gs://foundation-model-data/")
        else:
            self.data_root = data_root

    @property
    def columns(self) -> Sequence[str]:
        """Return dataset column names."""
        return list(self._data.columns)

    @property
    def available_splits(self) -> Sequence[str]:
        """Return registered split names."""
        return list(self.info.split_paths.keys())

    def _load(self) -> None:
        """Load the requested split manifest.

        Raises
        ------
        LookupError
            If ``split`` is not in the registered split paths.
        """
        if self.split not in self.info.split_paths:
            raise LookupError(f"Invalid split: {self.split}. Expected one of {list(self.info.split_paths.keys())}")

        location = self.info.split_paths[self.split]
        if anypath(location).suffix == ".jsonl":
            self._data = self._backend_class.from_json(location, lines=True, streaming=self._streaming)
        else:
            self._data = self._backend_class.from_csv(
                location,
                streaming=self._streaming,
            )

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["BirdSetTrainSplits", dict[str, Any]]:
        """Build a dataset instance from a ``DatasetConfig``.

        Returns
        -------
        tuple[BirdSetTrainSplits, dict[str, Any]]
            Dataset instance and transformation metadata.
        """
        cfg = dataset_config.model_dump(exclude={"dataset_name", "transformations"})
        # Benchmark transforms (train_val_split, etc.) use pandas-style column access.
        backend: BackendType = cfg["backend"]
        if backend == "polars":
            backend = "pandas"

        ds = cls(
            split=cfg["split"],
            output_take_and_give=cfg["output_take_and_give"],
            data_root=cfg["data_root"],
            sample_rate=cfg["sample_rate"],
            backend=backend,
            streaming=cfg["streaming"],
        )

        if dataset_config.transformations:
            transform_metadata = ds.apply_transformations(dataset_config.transformations)
            return ds, transform_metadata

        return ds, {}

    def __len__(self) -> int:
        """Return the number of rows in the loaded split.

        Returns
        -------
        int
            Row count.

        Raises
        ------
        RuntimeError
            If the split has not been loaded yet.
        """
        if self._data is None:
            raise RuntimeError("No split has been loaded yet. Call load() first.")
        if self._streaming:
            raise NotImplementedError("Length is not available in streaming mode. Iterate over the dataset.")
        return len(self._data)

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        audio_path = anypath(self.data_root) / row["path"]
        audio, sample_rate = read_audio(audio_path)
        audio = audio.astype(np.float32)
        audio = audio_stereo_to_mono(audio, mono_method="average")

        if self.sample_rate is not None and sample_rate != self.sample_rate:
            audio = librosa.resample(
                y=audio,
                orig_sr=sample_rate,
                target_sr=self.sample_rate,
                scale=True,
                res_type="kaiser_best",
            )
            sample_rate = self.sample_rate

        row["audio"] = audio
        row["sample_rate"] = sample_rate

        if self.output_take_and_give:
            item = {}
            for key, value in self.output_take_and_give.items():
                item[value] = row[key]
            return item

        return row

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return one processed sample.

        Returns
        -------
        dict[str, Any]
            Processed sample with audio and metadata.
        """
        row = self._data[idx]
        return self._process(row)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over processed samples.

        Yields
        ------
        dict[str, Any]
            Processed sample with audio and metadata.
        """
        for row in self._data:
            yield self._process(row)

    def __str__(self) -> str:
        """Return a short human-readable description.

        Returns
        -------
        str
            Human-readable summary of name, version, split, and available splits.
        """
        return (
            f"{self.info.name} (v{self.info.version}), split={self.split}\n"
            f"Description: {self.info.description}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
