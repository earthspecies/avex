"""VFPA Killer Whales dataset

This dataset contains killer whale call annotations from the Vancouver Fraser Port Authority
(VFPA) as part of the DCLDE 2026 competition. The dataset includes both Southern Resident
Killer Whale (SRKW) calls and Transient (Bigg's) killer whale calls from the Pacific Northwest.

The data includes annotations from two main locations:
- Boundary Pass (BP): Strait of Georgia region
- Roberts Bank (RB): Fraser River region

Each annotation contains temporal and frequency bounds for individual calls, along with
detailed call type classifications using the standard SRKW call catalog (S-series) and
Transient call types (T-series).

The dataset structure includes:
* ``filename``        – audio file name
* ``path``           – relative path to audio file
* ``start``          – call start time in seconds
* ``end``            – call end time in seconds
* ``freq_min``       – minimum frequency in Hz
* ``freq_max``       – maximum frequency in Hz
* ``sound_id_species`` – species identification
* ``kw_ecotype``     – killer whale ecotype (SRKW, Transient, etc.)
* ``pod``            – pod identification (for SRKW)
* ``call_type``      – specific call type (S01, S44, T03, etc.)
* ``signal_type``    – signal classification
* ``confidence``     – annotation confidence level
* ``comments``       – additional notes

Examples
--------
>>> from esp_data.datasets import VFPAKillerWhales
>>> # Load all killer whale calls
>>> dataset = VFPAKillerWhales(split="all", sample_rate=16000)
>>> print(f"Total samples: {len(dataset)}")
Total samples: 1981

>>> # Load only SRKW calls
>>> srkw_dataset = VFPAKillerWhales(split="srkw", sample_rate=16000)
>>> print(f"SRKW samples: {len(srkw_dataset)}")
SRKW samples: 1542

>>> # Load only Transient calls
>>> transient_dataset = VFPAKillerWhales(split="transient", sample_rate=16000)
>>> print(f"Transient samples: {len(transient_dataset)}")
Transient samples: 439

>>> # Access a sample
>>> sample = dataset[0]
>>> sample.keys()
dict_keys(['filename', 'path', 'start', 'end', 'freq_min', 'freq_max',
           'sound_id_species', 'kw_ecotype', 'pod', 'call_type',
           'signal_type', 'confidence', 'comments', 'audio'])
"""

import os
from typing import Any, Dict, Iterator, Optional

import librosa
import numpy as np
import pandas as pd
from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio


@register_dataset
class VFPAKillerWhales(Dataset):
    """VFPA Killer Whales call dataset with SRKW and Transient calls."""

    info = DatasetInfo(
        name="vfpa_killer_whales",
        owner="dclde_2026",
        split_paths={
            "all": ("/home/david_earthspecies_org/esp-data/vfpa/annotations_processed/all.csv"),
            "srkw": (
                "/home/david_earthspecies_org/esp-data/"
                "dclde_2026_killer_whales/vfpa/annotations_processed/srkw.csv"
            ),
            "transient": (
                "/home/david_earthspecies_org/esp-data/"
                "dclde_2026_killer_whales/vfpa/annotations_processed/transient.csv"
            ),
            "boundary_pass": (
                "/home/david_earthspecies_org/esp-data/"
                "dclde_2026_killer_whales/vfpa/annotations_processed/"
                "boundary_pass.csv"
            ),
            "roberts_bank": (
                "/home/david_earthspecies_org/esp-data/"
                "dclde_2026_killer_whales/vfpa/annotations_processed/"
                "roberts_bank.csv"
            ),
            "srkw_standard": (
                "/home/david_earthspecies_org/esp-data/"
                "dclde_2026_killer_whales/vfpa/annotations_processed/"
                "srkw_standard.csv"
            ),
            "srkw_standard_balanced": (
                "/home/david_earthspecies_org/esp-data/"
                "dclde_2026_killer_whales/vfpa/annotations_processed/"
                "srkw_standard_balanced.csv"
            ),
        },
        version="1.0.0",
        description=(
            "VFPA killer whale call annotations from DCLDE 2026, including both "
            "Southern Resident (SRKW) and Transient killer whale calls from "
            "Pacific Northwest waters."
        ),
        sources=["VFPA", "DCLDE 2026"],
        license="CC-BY-4.0",
    )

    def __init__(
        self,
        split: str = "all",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: Optional[int] = None,
        data_root: Optional[str | AnyPathT] = None,
        call_types_filter: Optional[list[str]] = None,
    ) -> None:
        """Create a VFPAKillerWhales dataset instance.

        Parameters
        ----------
        split : str
            Which subset to load:
            - "all": All killer whale calls (default)
            - "srkw": Southern Resident calls only (S-series)
            - "transient": Transient calls only (T-series)
            - "boundary_pass": Boundary Pass location only
            - "roberts_bank": Roberts Bank location only
            - "srkw_standard": Standard SRKW calls only (no uncertain/multi-call)
            - "srkw_standard_balanced": Balanced SRKW standard calls
              (max 50 per type, no singletons)
        output_take_and_give : dict[str, str] | None
            Mapping from original column names to desired output names.
        sample_rate : int | None
            Target sample rate. If provided, audio is resampled.
        data_root : str | AnyPathT | None
            Custom root directory for audio files. If None, uses default.
        call_types_filter : list[str] | None
            Optional list of specific call types to include (e.g., ["S01", "S44"]).

        """
        super().__init__(output_take_and_give)
        self.split = split
        self.sample_rate = sample_rate
        self.call_types_filter = call_types_filter

        # Set data root - point to the DCLDE VFPA directory
        if data_root is None:
            self.data_root = anypath("../esp-data/vfpa")
        else:
            self.data_root = anypath(data_root)

        # Resolve relative paths to absolute paths for consistency
        # This ensures that relative paths like "../esp-data/vfpa" are resolved
        # to absolute paths based on the current working directory
        try:
            # Try to resolve the path if it's a local path (not a cloud path)
            if hasattr(self.data_root, "resolve") and not str(self.data_root).startswith(
                ("gs://", "s3://", "http://", "https://")
            ):
                resolved = self.data_root.resolve()
                # Only use resolved path if it exists or if original was relative
                if resolved.exists() or (
                    isinstance(data_root, str) and not os.path.isabs(data_root)
                ):
                    self.data_root = anypath(resolved)
        except (AttributeError, ValueError, OSError):
            # If resolution fails (e.g., for cloud paths), keep the original path
            pass

        self._data: pd.DataFrame | None = None
        self._load()

    @property
    def columns(self) -> list[str]:
        """Return the DataFrame column names.

        Returns
        -------
        list[str]
            Column names present in the loaded dataframe.

        Raises
        ------
        RuntimeError
            If the dataset has not been loaded yet.
        """
        if self._data is None:
            raise RuntimeError("Dataset not loaded. Call _load() first.")
        return list(self._data.columns)

    @property
    def available_splits(self) -> list[str]:
        """Return the names of available splits."""
        return list(self.info.split_paths.keys())

    def _load(self) -> None:
        """Load the preprocessed split CSV file.

        Raises
        ------
        LookupError
            If the requested split is not defined in ``split_paths``.
        FileNotFoundError
            If the CSV file cannot be found after trying multiple path resolution strategies.
        """
        if self.split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split: {self.split}. "
                f"Available splits: {list(self.info.split_paths.keys())}"
            )

        # Get the CSV path from split_paths
        original_path = self.info.split_paths[self.split]
        csv_path = None

        # Strategy 1: Try the path as-is
        potential_path = anypath(original_path)
        if potential_path.exists() and potential_path.is_file():
            csv_path = potential_path

        # Strategy 2: If path doesn't exist and is relative, try resolving relative to data_root
        if csv_path is None:
            # Check if path is relative (starts with ../ or ./)
            if isinstance(original_path, str) and (
                original_path.startswith("../") or original_path.startswith("./")
            ):
                # Normalize the relative path
                normalized = original_path
                if normalized.startswith("../"):
                    normalized = normalized[3:]
                elif normalized.startswith("./"):
                    normalized = normalized[2:]
                # Remove common prefixes that might be in the path
                if normalized.startswith("esp-data/"):
                    normalized = normalized.replace("esp-data/", "")
                if normalized.startswith("dclde_2026_killer_whales/"):
                    normalized = normalized.replace("dclde_2026_killer_whales/", "")
                if normalized.startswith("vfpa/"):
                    normalized = normalized.replace("vfpa/", "")

                # Try relative to data_root
                potential_path = anypath(self.data_root / normalized)
                if potential_path.exists() and potential_path.is_file():
                    csv_path = potential_path

        # Strategy 3: Try in annotations_processed subdirectory of data_root
        if csv_path is None:
            # Extract just the filename from the original path
            filename = os.path.basename(original_path)
            potential_path = anypath(self.data_root / "annotations_processed" / filename)
            if potential_path.exists() and potential_path.is_file():
                csv_path = potential_path

        # Strategy 4: Try parent directory of data_root (in case data_root points to vfpa/)
        if csv_path is None:
            # Extract filename
            filename = os.path.basename(original_path)
            # Try data_root parent / annotations_processed
            potential_path = anypath(self.data_root.parent / "annotations_processed" / filename)
            if potential_path.exists() and potential_path.is_file():
                csv_path = potential_path

        # If still not found, raise an error with helpful information
        if csv_path is None:
            filename = os.path.basename(original_path)
            normalized_for_display = (
                original_path.replace("esp-data/", "")
                .replace("dclde_2026_killer_whales/", "")
                .replace("vfpa/", "")
                if isinstance(original_path, str)
                else str(original_path)
            )
            raise FileNotFoundError(
                f"CSV file not found for split '{self.split}'\n"
                f"Original path: {original_path}\n"
                f"Data root: {self.data_root}\n"
                f"Tried paths:\n"
                f"  1. {anypath(original_path)}\n"
                f"  2. {anypath(self.data_root / normalized_for_display) if isinstance(original_path, str) else 'N/A'}\n"  # noqa: E501
                f"  3. {anypath(self.data_root / 'annotations_processed' / filename)}\n"
                f"  4. {anypath(self.data_root.parent / 'annotations_processed' / filename)}\n"
                f"Please check that the CSV file exists at one of these locations."
            )

        # Load the CSV file
        self._data = pd.read_csv(csv_path, keep_default_na=False, na_values=[""])

        # Apply call types filter if specified
        if self.call_types_filter:
            mask = self._data["call_type"].isin(self.call_types_filter)
            self._data = self._data[mask]
            # Reset index after filtering
            self._data = self._data.reset_index(drop=True)

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns
        -------
        int
            Dataset length.

        Raises
        ------
        RuntimeError
            If the dataset has not been loaded yet.
        """
        if self._data is None:
            raise RuntimeError("Dataset not loaded. Call _load() first.")
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a specific sample from the dataset.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        Dict[str, Any]
            Sample metadata and audio segment.

        Raises
        ------
        RuntimeError
            If the dataset has not been loaded yet.
        IndexError
            If ``idx`` is outside the dataset bounds.
        FileNotFoundError
            If the audio file cannot be found after trying multiple path resolution strategies.
        """
        if self._data is None:
            raise RuntimeError("Dataset not loaded. Call _load() first.")

        if idx >= len(self._data):
            raise IndexError(f"Index {idx} out of range (dataset length: {len(self._data)})")

        row = self._data.iloc[idx].to_dict()

        # Construct audio file path using filename mapping
        # The annotations use date-based paths, but actual files are in location subdirs
        filename = row["filename"]

        # Try to find the file in the audio subdirectories
        audio_path = None
        audio_base = self.data_root / "audio"

        # Check if audio_base exists before iterating
        if audio_base.exists() and audio_base.is_dir():
            # Check each subdirectory for the file
            for subdir in audio_base.iterdir():
                if subdir.is_dir():
                    potential_path = subdir / filename
                    if potential_path.exists():
                        audio_path = potential_path
                        break

        # Fallback: try multiple strategies to find the audio file
        if audio_path is None:
            annotation_path = row.get("path", "")
            audio_path = None  # Will be set by one of the strategies below

            # Strategy 1: Try using the annotation path relative to data_root/audio
            # Handle both relative paths (../esp-data/vfpa/audio/...) and absolute paths
            if annotation_path:
                # First, try resolving the path relative to data_root if it's a relative path
                if annotation_path.startswith("../") or annotation_path.startswith("./"):
                    # Resolve relative path: ../esp-data/vfpa/audio/file.wav
                    # relative to data_root = /home/david_earthspecies_org/esp-data/vfpa
                    # Should resolve to: /home/david_earthspecies_org/esp-data/vfpa/audio/file.wav
                    try:
                        # Create a path from data_root and resolve the relative path
                        resolved = (self.data_root / annotation_path).resolve()
                        if resolved.exists() and resolved.is_file():
                            audio_path = resolved
                    except (ValueError, RuntimeError):
                        # If resolution fails, fall through to normalization approach
                        pass

                # If resolution didn't work, try normalization approach
                if audio_path is None:
                    # Remove leading ../ or ./ components and use the path relative to audio
                    normalized_path = annotation_path
                    if normalized_path.startswith("../"):
                        normalized_path = normalized_path[3:]
                    elif normalized_path.startswith("./"):
                        normalized_path = normalized_path[2:]
                    # Remove any esp-data/vfpa/audio prefix if present
                    if normalized_path.startswith("esp-data/vfpa/audio/"):
                        normalized_path = normalized_path.replace("esp-data/vfpa/audio/", "")
                    elif normalized_path.startswith("vfpa/audio/"):
                        normalized_path = normalized_path.replace("vfpa/audio/", "")
                    elif normalized_path.startswith("audio/"):
                        normalized_path = normalized_path.replace("audio/", "")

                    potential_path = self.data_root / "audio" / normalized_path
                    if potential_path.exists() and potential_path.is_file():
                        audio_path = potential_path

            # Strategy 2: If path-based approach didn't work, try just filename
            if audio_path is None:
                potential_path = self.data_root / "audio" / filename
                if potential_path.exists() and potential_path.is_file():
                    audio_path = potential_path

            # Strategy 3: If still not found, recursively search for filename in audio directory
            if audio_path is None and audio_base.exists() and audio_base.is_dir():
                for audio_file in audio_base.rglob(filename):
                    if audio_file.is_file():
                        audio_path = audio_file
                        break

            # Strategy 4: Final fallback - use data_root/audio/filename even if it doesn't exist yet
            # (will raise error below if it truly doesn't exist)
            if audio_path is None:
                audio_path = self.data_root / "audio" / filename

        # Check if audio file exists before trying to load
        if not audio_path.exists() or not audio_path.is_file():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}\n"
                f"Expected data_root: {self.data_root}\n"
                f"Filename from annotation: {filename}\n"
                f"Path from annotation: {row.get('path', 'N/A')}\n"
                f"Audio base directory exists: {audio_base.exists() if audio_base else False}\n"
                f"Please check that:\n"
                f"  1. The data_root path '{self.data_root}' is correct\n"
                f"  2. The audio directory exists at '{self.data_root / 'audio'}'\n"
                f"  3. The audio file exists at the expected location"
            )

        # Load the full audio file
        audio, sr = read_audio(audio_path)
        audio = audio.astype(np.float32)
        audio = audio_stereo_to_mono(audio, mono_method="average")

        # Extract the call segment based on start/end times
        start_sample = int(row["start"] * sr)
        end_sample = int(row["end"] * sr)

        # Ensure we don't go beyond audio bounds
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)

        # Extract the call segment
        call_audio = audio[start_sample:end_sample]

        # Resample if requested
        if self.sample_rate is not None and sr != self.sample_rate:
            call_audio = librosa.resample(
                y=call_audio,
                orig_sr=sr,
                target_sr=self.sample_rate,
                scale=True,
                res_type="kaiser_best",
            )

        row["audio"] = call_audio

        # Add derived fields
        row["duration"] = row["end"] - row["start"]
        row["frequency_range"] = row["freq_max"] - row["freq_min"]

        # Apply output mapping if requested
        if self.output_take_and_give:
            mapped: Dict[str, Any] = {}
            for src, dst in self.output_take_and_give.items():
                if src in row:
                    mapped[dst] = row[src]
            # Always include audio unless explicitly mapped
            if "audio" not in [v for v in self.output_take_and_give.values()]:
                mapped["audio"] = row["audio"]
            return mapped

        return row

    @classmethod
    def from_config(
        cls, dataset_config: DatasetConfig
    ) -> tuple["VFPAKillerWhales", dict[str, Any]]:
        """Instantiate from a DatasetConfig.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Configuration dictionary containing dataset parameters.

        Returns
        -------
        tuple[VFPAKillerWhales, dict[str, Any]]
            Dataset instance and metadata from transformations.

        Raises
        ------
        LookupError
            If the specified split is not available.
        """
        cfg = dataset_config.model_dump(exclude=("dataset_name", "transformations"))

        split = cfg.get("split", "all")
        if split not in cls.info.split_paths:
            raise LookupError(
                f"Invalid split '{split}'. Available splits: {', '.join(cls.info.split_paths)}"
            )

        ds = cls(
            split=split,
            output_take_and_give=cfg.get("output_take_and_give"),
            data_root=cfg.get("data_root"),
            sample_rate=cfg.get("sample_rate"),
            call_types_filter=cfg.get("call_types_filter"),
        )

        if dataset_config.transformations:
            transform_metadata = ds.apply_transformations(dataset_config.transformations)
            return ds, transform_metadata
        return ds, {}

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over samples in the dataset.

        Yields
        ------
        Dict[str, Any]
            Sample metadata and audio payloads.
        """
        for i in range(len(self)):
            yield self[i]

    def __str__(self) -> str:
        """Return string representation of the dataset.

        Returns
        -------
        str
            Human-readable summary of the dataset state.
        """
        base = f"{self.info.name} (v{self.info.version})"

        # Add split-specific info if data is loaded
        split_info = ""
        if self._data is not None:
            split_info = f"\nCurrent split: {self.split} ({len(self._data)} samples)"
            if len(self._data) > 0:
                call_types = self._data["call_type"].nunique()
                split_info += f"\nUnique call types: {call_types}"

        return (
            f"{base}\n"
            f"Description: {self.info.description}\n"
            f"Sources: {', '.join(self.info.sources)}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
            f"{split_info}"
        )

    def get_call_type_distribution(self) -> pd.Series:
        """Get the distribution of call types in the current split.

        Returns
        -------
        pd.Series
            Call type counts sorted by frequency.

        Raises
        ------
        RuntimeError
            If the dataset has not been loaded yet.
        """
        if self._data is None:
            raise RuntimeError("Dataset not loaded. Call _load() first.")
        return self._data["call_type"].value_counts()

    def get_ecotype_distribution(self) -> pd.Series:
        """Get the distribution of ecotypes in the current split.

        Returns
        -------
        pd.Series
            Ecotype counts.

        Raises
        ------
        RuntimeError
            If the dataset has not been loaded yet.
        """
        if self._data is None:
            raise RuntimeError("Dataset not loaded. Call _load() first.")
        return self._data["kw_ecotype"].value_counts()
