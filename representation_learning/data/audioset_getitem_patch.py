"""
Improved __getitem__ and _load patches for AudioSet dataset.

This module provides patches to fix various issues with AudioSet dataset loading,
including NaN handling and proper JSON label parsing.
"""

from __future__ import annotations

import json
import logging
from io import StringIO
from typing import TYPE_CHECKING, Any, Dict

import librosa
import numpy as np
import pandas as pd
from esp_data.io import anypath, audio_stereo_to_mono, read_audio

if TYPE_CHECKING:
    from esp_data.datasets.audioset import AudioSet

logger = logging.getLogger(__name__)


def patch_audioset_getitem() -> None:
    """Patch AudioSet __getitem__ method with the improved version.

    Returns:
        None: This function modifies the AudioSet class in-place.
    """

    def new_getitem(self: "AudioSet", idx: int) -> Dict[str, Any]:
        """Improved __getitem__ method that avoids start_time/end_time issues.

        Returns:
            Dict[str, Any]: Dictionary containing audio data and metadata.

        Raises:
            IndexError: If the index is out of range.
        """
        if idx >= len(self._data):
            raise IndexError(
                f"Index {idx} out of bounds for dataset of length {len(self._data)}."
            )

        row = self._data.iloc[idx].to_dict()

        # Ensure audio path is valid
        if self.data_root:
            audio_path = anypath(self.data_root) / row["local_path"]
        else:
            audio_path = anypath(row["local_path"])

        # Load full audio file (not using start_time/end_time to avoid segment issues)
        audio, sr = read_audio(audio_path)
        audio = audio.astype(np.float32)
        audio = audio_stereo_to_mono(audio, mono_method="average")

        if self.sample_rate is not None and sr != self.sample_rate:
            audio = librosa.resample(
                y=audio,
                orig_sr=sr,
                target_sr=self.sample_rate,
                scale=True,
                res_type="kaiser_best",
            )

        # AudioSet likes to call this 'audio'
        row["audio"] = audio

        if self.output_take_and_give:
            item = {}
            for key, value in self.output_take_and_give.items():
                item[value] = row[key]
        else:
            item = row

        return item

    return new_getitem


def patch_audioset_load() -> None:
    """Patch AudioSet _load method with proper JSON label parsing.

    Returns:
        None: This function modifies the AudioSet class in-place.
    """

    def new_load(self: "AudioSet") -> None:
        """Improved _load method with proper JSON label parsing.

        Raises:
            LookupError: If the split is not found in split_paths.
        """
        if self.split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split: {self.split}."
                "Expected one of {list(self.info.split_paths.keys())}"
            )

        location = self.info.split_paths[self.split]
        # Read CSV content
        csv_text = anypath(location).read_text(encoding="utf-8")

        # Converter to parse JSON-encoded labels into Python lists
        def parse_label(value: str) -> list:
            if pd.isna(value) or value == "":
                return []
            return json.loads(value)

        self._data = pd.read_csv(StringIO(csv_text), converters={"labels": parse_label})

    return new_load


def apply_audioset_patches() -> None:
    """Apply the patches to fix AudioSet NaN issues.

    Returns:
        None: Patches are applied in-place to the AudioSet class.
    """
    try:
        from esp_data.datasets.audioset import AudioSet

        # Patch the __getitem__ method
        AudioSet.__getitem__ = patch_audioset_getitem()

        # Patch the _load method for proper label parsing
        AudioSet._load = patch_audioset_load()

        logger.info("Applied AudioSet patches to fix NaN issues")
        return True

    except Exception as e:
        logger.error(f"Failed to apply AudioSet patches: {e}")
        return False


# Apply patches when this module is imported
if apply_audioset_patches():
    logger.info("AudioSet dataset patched successfully")
else:
    logger.warning("Failed to patch AudioSet dataset")
