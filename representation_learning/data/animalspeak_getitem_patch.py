"""
Improved __getitem__ patch for AnimalSpeak dataset to handle audio chunking.

This patch addresses memory issues by loading smaller audio chunks instead of
full recordings, particularly helpful for long recordings.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Any, Dict

import librosa
import numpy as np
from esp_data.io import anypath, audio_stereo_to_mono, read_audio

if TYPE_CHECKING:
    from esp_data.datasets.animalspeak import AnimalSpeak

logger = logging.getLogger(__name__)


def patch_animalspeak_getitem(
    max_duration_seconds: float = 30.0, chunk_selection: str = "random"
) -> bool:
    """Patch AnimalSpeak __getitem__ method to load audio chunks.

    Parameters
    ----------
    max_duration_seconds : float
        Maximum duration in seconds to load from each audio file.
    chunk_selection : str
        How to select the chunk. Options: "random", "start".

    Returns
    -------
    bool
        True if the patch was successfully applied.
    """

    def new_getitem(self: "AnimalSpeak", idx: int) -> Dict[str, Any]:
        """Improved __getitem__ method that loads audio chunks instead of full files.

        Returns
        -------
        Dict[str, Any]
            Sample dictionary containing audio data and metadata.

        Raises
        ------
        IndexError
            If the index is out of bounds for the dataset.
        ValueError
            If there are issues loading the audio file.
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

        try:
            # First, get audio duration without loading the full file
            total_duration = librosa.get_duration(path=str(audio_path))

            # Determine chunk to load
            if total_duration <= max_duration_seconds:
                # Load full audio if it's already short enough
                start_time = 0
                duration = None
            else:
                # Calculate chunk parameters
                if chunk_selection == "random":
                    max_start_time = total_duration - max_duration_seconds
                    start_time = random.uniform(0, max_start_time)
                elif chunk_selection == "start":
                    start_time = 0
                else:
                    raise ValueError(f"Unknown chunk_selection: {chunk_selection}")

                duration = max_duration_seconds

            # Load the audio chunk
            if duration is not None:
                # Load specific segment using librosa (keep original sample rate)
                audio, sr = librosa.load(
                    str(audio_path),
                    sr=None,  # Keep original sample rate
                    offset=start_time,
                    duration=duration,
                    dtype=np.float32,
                )
            else:
                # Load full audio
                audio, sr = read_audio(audio_path)
                audio = audio.astype(np.float32)

        except Exception as e:
            logger.warning(
                f"Failed to load audio chunk for {audio_path}, "
                f"falling back to full load: {e}"
            )
            # Fallback to original method
            audio, sr = read_audio(audio_path)
            audio = audio.astype(np.float32)

        # Convert to mono
        audio = audio_stereo_to_mono(audio, mono_method="average")

        # Resample if needed
        if self.sample_rate is not None and sr != self.sample_rate:
            audio = librosa.resample(
                y=audio,
                orig_sr=sr,
                target_sr=self.sample_rate,
                scale=True,
                res_type="kaiser_best",
            )

        # AnimalSpeak uses 'audio' key
        row["audio"] = audio

        if self.output_take_and_give:
            item = {}
            for key, value in self.output_take_and_give.items():
                item[value] = row[key]
        else:
            item = row

        return item

    return new_getitem


def apply_animalspeak_patches(
    max_duration_seconds: float = 30.0, chunk_selection: str = "random"
) -> bool:
    """Apply the patches to handle long AnimalSpeak audio files.

    Parameters
    ----------
    max_duration_seconds : float
        Maximum duration in seconds to load from each audio file.
    chunk_selection : str
        How to select the chunk. Options: "random", "start".

    Returns
    -------
    bool
        True if patches were successfully applied.
    """
    try:
        from esp_data.datasets.animalspeak import AnimalSpeak

        # Patch the __getitem__ method
        AnimalSpeak.__getitem__ = patch_animalspeak_getitem(
            max_duration_seconds=max_duration_seconds, chunk_selection=chunk_selection
        )

        logger.info(
            f"Applied AnimalSpeak patches with max_duration={max_duration_seconds}s, "
            f"chunk_selection={chunk_selection}"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to apply AnimalSpeak patches: {e}")
        return False
