import os
from functools import lru_cache
from typing import Self

import cloudpathlib
import numpy as np
import soundfile as sf
from google.cloud.storage.client import Client


@lru_cache(maxsize=1)
def _get_client() -> cloudpathlib.GSClient:
    return cloudpathlib.GSClient(storage_client=Client(), file_cache_mode="close_file")


class GSPath(cloudpathlib.GSPath):
    """
    A wrapper for the cloudpathlib GSPath that provides a default client.
    This avoids issues when the GOOGLE_APPLICATION_CREDENTIALS variable is not set.
    """

    def __init__(
        self,
        client_path: str | Self | cloudpathlib.AnyPath,
    ) -> None:
        super().__init__(client_path, client=_get_client())


# TODO (gagan): Use esp_data.io for reading audio files
def read_audio(audio_path: os.PathLike) -> tuple[np.ndarray, int]:
    """Read an audio file and return the audio data and sample rate.
    Parameters
    ----------
    audio_path : os.PathLike
        The path to the audio file. Can be a local path or a GSPath object.
    Returns
    -------
    tuple[np.ndarray, int]
        A tuple containing the audio data as a numpy array and
        the sample rate as an integer.

    Raises
    -------
    FileNotFoundError
        If the audio file does not exist.
    ValueError
        If the audio file cannot be read or is not in a supported format.
    """
    # Open the audio file. Using the .open('rb') method works for both local and
    # GSPath objects.
    try:
        with audio_path.open("rb") as f:
            audio, sr = sf.read(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Audio file not found: {audio_path}") from e
    except ValueError as e:
        raise ValueError(
            f"Could not read audio file {audio_path}. "
            "Ensure it is in a supported format (e.g., WAV, FLAC, MP3, OGG)."
        ) from e

    if audio.ndim > 1:
        # find the channel dim
        channel_dim = np.argmin(audio.shape)
        # take mean across the channel dimension
        audio = np.mean(audio, axis=channel_dim)

    return audio, sr


def read_audio_clip(
    audio_path: os.PathLike,
    start_time: float = None,
    end_time: float = None,
    to_mono: bool = True,
    mono_method: str = "average",
) -> tuple[np.ndarray, int]:
    """Read an audio file and return a specific portion of the audio data
    and sample rate.

    Parameters
    ----------
    audio_path : os.PathLike
        The path to the audio file. Can be a local path or a GSPath object.
    start_time : float, optional
        Start time in seconds. If None, starts from the beginning.
    end_time : float, optional
        End time in seconds. If None, reads until the end.
    to_mono : bool, optional
        Whether to convert multi-channel audio to mono. Default is True.
    mono_method : str, optional
        Method for converting to mono. Options are "average" (default) or
        "first_channel". Only used when to_mono is True.

    Returns
    -------
    tuple[np.ndarray, int]
        A tuple containing the audio data as a numpy array and
        the sample rate as an integer.

    Raises
    -------
    FileNotFoundError
        If the audio file does not exist.
    ValueError
        If the audio file cannot be read or is not in a supported format,
        if start_time >= end_time, or if mono_method is not valid.
    """
    # Validate parameters
    if start_time is not None and start_time < 0:
        raise ValueError("start_time must be non-negative")
    if end_time is not None and end_time < 0:
        raise ValueError("end_time must be non-negative")
    if start_time is not None and end_time is not None and start_time >= end_time:
        raise ValueError("start_time must be less than end_time")

    # Validate mono_method
    valid_mono_methods = {"average", "first_channel"}
    if mono_method not in valid_mono_methods:
        raise ValueError(
            f"mono_method must be one of {valid_mono_methods}, got '{mono_method}'"
        )

    # Open the audio file. Using the .open('rb') method works for both local and
    # GSPath objects.
    try:
        with audio_path.open("rb") as f:
            # Get file info first to calculate frame positions
            info = sf.info(f)
            sr = info.samplerate
            total_frames = info.frames

            # Calculate start and stop frames
            start_frame = int(start_time * sr) if start_time is not None else 0

            if end_time is not None:
                stop_frame = int(end_time * sr)
                # Ensure we don't go beyond the file
                stop_frame = min(stop_frame, total_frames)
            else:
                stop_frame = total_frames

            # Validate frame bounds
            if start_frame >= total_frames:
                raise ValueError(
                    f"start_time ({start_time}s) is beyond the audio duration"
                )

            # Calculate number of frames to read
            frames_to_read = stop_frame - start_frame

            if frames_to_read <= 0:
                raise ValueError("No audio data in the specified time range")

            # Read the specific portion of the audio
            f.seek(0)  # Reset file pointer
            audio = sf.read(f, frames=frames_to_read, start=start_frame)[0]

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Audio file not found: {audio_path}") from e
    except ValueError as e:
        # Re-raise our custom ValueError messages
        if "start_time" in str(e) or "end_time" in str(e) or "time range" in str(e):
            raise
        # Handle soundfile reading errors
        raise ValueError(
            f"Could not read audio file {audio_path}. "
            "Ensure it is in a supported format (e.g., WAV, FLAC, MP3, OGG)."
        ) from e
    except Exception as e:
        raise ValueError(
            f"Could not read audio file {audio_path}. "
            "Ensure it is in a supported format (e.g., WAV, FLAC, MP3, OGG)."
        ) from e

    # Handle multi-channel audio conversion
    if to_mono and audio.ndim > 1:
        if mono_method == "average":
            # Find the channel dimension (typically the smaller dimension)
            channel_dim = np.argmin(audio.shape)
            # Take mean across the channel dimension
            audio = np.mean(audio, axis=channel_dim)
        elif mono_method == "first_channel":
            # Take the first channel
            if audio.shape[0] < audio.shape[1]:
                # Channels are in the first dimension
                audio = audio[0]
            else:
                # Channels are in the second dimension
                audio = audio[:, 0]

    return audio, sr
