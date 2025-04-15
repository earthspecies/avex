import numpy as np
import random

def pad_or_window(
    wav: np.ndarray,
    target_len: int,
    window_selection: str = "random",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ensure the waveform has exactly `target_len` samples.

    Returns
    -------
    windowed_wav : np.ndarray [target_len]          – either truncated or padded
    padding_mask : np.ndarray [target_len] bool     – True where audio is real

    Notes
    -----
    * **Longer** than target_len → choose a window of length `target_len`.
      - `mode="random"` picks a random start index.
      - Other modes can be added later (e.g. "center").
    * **Shorter** than target_len → pad **zeros** at the end.
    """
    if window_selection != "random":
            raise NotImplementedError(f"Window mode '{window_selection}' not implemented")
    
    wav_len = len(wav)

    if wav_len == target_len:
        mask = np.ones(target_len, dtype=bool)
        return wav.astype(np.float32), mask

    if wav_len > target_len:  # need to crop
        
        start = random.randint(0, wav_len - target_len)
        end   = start + target_len
        window = wav[start:end]
        mask   = np.ones(target_len, dtype=bool)
        return window.astype(np.float32), mask

    # wav_len < target_len  → pad zeros
    pad_len = target_len - wav_len
    padded  = np.pad(wav, (0, pad_len), mode="constant")
    mask    = np.zeros(target_len, dtype=bool)
    mask[:wav_len] = True
    return padded.astype(np.float32), mask
