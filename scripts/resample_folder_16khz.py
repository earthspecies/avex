#!/usr/bin/env python3
"""
Recursively copy a directory, resampling every audio file to 16 kHz
with librosa's “kaiser_best” resampler.

Example
-------
python resample_folder_16khz.py \
    --input_folder  /path/to/src \
    --output_folder /path/to/src_16khz \
    --num_workers   8
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Tuple

import librosa
import soundfile as sf
from tqdm import tqdm

TARGET_SR = 16_000
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus", ".aiff", ".au"}


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def _is_audio(p: Path) -> bool:
    return p.suffix.lower() in AUDIO_EXTS


def _resample_one(
    src: Path,
    dst: Path,
    root: Path,
    target_sr: int = TARGET_SR,
) -> Tuple[bool, str]:
    """
    Worker: copy or resample `src` to `dst`.

    Returns
    -------
    (success, message) where message is relative to `root`.
    """
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)

        if not _is_audio(src):
            shutil.copy2(src, dst)
            return True, f"copied {src.relative_to(root)}"

        # librosa loads: (n,) mono or (n_channels, n_samples) multi‑channel
        y, sr = librosa.load(src, sr=None, mono=False)
        if sr != target_sr:
            y = librosa.resample(
                y,
                orig_sr=sr,
                target_sr=target_sr,
                res_type="kaiser_best",
                scale=True,
            )

        # soundfile expects (frames, channels)
        sf.write(dst, y.T if y.ndim > 1 else y, target_sr, subtype="PCM_16")
        return True, f"resampled {src.relative_to(root)}"

    except Exception as e:  # pylint: disable=broad-except
        logging.error(f"{src}: {e}")
        return False, f"FAILED {src.relative_to(root)}"


def _gather_files(src_root: Path, out_root: Path) -> Iterable[Tuple[Path, Path]]:
    """Yield (src, dst) pairs for every file in src_root."""
    for p in src_root.rglob("*"):
        if p.is_file():
            yield p, out_root / p.relative_to(src_root)


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="Recursive 16 kHz resampler")
    parser.add_argument("--input_folder", required=True, type=Path)
    parser.add_argument("--output_folder", type=Path)
    parser.add_argument("--num_workers", type=int, default=mp.cpu_count())
    args = parser.parse_args()

    src_root = args.input_folder.expanduser().resolve()
    if not src_root.exists():
        sys.exit(f"Input folder {src_root} does not exist")

    out_root = (
        args.output_folder or src_root.parent / f"{src_root.name}_16khz"
    ).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("resample.log", "a")],
    )

    all_files = list(_gather_files(src_root, out_root))
    to_do = [(s, d) for s, d in all_files if not d.exists()]

    logging.info(
        "total files: %d, already done: %d, remaining: %d",
        len(all_files),
        len(all_files) - len(to_do),
        len(to_do),
    )

    if not to_do:
        logging.info("Nothing to do – output is up to date")
        return 0

    # Spawn is safest on SLURM / CUDA nodes.
    mp.set_start_method("spawn", force=True)

    ok = fail = 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {
            pool.submit(_resample_one, s, d, src_root): (s, d) for s, d in to_do
        }

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Resampling"):
            success, msg = fut.result()
            if success:
                ok += 1
                logging.info(msg)
            else:
                fail += 1
                logging.error(msg)

    logging.info("Done – success %d, failed %d", ok, fail)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
