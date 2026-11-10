"""Vendored XC_1M MAE encoder (Moritz, /mnt/home/moritz_mvp). xenomae.py is a byte-identical
copy of the MVP file -- do not edit it; adapt in avex/models/xenomae_model.py instead."""

from .xenomae import CLIP_SAMPLES, SAMPLE_RATE, XenoMAE, preprocess_audio

__all__ = ["CLIP_SAMPLES", "SAMPLE_RATE", "XenoMAE", "preprocess_audio"]
