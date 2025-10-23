# Export the audio processor that's actually used by the HuggingFace implementation
from .audio_processor import EATAudioProcessor  # noqa: F401

__all__ = ["EATAudioProcessor"]
