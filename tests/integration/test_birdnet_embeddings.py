"""Integration tests for BirdNET embedding extraction at various sample rates.

Verifies the fixes for issue #161: BirdNET embedding extraction broken for
non-48 kHz input. Tests use real audio files from tests/samples/.
"""

import numpy as np
import pytest
import soundfile as sf
import torch

from avex.models.birdnet import Model as BirdNetModel

# ---------------------------------------------------------------------------
# Paths to real audio samples
# ---------------------------------------------------------------------------
SAMPLES_16KHZ = [
    "tests/samples/animalspeak2/16khz/Xeno-canto/XC564654-200602-006_NR5N6_20h14_Sturtur.flac",
    "tests/samples/animalspeak2/16khz/iNaturalist/246886.flac",
]

SAMPLE_48KHZ = "tests/samples/insectset_459/Diceroprocta_eugraphica_IN50366825_110124_cut.wav"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def model() -> BirdNetModel:
    # BirdNet model without classifier, session-scoped for speed
    return BirdNetModel(num_classes=None, device="cpu")


def _load_mono(path: str) -> tuple[np.ndarray, int]:
    # Load an audio file and return (mono waveform, sample_rate).
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data, sr


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestBirdNetEmbeddings16kHz:
    """Tests targeting the 16 kHz input path (the primary bug trigger)."""

    def test_different_clips_produce_different_embeddings(self, model: BirdNetModel) -> None:
        """Two distinct 16 kHz clips must NOT produce identical embeddings."""
        embeddings = []
        for path in SAMPLES_16KHZ:
            wave, sr = _load_mono(path)
            assert sr == 16_000, f"Expected 16 kHz sample, got {sr}"
            # Take up to 5 seconds
            clip = wave[: sr * 5]
            tensor = torch.from_numpy(clip).unsqueeze(0)  # (1, T)
            emb = model.extract_embeddings(tensor, aggregation="mean")
            embeddings.append(emb)

        # Embeddings should differ for different audio content
        cosine_sim = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1]).item()
        assert cosine_sim < 0.99, (
            f"Embeddings for different 16 kHz clips are nearly identical "
            f"(cosine similarity {cosine_sim:.4f}), indicating the bug is not fixed"
        )

    def test_short_clip_produces_valid_embedding(self, model: BirdNetModel) -> None:
        """A clip shorter than 1.5 s at 16 kHz must still produce a valid embedding."""
        wave, sr = _load_mono(SAMPLES_16KHZ[0])
        # Take only 1 second (< 1.5 s threshold)
        short_clip = wave[: sr * 1]
        assert len(short_clip) < int(1.5 * sr), "Clip should be shorter than 1.5 s"

        tensor = torch.from_numpy(short_clip).unsqueeze(0)
        emb = model.extract_embeddings(tensor, aggregation="mean")

        assert emb.shape == (1, 1024)
        assert not torch.all(emb == 0), "Embedding should not be all zeros"
        assert not torch.isnan(emb).any()

    def test_embeddings_are_deterministic(self, model: BirdNetModel) -> None:
        """Same 16 kHz clip extracted twice must produce identical embeddings."""
        wave, sr = _load_mono(SAMPLES_16KHZ[0])
        clip = wave[: sr * 5]
        tensor = torch.from_numpy(clip).unsqueeze(0)

        emb1 = model.extract_embeddings(tensor, aggregation="mean")
        emb2 = model.extract_embeddings(tensor, aggregation="mean")

        torch.testing.assert_close(emb1, emb2)


class TestBirdNetEmbeddings48kHz:
    """Tests for native 48 kHz input (regression guard)."""

    def test_48khz_produces_valid_embeddings(self, model: BirdNetModel) -> None:
        """48 kHz audio should produce valid, non-zero 1024-d embeddings."""
        wave, sr = _load_mono(SAMPLE_48KHZ)
        # Take up to 5 seconds
        clip = wave[: sr * 5]
        tensor = torch.from_numpy(clip).unsqueeze(0)

        emb = model.extract_embeddings(tensor, aggregation="mean")

        assert emb.shape == (1, 1024)
        assert not torch.all(emb == 0), "Embedding should not be all zeros"
        assert not torch.isnan(emb).any()
        assert not torch.isinf(emb).any()


class TestBirdNetEmbeddingsBatch:
    """Tests for batch embedding extraction."""

    def test_batch_extraction(self, model: BirdNetModel) -> None:
        """Batch of 2 clips at 16 kHz should return (2, 1024) embeddings."""
        clips = []
        for path in SAMPLES_16KHZ:
            wave, sr = _load_mono(path)
            clip = wave[: sr * 5]
            clips.append(clip)

        # Pad to same length for batching
        max_len = max(len(c) for c in clips)
        padded = [np.pad(c, (0, max_len - len(c))) for c in clips]
        batch = torch.from_numpy(np.stack(padded))  # (2, T)

        emb = model.extract_embeddings(batch, aggregation="mean")

        assert emb.shape == (2, 1024)
        assert not torch.isnan(emb).any()
        # The two embeddings should differ
        assert not torch.allclose(emb[0], emb[1]), "Batch embeddings should differ for different clips"
