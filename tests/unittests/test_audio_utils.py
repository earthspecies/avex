"""Unit tests for audio processing utilities."""

import numpy as np
import pytest
import torch

from representation_learning.configs import AudioConfig
from representation_learning.data.audio_utils import (
    AudioProcessor,
    pad_or_window,
)
from representation_learning.data.dataset import Collater


@pytest.fixture
def audio_config() -> AudioConfig:
    """Create a standard audio configuration for testing.

    Returns:
        AudioConfig: A standard audio configuration for testing.
    """
    return AudioConfig(
        sample_rate=16000,
        n_fft=2048,
        hop_length=512,
        win_length=2048,
        window="hann",
        n_mels=128,
        representation="mel_spectrogram",
        normalize=True,
        target_length_seconds=10,
        window_selection="random",
    )


@pytest.fixture
def sample_waveform() -> torch.Tensor:
    """Create a sample waveform for testing.

    Returns:
        torch.Tensor: A random waveform tensor of length 16000 (1 second at 16kHz).
    """
    return torch.randn(16000)  # 1 second of audio at 16kHz


class TestPadOrWindow:
    """Test cases for the pad_or_window function."""

    def test_exact_length(self) -> None:
        """Test when input length equals target length."""
        wav = torch.randn(100)
        target_len = 100
        processed_wav, mask = pad_or_window(wav, target_len)
        assert processed_wav.shape == (100,)
        assert mask.shape == (100,)
        assert not torch.all(mask)  # All False when lengths match (inverted mask)

    def test_padding(self) -> None:
        """Test padding shorter input."""
        wav = torch.randn(50)
        target_len = 100
        processed_wav, mask = pad_or_window(wav, target_len)
        assert processed_wav.shape == (100,)
        assert mask.shape == (100,)
        assert not torch.any(mask[:50])  # First 50 should be False (inverted mask)
        assert torch.all(mask[50:])  # Last 50 should be True (inverted mask)

    def test_window_random(self) -> None:
        """Test random window selection."""
        wav = torch.randn(200)
        target_len = 100
        processed_wav, mask = pad_or_window(wav, target_len, window_selection="random")
        assert processed_wav.shape == (100,)
        assert mask.shape == (100,)
        assert not torch.all(mask)  # All False for windowing (inverted mask)

    def test_window_center(self) -> None:
        """Test center window selection."""
        wav = torch.randn(200)
        target_len = 100
        processed_wav, mask = pad_or_window(wav, target_len, window_selection="center")
        assert processed_wav.shape == (100,)
        assert mask.shape == (100,)
        assert not torch.all(mask)  # All False for windowing (inverted mask)
        # Center window should take middle portion
        expected_start = (200 - 100) // 2
        assert torch.allclose(processed_wav, wav[expected_start : expected_start + 100])

    def test_window_start(self) -> None:
        """Test start window selection."""
        wav = torch.randn(200)
        target_len = 100
        processed_wav, mask = pad_or_window(wav, target_len, window_selection="start")
        assert processed_wav.shape == (100,)
        assert mask.shape == (100,)
        assert not torch.all(mask)  # All False for windowing (inverted mask)
        assert torch.allclose(processed_wav, wav[:100])  # Should take first 100 samples

    def test_invalid_window_selection(self) -> None:
        """Test invalid window selection raises error."""
        wav = torch.randn(200)
        target_len = 100
        with pytest.raises(ValueError, match="Unknown window selection"):
            pad_or_window(wav, target_len, window_selection="invalid")

    def test_mask_inversion(self) -> None:
        """Test mask inversion behavior."""
        wav = torch.randn(50)
        target_len = 100
        processed_wav, mask = pad_or_window(wav, target_len, invert=False)
        assert torch.all(mask[:50])  # First 50 should be True (non-inverted mask)
        assert not torch.any(mask[50:])  # Last 50 should be False (non-inverted mask)


class TestCollaterTwoStepProcessing:
    """Test cases for the Collater two-step audio processing."""

    def test_no_dataset_constraint(self) -> None:
        """Test collater without dataset constraint behaves like before."""
        collater = Collater(
            audio_max_length_seconds=5,  # Model wants 5s
            sr=16000,
            window_selection="start",
            num_labels=2,
            dataset_audio_max_length_seconds=None,  # No dataset constraint
        )

        # Create batch with 10-second audio
        long_audio = torch.randn(16000 * 10)  # 10 seconds
        batch = [{"audio": long_audio, "label": 0}]

        result = collater(batch)

        # Should be truncated to 5 seconds (model requirement)
        assert result["raw_wav"].shape == (1, 16000 * 5)

    def test_dataset_constraint_larger_than_model(self) -> None:
        """Test when dataset constraint is larger than model requirement."""
        collater = Collater(
            audio_max_length_seconds=3,  # Model wants 3s
            sr=16000,
            window_selection="start",
            num_labels=2,
            dataset_audio_max_length_seconds=5,  # Dataset allows up to 5s
        )

        # Create batch with 10-second audio
        long_audio = torch.randn(16000 * 10)  # 10 seconds
        batch = [{"audio": long_audio, "label": 0}]

        result = collater(batch)

        # Should be truncated to 3 seconds (model requirement is smaller)
        assert result["raw_wav"].shape == (1, 16000 * 3)

    def test_dataset_constraint_smaller_than_model(self) -> None:
        """Test when dataset constraint is smaller than model requirement."""
        collater = Collater(
            audio_max_length_seconds=10,  # Model wants 10s
            sr=16000,
            window_selection="start",
            num_labels=2,
            dataset_audio_max_length_seconds=5,  # Dataset allows only 5s
        )

        # Create batch with 15-second audio
        long_audio = torch.randn(16000 * 15)  # 15 seconds
        batch = [{"audio": long_audio, "label": 0}]

        result = collater(batch)

        # Should be:
        # 1. First truncated to 5s (dataset constraint)
        # 2. Then padded to 10s (model requirement)
        assert result["raw_wav"].shape == (1, 16000 * 10)

        # The first 5 seconds should be the original audio (non-zero)
        # The last 5 seconds should be padding (zeros)
        first_5s = result["raw_wav"][0, : 16000 * 5]
        last_5s = result["raw_wav"][0, 16000 * 5 :]

        # First 5s should not be all zeros (contains actual audio data)
        assert not torch.allclose(first_5s, torch.zeros_like(first_5s))
        # Last 5s should be all zeros (padding)
        assert torch.allclose(last_5s, torch.zeros_like(last_5s))

    def test_dataset_constraint_with_short_audio(self) -> None:
        """Test dataset constraint when audio is shorter than constraint."""
        collater = Collater(
            audio_max_length_seconds=10,  # Model wants 10s
            sr=16000,
            window_selection="start",
            num_labels=2,
            dataset_audio_max_length_seconds=5,  # Dataset allows up to 5s
        )

        # Create batch with 3-second audio (shorter than both constraints)
        short_audio = torch.randn(16000 * 3)  # 3 seconds
        batch = [{"audio": short_audio, "label": 0}]

        result = collater(batch)

        # Should be padded to 10s (model requirement)
        assert result["raw_wav"].shape == (1, 16000 * 10)

        # First 3s should be original, rest should be padding
        first_3s = result["raw_wav"][0, : 16000 * 3]
        padding = result["raw_wav"][0, 16000 * 3 :]

        assert torch.allclose(first_3s, short_audio)
        assert torch.allclose(padding, torch.zeros_like(padding))


class TestAudioProcessor:
    """Test cases for the AudioProcessor class."""

    def test_init(self, audio_config: AudioConfig) -> None:
        """Test AudioProcessor initialization."""
        processor = AudioProcessor(audio_config)
        assert processor.sr == audio_config.sample_rate
        assert processor.n_fft == audio_config.n_fft
        assert processor.hop_length == audio_config.hop_length
        assert processor.win_length == audio_config.win_length
        assert processor.window_type == audio_config.window
        assert processor.n_mels == audio_config.n_mels
        assert processor.representation == audio_config.representation
        assert processor.normalize == audio_config.normalize

    def test_raw_audio_processor(self, audio_config: AudioConfig) -> None:
        """Test raw audio processing."""
        audio_config.representation = "raw"
        processor = AudioProcessor(audio_config)

        # Test with 1D input
        wav_1d = torch.randn(16000)
        output = processor(wav_1d)
        assert output.shape == (1, 16000)

        # Test with 2D input
        wav_2d = torch.randn(2, 16000)
        output = processor(wav_2d)
        assert output.shape == (2, 16000)

    def test_spectrogram_processor(self, audio_config: AudioConfig) -> None:
        """Test spectrogram processing."""
        audio_config.representation = "spectrogram"
        processor = AudioProcessor(audio_config)

        wav = torch.randn(1, 16000)
        output = processor(wav)

        # Check output shape - should be (batch, freq, time)
        assert len(output.shape) == 3
        assert output.shape[0] == 1  # batch size
        assert output.shape[1] == audio_config.n_fft // 2 + 1  # frequency bins

    def test_mel_spectrogram_processor(self, audio_config: AudioConfig) -> None:
        """Test mel spectrogram processing."""
        audio_config.representation = "mel_spectrogram"
        processor = AudioProcessor(audio_config)

        wav = torch.randn(1, 16000)
        output = processor(wav)

        # Check output shape - should be (batch, n_mels, time)
        assert len(output.shape) == 3
        assert output.shape[0] == 1  # batch size
        assert output.shape[1] == audio_config.n_mels  # mel bins

    def test_invalid_representation(self, audio_config: AudioConfig) -> None:
        """Test that invalid representation raises error."""
        audio_config.representation = "invalid"
        processor = AudioProcessor(audio_config)

        wav = torch.randn(1, 16000)
        with pytest.raises(ValueError, match="Unknown representation"):
            processor(wav)

    def test_invalid_window_type(self) -> None:
        """Test invalid window type raises error."""
        # Create a valid config first
        audio_config = AudioConfig()
        processor = AudioProcessor(audio_config)
        # Then try to set an invalid window type
        processor.window_type = "invalid"
        with pytest.raises(ValueError, match="Unknown window type"):
            processor._get_window()

    def test_batch_processing(self, audio_config: AudioConfig) -> None:
        """Test processing multiple waveforms in a batch."""
        batch_size = 4
        waveforms = torch.randn(batch_size, 16000)
        processor = AudioProcessor(audio_config)
        output = processor(waveforms)
        # Calculate time bins based on torch.stft formula
        expected_time_bins = (
            16000 + audio_config.hop_length - 1
        ) // audio_config.hop_length
        assert output.shape == (
            batch_size,
            audio_config.n_mels,
            expected_time_bins,
        )

    def test_numpy_input(self, audio_config: AudioConfig) -> None:
        """Test processing numpy array input."""
        waveform = np.random.randn(16000)
        processor = AudioProcessor(audio_config)
        # Convert numpy array to torch tensor before processing
        waveform_tensor = torch.from_numpy(waveform).float()
        output = processor(waveform_tensor)
        # Calculate time bins based on torch.stft formula
        expected_time_bins = (
            16000 + audio_config.hop_length - 1
        ) // audio_config.hop_length
        assert output.shape == (1, audio_config.n_mels, expected_time_bins)
