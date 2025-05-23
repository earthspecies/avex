"""Unit tests for audio processing utilities."""

import numpy as np
import pytest
import torch

from representation_learning.configs import AudioConfig
from representation_learning.data.audio_utils import (
    AudioProcessor,
    pad_or_window,
)


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

    def test_raw_representation(
        self, audio_config: AudioConfig, sample_waveform: torch.Tensor
    ) -> None:
        """Test raw waveform representation."""
        audio_config.representation = "raw"
        processor = AudioProcessor(audio_config)
        output = processor(sample_waveform)
        assert output.shape == (1, 16000)  # (batch, time)
        assert torch.allclose(output.squeeze(), sample_waveform)

    def test_spectrogram_representation(
        self, audio_config: AudioConfig, sample_waveform: torch.Tensor
    ) -> None:
        """Test spectrogram representation."""
        audio_config.representation = "spectrogram"
        processor = AudioProcessor(audio_config)
        output = processor(sample_waveform)
        expected_freq_bins = audio_config.n_fft // 2 + 1
        # Calculate time bins based on torch.stft formula
        expected_time_bins = (
            16000 + audio_config.hop_length - 1
        ) // audio_config.hop_length
        assert output.shape == (1, expected_freq_bins, expected_time_bins)

    def test_mel_spectrogram_representation(
        self, audio_config: AudioConfig, sample_waveform: torch.Tensor
    ) -> None:
        """Test mel spectrogram representation."""
        audio_config.representation = "mel_spectrogram"
        processor = AudioProcessor(audio_config)
        output = processor(sample_waveform)
        # Calculate time bins based on torch.stft formula
        expected_time_bins = (
            16000 + audio_config.hop_length - 1
        ) // audio_config.hop_length
        assert output.shape == (1, audio_config.n_mels, expected_time_bins)

    def test_normalization(
        self, audio_config: AudioConfig, sample_waveform: torch.Tensor
    ) -> None:
        """Test normalization behavior."""
        audio_config.normalize = True
        processor = AudioProcessor(audio_config)
        output = processor(sample_waveform)
        # Check if output is normalized (values between 0 and 1)
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)

    def test_invalid_representation(
        self, audio_config: AudioConfig, sample_waveform: torch.Tensor
    ) -> None:
        """Test invalid representation raises error."""
        audio_config.representation = "invalid"
        processor = AudioProcessor(audio_config)
        with pytest.raises(ValueError, match="Unknown representation"):
            processor(sample_waveform)

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
        assert output.shape == (batch_size, audio_config.n_mels, expected_time_bins)

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
