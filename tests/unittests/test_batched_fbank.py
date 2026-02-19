"""Tests that _BatchedFbank reproduces torchaudio.compliance.kaldi.fbank exactly."""

import torch
import torchaudio.compliance.kaldi as ta_kaldi

from avex.models.beats.beats import _BatchedFbank


def _kaldi_fbank_loop(waveforms: torch.Tensor, num_mel_bins: int = 128) -> torch.Tensor:
    """Reference: per-sample kaldi fbank, matching the old BEATs.preprocess loop.

    Returns:
        Stacked fbank features of shape ``[B, num_frames, num_mel_bins]``.
    """
    fbanks = []
    for waveform in waveforms:
        waveform = waveform.unsqueeze(0)
        fb = ta_kaldi.fbank(
            waveform,
            num_mel_bins=num_mel_bins,
            sample_frequency=16000,
            frame_length=25,
            frame_shift=10,
        )
        fbanks.append(fb)
    return torch.stack(fbanks, dim=0)


class TestBatchedFbankEquivalence:
    """Verify _BatchedFbank matches kaldi fbank for various inputs."""

    def _check(
        self,
        waveforms: torch.Tensor,
        num_mel_bins: int = 128,
        *,
        atol: float = 1e-4,
        rtol: float = 1e-4,
    ) -> None:
        scaled = waveforms * 2**15

        fbank_module = _BatchedFbank(num_mel_bins=num_mel_bins)
        batched = fbank_module(scaled)

        reference = _kaldi_fbank_loop(scaled, num_mel_bins=num_mel_bins)

        assert batched.shape == reference.shape, (
            f"Shape mismatch: batched {batched.shape} vs reference {reference.shape}"
        )
        torch.testing.assert_close(batched, reference, atol=atol, rtol=rtol)

    def test_single_sample_sine(self) -> None:
        t = torch.linspace(0, 1, 16000)
        waveforms = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)
        self._check(waveforms)

    def test_batch_random(self) -> None:
        torch.manual_seed(42)
        waveforms = torch.randn(4, 16000)
        self._check(waveforms)

    def test_ten_second_clip(self) -> None:
        torch.manual_seed(7)
        waveforms = torch.randn(2, 160000)
        self._check(waveforms)

    def test_short_clip(self) -> None:
        torch.manual_seed(0)
        waveforms = torch.randn(1, 4000)
        self._check(waveforms)

    def test_different_mel_bins(self) -> None:
        torch.manual_seed(99)
        waveforms = torch.randn(2, 32000)
        self._check(waveforms, num_mel_bins=64)

    def test_256_mel_bins(self) -> None:
        torch.manual_seed(99)
        waveforms = torch.randn(2, 32000)
        self._check(waveforms, num_mel_bins=256)

    def test_frame_count_matches(self) -> None:
        """Verify the number of frames matches kaldi's snip_edges=True formula."""
        for n_samples in [4000, 8000, 16000, 32000, 160000]:
            waveforms = torch.zeros(1, n_samples)
            fbank_module = _BatchedFbank()
            out = fbank_module(waveforms)
            expected_frames = 1 + (n_samples - 400) // 160
            assert out.shape[1] == expected_frames, (
                f"n_samples={n_samples}: got {out.shape[1]}, expected {expected_frames}"
            )

    def test_gpu_matches_cpu(self) -> None:
        if not torch.cuda.is_available():
            return
        torch.manual_seed(42)
        waveforms = torch.randn(4, 16000)
        scaled = waveforms * 2**15

        fbank_cpu = _BatchedFbank()
        out_cpu = fbank_cpu(scaled)

        fbank_gpu = _BatchedFbank().cuda()
        out_gpu = fbank_gpu(scaled.cuda())

        torch.testing.assert_close(out_cpu, out_gpu.cpu(), atol=1e-4, rtol=1e-4)

    def test_gpu_vs_kaldi_ten_second(self) -> None:
        """GPU _BatchedFbank vs CPU ta_kaldi.fbank on long clips.

        GPU and CPU FFT/matmul use different float32 reduction order, causing
        up to ~1e-2 max_diff in log-mel space depending on input content.
        After BEATs normalization (÷13.1), this is <1e-3 — completely negligible.
        The algorithm is verified exact on CPU (atol=1e-4 tests above).
        """
        if not torch.cuda.is_available():
            return
        torch.manual_seed(123)
        waveforms = torch.randn(8, 160_000)
        scaled = waveforms * 2**15

        ref = _kaldi_fbank_loop(scaled)

        fbank_gpu = _BatchedFbank().cuda()
        gpu_out = fbank_gpu(scaled.cuda()).cpu()

        assert ref.shape == gpu_out.shape
        torch.testing.assert_close(ref, gpu_out, atol=2e-2, rtol=1e-3)

    def test_end_to_end_preprocess(self) -> None:
        """Verify the full BEATs.preprocess path matches the old kaldi loop."""
        from avex.models.beats.beats import BEATs, BEATsConfig

        cfg = BEATsConfig(finetuned_model=False)
        model = BEATs(cfg)
        model.eval()

        torch.manual_seed(42)
        waveforms = torch.randn(2, 16000)

        new_fbank = model.preprocess(waveforms)

        scaled = waveforms * 2**15
        old_fbank = _kaldi_fbank_loop(scaled)
        old_fbank = (old_fbank - 15.41663) / (2 * 6.55582)

        torch.testing.assert_close(new_fbank, old_fbank, atol=1e-4, rtol=1e-4)
