"""Band scoring modules for multiband audio processing.

This module provides methods to compute informativeness scores for each
frequency band, which can be used for band selection or hybrid fusion.

Scoring methods:
- Entropy: Higher entropy = more uniform spectral distribution = more info
- Flux: Higher flux = more temporal change = more dynamic content
- GMM: Fit Gaussian mixture to spectrum, score bands by mixture weights
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class BandScores:
    """Container for per-band scores."""

    entropy: Optional[torch.Tensor] = None  # (N, num_bands)
    flux: Optional[torch.Tensor] = None  # (N, num_bands)
    gmm: Optional[torch.Tensor] = None  # (N, num_bands)

    def to_tensor(self) -> torch.Tensor:
        """Stack available scores into (N, num_bands, num_scores) tensor."""
        scores = []
        if self.entropy is not None:
            scores.append(self.entropy)
        if self.flux is not None:
            scores.append(self.flux)
        if self.gmm is not None:
            scores.append(self.gmm)
        if not scores:
            raise ValueError("No scores available")
        return torch.stack(scores, dim=-1)

    @property
    def num_score_types(self) -> int:
        """Number of score types available."""
        count = 0
        if self.entropy is not None:
            count += 1
        if self.flux is not None:
            count += 1
        if self.gmm is not None:
            count += 1
        return count


class BandScorer(nn.Module):
    """Computes informativeness scores for frequency bands.

    Given a spectrogram, computes entropy and/or flux scores for each
    frequency band. These scores can be used for:
    - Band selection (pick top-k most informative bands)
    - Hybrid fusion (incorporate scores into learned fusion)

    Parameters
    ----------
    sample_rate : int
        Audio sample rate (for computing frequency bins)
    band_width_hz : int
        Width of each frequency band
    step_hz : int, optional
        Step size between bands. Defaults to band_width_hz
    score_types : list of str
        Which scores to compute: ["entropy", "flux", "gmm"]
    n_mels : int
        Number of mel bins in input spectrogram
    """

    def __init__(
        self,
        sample_rate: int,
        band_width_hz: int = 8000,
        step_hz: Optional[int] = None,
        score_types: Optional[List[str]] = None,
        n_mels: int = 128,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.band_width_hz = band_width_hz
        self.step_hz = step_hz or band_width_hz
        self.score_types = score_types or ["entropy", "flux"]
        self.n_mels = n_mels

        # Compute band grid
        self.bands = self._compute_bands()

    def _compute_bands(self) -> List[Tuple[float, float]]:
        """Compute frequency bands up to Nyquist."""
        nyquist = self.sample_rate / 2.0
        bands = []
        f = 0.0
        while f < nyquist:
            f_high = min(f + self.band_width_hz, nyquist)
            bands.append((f, f_high))
            f += self.step_hz
        return bands

    @property
    def num_bands(self) -> int:
        return len(self.bands)

    def forward(self, spec: torch.Tensor) -> BandScores:
        """Compute band scores from spectrogram.

        Parameters
        ----------
        spec : torch.Tensor
            Log-power spectrogram of shape (N, F, T) or (N, C, F, T)

        Returns
        -------
        BandScores
            Container with computed scores
        """
        # Normalize to (N, F, T)
        spec = self._normalize_spec(spec)

        scores = BandScores()

        if "entropy" in self.score_types:
            scores.entropy = self._compute_entropy(spec)

        if "flux" in self.score_types:
            scores.flux = self._compute_flux(spec)

        if "gmm" in self.score_types:
            scores.gmm = self._compute_gmm(spec)

        return scores

    def _normalize_spec(self, spec: torch.Tensor) -> torch.Tensor:
        """Normalize spectrogram to (N, F, T)."""
        if spec.ndim == 4:
            # (N, C, F, T) -> average over channels
            spec = spec.mean(dim=1)
        if spec.ndim == 2:
            spec = spec.unsqueeze(0)
        return spec

    def _compute_entropy(self, spec: torch.Tensor) -> torch.Tensor:
        """Compute spectral entropy for each band.

        Higher entropy indicates more uniform spectral distribution,
        which often correlates with more information content.

        Parameters
        ----------
        spec : torch.Tensor
            Log-power spectrogram of shape (N, F, T)

        Returns
        -------
        torch.Tensor
            Entropy scores of shape (N, num_bands)
        """
        N, F, T = spec.shape
        device = spec.device
        nyquist = self.sample_rate / 2.0
        freq_per_bin = nyquist / F

        # Convert log-power to linear power
        power = spec.exp()
        power_sum_t = power.sum(dim=-1) + 1e-12  # (N, F)

        entropies = []
        for f_low, f_high in self.bands:
            bin_low = int(f_low / freq_per_bin)
            bin_high = min(int(f_high / freq_per_bin), F)

            if bin_high <= bin_low:
                entropies.append(torch.zeros(N, device=device))
                continue

            band_pow = power_sum_t[:, bin_low:bin_high]  # (N, bins)

            # Normalize to probability distribution
            band_prob = band_pow / (band_pow.sum(dim=-1, keepdim=True) + 1e-12)

            # Shannon entropy
            entropy = -(band_prob * (band_prob + 1e-12).log()).sum(dim=-1)
            entropies.append(entropy)

        return torch.stack(entropies, dim=1)  # (N, num_bands)

    def _compute_flux(self, spec: torch.Tensor) -> torch.Tensor:
        """Compute spectral flux for each band.

        Higher flux indicates more temporal change, which often
        correlates with dynamic content (e.g., vocalizations).

        Parameters
        ----------
        spec : torch.Tensor
            Log-power spectrogram of shape (N, F, T)

        Returns
        -------
        torch.Tensor
            Flux scores of shape (N, num_bands)
        """
        N, F, T = spec.shape
        device = spec.device
        nyquist = self.sample_rate / 2.0
        freq_per_bin = nyquist / F

        # Convert to power and compute temporal difference
        power = spec.exp()
        diff = power[:, :, 1:] - power[:, :, :-1]
        pos_diff = torch.clamp(diff, min=0.0)  # Only positive changes
        flux_per_freq = pos_diff.sum(dim=-1)  # (N, F)

        fluxes = []
        for f_low, f_high in self.bands:
            bin_low = int(f_low / freq_per_bin)
            bin_high = min(int(f_high / freq_per_bin), F)

            if bin_high <= bin_low or T < 2:
                fluxes.append(torch.zeros(N, device=device))
                continue

            band_flux = flux_per_freq[:, bin_low:bin_high].sum(dim=-1)
            fluxes.append(band_flux)

        return torch.stack(fluxes, dim=1)  # (N, num_bands)

    def _compute_gmm(self, spec: torch.Tensor) -> torch.Tensor:
        """Compute GMM-based scores for each band.

        Fits a Gaussian mixture model to the mean spectrum and scores
        bands based on mixture weights near their center frequencies.

        Parameters
        ----------
        spec : torch.Tensor
            Log-power spectrogram of shape (N, F, T)

        Returns
        -------
        torch.Tensor
            GMM scores of shape (N, num_bands)
        """
        try:
            from sklearn.mixture import GaussianMixture
        except ImportError:
            raise ImportError("GMM scoring requires scikit-learn: pip install scikit-learn")

        N, F, T = spec.shape
        device = spec.device
        nyquist = self.sample_rate / 2.0

        # Frequency bins
        freqs = torch.linspace(0, nyquist, F).cpu().numpy().reshape(-1, 1)

        all_scores = []
        for i in range(N):
            # Mean spectrum for this sample
            power = spec[i].exp()
            mean_spec = power.mean(dim=-1)  # (F,)
            weights = (mean_spec / (mean_spec.sum() + 1e-12)).cpu().numpy()

            # Fit GMM
            gmm = GaussianMixture(
                n_components=3,
                covariance_type="full",
                random_state=0,
            )
            try:
                gmm.fit(freqs, sample_weight=weights)
            except TypeError:
                gmm.fit(freqs)

            # Score each band by mixture weights
            band_scores = torch.zeros(len(self.bands), device=device)
            means = gmm.means_.flatten()

            for k in range(gmm.n_components):
                center = means[k]
                w_k = gmm.weights_[k]

                # Find nearest band
                dists = [abs((b[0] + b[1]) / 2 - center) for b in self.bands]
                idx = int(torch.tensor(dists).argmin().item())
                band_scores[idx] += float(w_k)

            all_scores.append(band_scores)

        return torch.stack(all_scores, dim=0)  # (N, num_bands)

    def select_top_k(
        self, spec: torch.Tensor, k: int = 1, method: str = "entropy"
    ) -> List[Tuple[float, float]]:
        """Select top-k most informative bands.

        Parameters
        ----------
        spec : torch.Tensor
            Log-power spectrogram
        k : int
            Number of bands to select
        method : str
            Scoring method: "entropy", "flux", or "gmm"

        Returns
        -------
        list of tuple
            Top-k bands as (f_low, f_high) tuples, sorted best to worst
        """
        scores = self.forward(spec)

        if method == "entropy":
            score_tensor = scores.entropy
        elif method == "flux":
            score_tensor = scores.flux
        elif method == "gmm":
            score_tensor = scores.gmm
        else:
            raise ValueError(f"Unknown method: {method}")

        if score_tensor is None:
            raise ValueError(f"Score type '{method}' not computed")

        # Average over batch
        mean_scores = score_tensor.mean(dim=0)  # (num_bands,)

        k = min(k, len(self.bands))
        topk_idx = torch.topk(mean_scores, k).indices.tolist()

        return [self.bands[i] for i in topk_idx]

    def get_band_info(self) -> List[Tuple[float, float]]:
        """Return frequency ranges for each band."""
        return self.bands.copy()
