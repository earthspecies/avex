"""Patch-level padding mask utilities for EAT model."""

import torch


class PatchPaddingHandler:
    """Handles patch-level padding masks for EAT spectrograms."""

    def __init__(
        self, patch_size: int = 16, hop_length: int = 160, threshold: float = 0.5
    ):
        """
        Args:
            patch_size: Size of patches in spectrogram (typically 16x16)
            hop_length: Hop length used in spectrogram computation
            threshold: Fraction of patch that must be real audio to be considered valid
        """
        self.patch_size = patch_size
        self.hop_length = hop_length
        self.threshold = threshold

    def compute_patch_mask(
        self,
        original_lengths: torch.Tensor,  # (B,) - original audio lengths in samples
        target_frames: int,  # target spectrogram length (e.g. 1024)
        n_mels: int = 128,  # number of mel bins
    ) -> torch.Tensor:
        """
        Convert audio sample lengths to patch-level validity masks.

        Args:
            original_lengths: Original audio lengths in samples before padding
            target_frames: Target number of frames in spectrogram
            n_mels: Number of mel frequency bins

        Returns:
            patch_mask: (B, num_patches) bool tensor - True = valid patch, False = padded
        """
        B = original_lengths.size(0)
        device = original_lengths.device

        # Convert audio samples to frame indices
        valid_frames = (original_lengths.float() / self.hop_length).ceil().long()

        # Calculate patch grid dimensions
        patches_per_row = target_frames // self.patch_size  # time dimension
        patches_per_col = n_mels // self.patch_size  # frequency dimension
        total_patches = patches_per_row * patches_per_col

        # Initialize all patches as valid
        patch_mask = torch.ones(B, total_patches, dtype=torch.bool, device=device)

        for batch_idx in range(B):
            valid_frame_count = valid_frames[batch_idx].item()

            # Find last fully valid patch in time dimension
            valid_patch_count = valid_frame_count // self.patch_size

            # Handle boundary patch
            if valid_frame_count % self.patch_size > 0:
                real_frames_in_boundary = valid_frame_count % self.patch_size
                if real_frames_in_boundary > 0:
                    valid_patch_count += 1

            # Mask patches that are entirely or mostly padding
            for time_patch in range(patches_per_row):
                if time_patch >= valid_patch_count:
                    # Mark all frequency patches in this time slice as invalid
                    start_idx = time_patch * patches_per_col
                    end_idx = (time_patch + 1) * patches_per_col
                    patch_mask[batch_idx, start_idx:end_idx] = False

        return patch_mask

    def apply_loss_mask(
        self,
        loss_tensor: torch.Tensor,
        patch_mask: torch.Tensor,
        masked_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply patch padding mask to loss computation.

        Args:
            loss_tensor: Per-element loss tensor
            patch_mask: (B, num_patches) validity mask
            masked_positions: (B, num_patches) mask indicating which patches were selected for masking

        Returns:
            Masked loss tensor with padded regions zeroed out
        """
        if patch_mask is None:
            return loss_tensor

        # Expand patch mask to match masked positions
        # Only apply to positions that were both masked AND valid
        valid_masked = patch_mask & masked_positions.bool()
        valid_masked_flat = valid_masked[masked_positions.bool()]

        # Zero out loss for invalid patches
        masked_loss = loss_tensor * valid_masked_flat.float()

        return masked_loss
