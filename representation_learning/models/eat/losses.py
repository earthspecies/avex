from __future__ import annotations

"""Self-supervised losses used in the original EAT pre-training.

The goal is *functional parity* without strict Fairseq dependencies.  We
therefore re-implement the three constituent losses and a convenience wrapper
`PretrainingLoss` that combines them using the same weighting scheme as in the
paper / upstream code.

Notation (matching upstream):
• *s* – student features  (B, T, D)
• *t* – teacher features  (B, T, D)
• *r* – decoder reconstruction (B, T, C)
• *x* – local (unmasked) features (B, T, C)
• *c* – utterance-level classifier prediction (B, D)
• *y* – utterance labels (B, D) – typically one-hot speaker ids (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Individual losses                                                          #
# --------------------------------------------------------------------------- #

def _safe_var(t: torch.Tensor) -> torch.Tensor:
    t = t.view(-1, t.size(-1))
    return torch.sqrt(t.var(dim=0, unbiased=False) + 1e-6).mean()


class Data2VecLoss(nn.Module):
    """Student-teacher regression loss (same as in wav2vec-2.0 / data2vec).

    We minimise the cosine distance between *s* and *t* plus variance
    regularisers as in DINO.
    """

    def __init__(self, beta: float = 0.0, scale: float | None = None) -> None:
        super().__init__()
        self.beta = beta  # teacher-centering term – kept for future use
        self.scale = scale

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:  # noqa: D401
        # Normalise
        s = F.normalize(student, dim=-1)
        t = F.normalize(teacher.detach(), dim=-1)

        if self.scale is None:
            scale = 1.0 / (s.size(-1) ** 0.5)
        else:
            scale = self.scale

        cos_sim = (s * t).sum(-1) * scale  # (B, T)
        loss = 2 - 2 * cos_sim.mean()  # minimise (1 − cosine)

        # variance regularisation (avoid collapse)
        loss += (_safe_var(student) + _safe_var(teacher)) * 0.01
        return loss


class ReconstructionLoss(nn.Module):
    """L2 loss between decoder output and local features."""

    def __init__(self):
        super().__init__()

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return F.mse_loss(recon, target)


class CLSLoss(nn.Module):
    """Simple cross-entropy for utterance-level classification."""

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return F.cross_entropy(logits, labels)


# --------------------------------------------------------------------------- #
#  Aggregation wrapper                                                        #
# --------------------------------------------------------------------------- #

class PretrainingLoss(nn.Module):
    """Aggregate EAT pre-training losses with configurable weights."""

    def __init__(
        self,
        d2v_weight: float = 1.0,
        recon_weight: float = 0.0,
        cls_weight: float = 0.0,
        beta: float = 0.0,
        scale: float | None = None,
    ) -> None:
        super().__init__()
        self.d2v_w = d2v_weight
        self.recon_w = recon_weight
        self.cls_w = cls_weight

        self.d2v_loss = Data2VecLoss(beta=beta, scale=scale)
        self.recon_loss = ReconstructionLoss() if recon_weight > 0 else None
        self.cls_loss = CLSLoss() if cls_weight > 0 else None

    def forward(
        self,
        *,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
        decoder_out: torch.Tensor | None = None,
        local_feat: torch.Tensor | None = None,
        clf_logits: torch.Tensor | None = None,
        clf_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:  # noqa: D401
        losses: dict[str, torch.Tensor] = {}

        l_d2v = self.d2v_loss(student_feat, teacher_feat)
        losses["d2v_loss"] = l_d2v * self.d2v_w

        if self.recon_w and decoder_out is not None and local_feat is not None:
            l_r = self.recon_loss(decoder_out, local_feat)
            losses["recon_loss"] = l_r * self.recon_w

        if self.cls_w and clf_logits is not None and clf_labels is not None:
            l_c = self.cls_loss(clf_logits, clf_labels)
            losses["cls_loss"] = l_c * self.cls_w

        losses["total"] = sum(losses.values())
        return losses


# --------------------------------------------------------------------------- #
#  Functional wrappers expected by eat.py                                     #
# --------------------------------------------------------------------------- #

def d2v_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    beta: float = 0.0,
    scale: float | None = None,
) -> torch.Tensor:
    """Convenience wrapper around *Data2VecLoss* so callers can use a functional
    interface.  Mirrors the signature used in the original Fairseq code.

    Parameters
    ----------
    pred : torch.Tensor
        Student predictions (B, T, D)
    target : torch.Tensor
        Teacher / ground-truth features (B, T, D)
    beta : float, optional
        Centering term (currently unused but kept for API compatibility)
    scale : float | None, optional
        Pre-computed "1/sqrt(D)" scaling factor.  If ``None`` the loss will
        compute it internally.
    """

    return Data2VecLoss(beta=beta, scale=scale)(pred, target)


def dino_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    t_student: float = 0.1,
    t_teacher: float = 0.05,
) -> torch.Tensor:
    """DINO-style cross-entropy between *student* and *teacher* outputs.

    This simplified variant is sufficient for the spectrogram-only use-case in
    EAT and avoids pulling in the full facebookresearch/dino code.
    """

    # Teacher probabilities are detached to stop gradients
    teacher_probs = F.softmax(teacher_logits / t_teacher, dim=-1).detach()
    student_log_probs = F.log_softmax(student_logits / t_student, dim=-1)
    loss = -(teacher_probs * student_log_probs).sum(dim=-1).mean()
    return loss
