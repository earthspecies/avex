from __future__ import annotations

import torch

from representation_learning.models.eat.losses import PretrainingLoss


def test_pretraining_loss_shapes() -> None:
    B, T, D = 2, 4, 8
    student = torch.randn(B, T, D)
    teacher = torch.randn(B, T, D)
    decoder_out = torch.randn(B, T, D)
    local = torch.randn(B, T, D)
    logits = torch.randn(B, 5)
    labels = torch.randint(0, 5, (B,))

    crit = PretrainingLoss(d2v_weight=1.0, recon_weight=0.5, cls_weight=0.3)
    out = crit(
        student_feat=student,
        teacher_feat=teacher,
        decoder_out=decoder_out,
        local_feat=local,
        clf_logits=logits,
        clf_labels=labels,
    )

    assert "total" in out and out["total"].ndim == 0
    assert out["d2v_loss"].ndim == 0
    assert out["recon_loss"].ndim == 0
    assert out["cls_loss"].ndim == 0
