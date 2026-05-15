"""Unit tests for AudioProtoPNet SED (frame-level) model.

All tests use ``pretrained=False`` — no checkpoint files required.

Coverage:
  - Model creation and attribute invariants
  - ``forward()`` — clip-level logits [B, C]
  - ``forward_frames()`` — per-frame probabilities [B, T, C]
  - Frame output properties: sigmoid range, time dimension scaling
  - Clip/frame consistency and pooling commutativity
  - Batch independence
  - Layer discovery: ``backbone.encoder.stages.{0-3}``
  - Hook-based embedding extraction
  - ``_cosine_activation`` activation properties
  - ``_LinearLayerWithoutNegativeConnections`` weight non-negativity and shape
  - Registry auto-discovery
"""

import pytest
import torch
import torch.nn as nn

from avex.models.audioprotopnet_sed import (
    Model as AudioProtoPNetSEDModel,
)
from avex.models.audioprotopnet_sed import (
    _cosine_activation,
    _LinearLayerWithoutNegativeConnections,
)

_SAMPLE_RATE = 32_000
_CLIP_SECONDS = 5
_NUM_CLASSES = 20


# =========================================================================== #
#  Fixtures                                                                    #
# =========================================================================== #


@pytest.fixture(scope="module")
def model() -> AudioProtoPNetSEDModel:
    m = AudioProtoPNetSEDModel(pretrained=False, device="cpu", num_classes=_NUM_CLASSES)
    m.eval()
    return m


@pytest.fixture(scope="module")
def audio_batch() -> torch.Tensor:
    """Batch of two 5-second random waveforms at 32 kHz.

    Returns
    -------
    torch.Tensor
        Shape ``(2, sample_rate * clip_seconds)``.
    """
    return torch.randn(2, _SAMPLE_RATE * _CLIP_SECONDS)


@pytest.fixture
def registered_last_stage(model: AudioProtoPNetSEDModel) -> str:
    """Discover layers, register a hook on the last stage, clean up after.

    Yields
    ------
    str
        Name of the last encoder stage with a hook registered.
    """
    model._layer_names = []
    model._discover_linear_layers()
    stage = model._layer_names[-1]
    model.register_hooks_for_layers([stage])
    yield stage
    model.deregister_all_hooks()


# =========================================================================== #
#  Model creation                                                              #
# =========================================================================== #


def test_pretrained_false_requires_num_classes() -> None:
    with pytest.raises(ValueError, match="num_classes"):
        AudioProtoPNetSEDModel(pretrained=False, device="cpu")


def test_model_has_expected_attributes(model: AudioProtoPNetSEDModel) -> None:
    assert model.num_classes == _NUM_CLASSES
    assert hasattr(model, "backbone")
    assert hasattr(model, "add_on_layers")
    assert hasattr(model, "prototype_vectors")
    assert isinstance(model.last_layer, _LinearLayerWithoutNegativeConnections)
    assert isinstance(model.add_on_layers, nn.Upsample)


def test_prototype_vectors_shape(model: AudioProtoPNetSEDModel) -> None:
    """prototype_vectors must be 4-D [P, C, 1, 1] for spatial conv2d."""
    pv = model.prototype_vectors
    assert pv.dim() == 4
    assert pv.shape[2] == 1
    assert pv.shape[3] == 1


# =========================================================================== #
#  process_audio                                                               #
# =========================================================================== #


def test_process_audio_output_shape(model: AudioProtoPNetSEDModel, audio_batch: torch.Tensor) -> None:
    mel = model.process_audio(audio_batch)
    assert mel.shape[:3] == torch.Size([2, 1, 256])  # [B, 1, n_mels, T]


# =========================================================================== #
#  forward() — clip-level                                                     #
# =========================================================================== #


def test_forward_output_shape(model: AudioProtoPNetSEDModel, audio_batch: torch.Tensor) -> None:
    with torch.no_grad():
        logits = model(audio_batch)
    assert logits.shape == (2, _NUM_CLASSES)
    assert logits.dtype in (torch.float32, torch.float16, torch.bfloat16)


def test_forward_padding_mask_ignored(model: AudioProtoPNetSEDModel, audio_batch: torch.Tensor) -> None:
    mask = torch.zeros(2, audio_batch.shape[1], dtype=torch.bool)
    with torch.no_grad():
        assert torch.allclose(model(audio_batch), model(audio_batch, padding_mask=mask))


# =========================================================================== #
#  forward_frames() — frame-level                                             #
# =========================================================================== #


def test_forward_frames_output_shape(model: AudioProtoPNetSEDModel, audio_batch: torch.Tensor) -> None:
    with torch.no_grad():
        probs = model.forward_frames(audio_batch)
    assert probs.dim() == 3
    assert probs.shape[0] == 2
    assert probs.shape[1] > 0
    assert probs.shape[2] == _NUM_CLASSES


def test_forward_frames_probabilities_in_range(model: AudioProtoPNetSEDModel, audio_batch: torch.Tensor) -> None:
    """forward_frames applies sigmoid — all values must be in [0, 1]."""
    with torch.no_grad():
        probs = model.forward_frames(audio_batch)
    assert probs.min().item() >= 0.0
    assert probs.max().item() <= 1.0


def test_forward_frames_padding_mask_ignored(model: AudioProtoPNetSEDModel, audio_batch: torch.Tensor) -> None:
    mask = torch.zeros(2, audio_batch.shape[1], dtype=torch.bool)
    with torch.no_grad():
        assert torch.allclose(
            model.forward_frames(audio_batch),
            model.forward_frames(audio_batch, padding_mask=mask),
        )


def test_forward_frames_longer_audio_more_frames(model: AudioProtoPNetSEDModel) -> None:
    """Doubling audio length should approximately double the number of frames."""
    with torch.no_grad():
        t_5s = model.forward_frames(torch.randn(1, _SAMPLE_RATE * 5)).shape[1]
        t_10s = model.forward_frames(torch.randn(1, _SAMPLE_RATE * 10)).shape[1]
    assert abs(t_10s - 2 * t_5s) <= 2, f"Expected ~{2 * t_5s} frames for 10s, got {t_10s}"


# =========================================================================== #
#  Clip / frame consistency                                                    #
# =========================================================================== #


def test_clip_and_frame_deterministic(model: AudioProtoPNetSEDModel, audio_batch: torch.Tensor) -> None:
    """Both forward paths are deterministic and share the same class dimension."""
    with torch.no_grad():
        clip1, clip2 = model(audio_batch), model(audio_batch)
        frames1, frames2 = model.forward_frames(audio_batch), model.forward_frames(audio_batch)
    assert clip1.shape[-1] == frames1.shape[-1]
    assert torch.allclose(clip1, clip2)
    assert torch.allclose(frames1, frames2)


def test_clip_forward_equals_global_max_pool_then_head(
    model: AudioProtoPNetSEDModel, audio_batch: torch.Tensor
) -> None:
    """forward() must equal global max-pool over activations → last_layer."""
    mel = model.process_audio(audio_batch)
    with torch.no_grad():
        features = model._backbone_features(mel)
        activations = _cosine_activation(features, model.prototype_vectors)
        global_max = activations.view(activations.shape[0], activations.shape[1], -1).max(dim=-1).values
        expected = model.last_layer(global_max)
        actual = model(audio_batch)
    assert torch.allclose(expected, actual, atol=1e-5)


# =========================================================================== #
#  Batch independence                                                          #
# =========================================================================== #


def test_forward_batch_independent(model: AudioProtoPNetSEDModel, audio_batch: torch.Tensor) -> None:
    with torch.no_grad():
        logits_solo = model(audio_batch[0:1])
        logits_batch = model(audio_batch)
    assert torch.allclose(logits_solo[0], logits_batch[0], atol=1e-5)


def test_forward_frames_batch_independent(model: AudioProtoPNetSEDModel, audio_batch: torch.Tensor) -> None:
    with torch.no_grad():
        probs_solo = model.forward_frames(audio_batch[0:1])
        probs_batch = model.forward_frames(audio_batch)
    assert torch.allclose(probs_solo[0], probs_batch[0], atol=1e-5)


# =========================================================================== #
#  Layer discovery                                                             #
# =========================================================================== #


def test_discover_layers(model: AudioProtoPNetSEDModel) -> None:
    model._layer_names = []
    model._discover_linear_layers()
    assert len(model._layer_names) == 4
    for name in model._layer_names:
        assert name.startswith("backbone.encoder.stages"), f"Unexpected: {name}"


def test_discover_layers_idempotent(model: AudioProtoPNetSEDModel) -> None:
    model._layer_names = []
    model._discover_linear_layers()
    first = list(model._layer_names)
    model._discover_linear_layers()
    assert model._layer_names == first


# =========================================================================== #
#  Hook-based embedding extraction                                             #
# =========================================================================== #


def test_hook_extraction_shape(
    model: AudioProtoPNetSEDModel,
    registered_last_stage: str,
    audio_batch: torch.Tensor,
) -> None:
    with torch.no_grad():
        model(audio_batch)
    assert registered_last_stage in model._hook_outputs
    assert model._hook_outputs[registered_last_stage].shape[0] == 2


def test_extract_embeddings_aggregation_mean(
    model: AudioProtoPNetSEDModel,
    registered_last_stage: str,
    audio_batch: torch.Tensor,
) -> None:
    with torch.no_grad():
        emb = model.extract_embeddings(audio_batch, aggregation="mean")
    assert emb.dim() == 2
    assert emb.shape[0] == 2


def test_extract_embeddings_no_aggregation(
    model: AudioProtoPNetSEDModel,
    registered_last_stage: str,
    audio_batch: torch.Tensor,
) -> None:
    with torch.no_grad():
        emb = model.extract_embeddings(audio_batch, aggregation="none")
    assert emb.dim() == 4
    assert emb.shape[0] == 2


# =========================================================================== #
#  _cosine_activation                                                          #
# =========================================================================== #


def test_cosine_activation_output_shape() -> None:
    B, C, H, W, P = 2, 16, 8, 20, 10
    out = _cosine_activation(torch.randn(B, C, H, W), torch.randn(P, C, 1, 1))
    assert out.shape == (B, P, H, W)


def test_cosine_activation_non_negative() -> None:
    out = _cosine_activation(torch.randn(2, 16, 8, 10), torch.randn(5, 16, 1, 1))
    assert out.min().item() >= 0.0


def test_cosine_activation_bounded() -> None:
    out = _cosine_activation(torch.randn(4, 32, 6, 6), torch.randn(8, 32, 1, 1))
    assert out.max().item() <= 1.02


def test_cosine_activation_uniform_features_uniform_output() -> None:
    """Uniform feature map → all spatial positions have identical activation."""
    C = 16
    val = torch.randn(C)
    features = val.view(1, C, 1, 1).expand(1, C, 4, 4).contiguous()
    out = _cosine_activation(features, val.view(1, C, 1, 1))
    assert out.std().item() < 1e-4


# =========================================================================== #
#  _LinearLayerWithoutNegativeConnections                                     #
# =========================================================================== #


def test_linear_layer_forward_shape() -> None:
    layer = _LinearLayerWithoutNegativeConnections(in_features=20, out_features=4)
    assert layer(torch.randn(3, 20)).shape == (3, 4)
    assert layer(torch.randn(2, 38, 20)).shape == (2, 38, 4)  # 3-D [B, T, F]


def test_linear_layer_effective_weights_non_negative() -> None:
    layer = _LinearLayerWithoutNegativeConnections(in_features=20, out_features=4)
    with torch.no_grad():
        layer.weight.fill_(-1.0)
    out = layer(torch.eye(20))
    assert torch.allclose(out, layer.bias.unsqueeze(0).expand(20, -1))


def test_linear_layer_requires_divisible_features() -> None:
    with pytest.raises(ValueError, match="divisible"):
        _LinearLayerWithoutNegativeConnections(in_features=10, out_features=3)


def test_linear_layer_block_diagonal_structure() -> None:
    K, C = 3, 4
    layer = _LinearLayerWithoutNegativeConnections(in_features=K * C, out_features=C, bias=False)
    with torch.no_grad():
        layer.weight.fill_(1.0)
    x = torch.zeros(1, K * C)
    x[0, :K] = 1.0
    out = layer(x)
    assert out[0, 0].item() > 0.0
    assert out[0, 1:].sum().item() == 0.0


# =========================================================================== #
#  convnext_cfg                                                                #
# =========================================================================== #


def test_model_mel_shape_reflects_convnext_cfg() -> None:
    """convnext_cfg overrides n_mels for process_audio output."""
    m = AudioProtoPNetSEDModel(
        pretrained=False,
        device="cpu",
        num_classes=4,
        convnext_cfg={
            "sample_rate": 32000,
            "n_fft": 2048,
            "hop_length": 256,
            "n_mels": 128,
            "norm_mean": 0.0,
            "norm_std": 1.0,
        },
    )
    m.eval()
    with torch.no_grad():
        mel = m.process_audio(torch.randn(1, 32000 * 5))
    assert mel.shape[2] == 128


# =========================================================================== #
#  Registry                                                                    #
# =========================================================================== #


def test_model_is_registered() -> None:
    from avex.models.utils.registry import get_model_class

    assert get_model_class("audioprotopnet_sed") is AudioProtoPNetSEDModel
