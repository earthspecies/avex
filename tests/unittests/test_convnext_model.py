"""Unit tests for the ConvNeXt audio encoder wrapper (``avex.models.convnext``).

All tests run offline with ``pretrained=False`` and a tiny architecture; the
BirdCODE weights themselves are covered by the official-model checksum and
output-regression suites.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from avex.configs import AudioConfig, ModelSpec
from avex.models.convnext import ConvNextInitConfig, Model
from avex.models.utils.factory import build_model_from_spec
from avex.models.utils.load import _load_checkpoint, load_model
from avex.models.utils.registry import get_model_class, get_model_spec, initialize_registry

_TINY_ARCH = {"depths": [1, 1, 1, 1], "hidden_sizes": [8, 16, 32, 64]}
_SR = 32_000
_SECONDS = 5


@pytest.fixture(autouse=True)
def _ensure_registry_initialized() -> None:
    """Other test modules clear the global registries; restore them before each test."""
    initialize_registry()


@pytest.fixture(scope="module")
def tiny_encoder() -> Model:
    """Randomly initialised, feature-only tiny ConvNeXt on CPU.

    Returns
    -------
    Model
        Model in eval mode.
    """
    m = Model(pretrained=False, device="cpu", init_config=_TINY_ARCH)
    m.eval()
    return m


@pytest.fixture(scope="module")
def audio() -> torch.Tensor:
    """Two 5-second random waveforms at 32 kHz.

    Returns
    -------
    torch.Tensor
        Shape ``(2, 160000)``.
    """
    torch.manual_seed(0)
    return torch.randn(2, _SR * _SECONDS)


# --------------------------------------------------------------------------- #
#  Config                                                                       #
# --------------------------------------------------------------------------- #


def test_init_config_defaults_match_birdset_setup() -> None:
    cfg = ConvNextInitConfig.from_mapping(None)
    assert cfg.depths == (3, 3, 27, 3)
    assert cfg.hidden_sizes == (128, 256, 512, 1024)
    assert (cfg.sample_rate, cfg.n_fft, cfg.hop_length, cfg.n_mels) == (32_000, 2048, 256, 256)
    assert cfg.norm_mean == pytest.approx(-13.369)
    assert cfg.norm_std == pytest.approx(13.162)


def test_init_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="Unknown convnext init_config keys"):
        ConvNextInitConfig.from_mapping({"n_mels": 64, "bogus": 1})


def test_init_config_passthrough(tiny_encoder: Model) -> None:
    assert tiny_encoder.init_config.depths == (1, 1, 1, 1)
    assert tiny_encoder.init_config.hidden_sizes == (8, 16, 32, 64)
    assert tuple(tiny_encoder.backbone.config.hidden_sizes) == (8, 16, 32, 64)


# --------------------------------------------------------------------------- #
#  Forward                                                                      #
# --------------------------------------------------------------------------- #


def test_process_audio_shape(tiny_encoder: Model, audio: torch.Tensor) -> None:
    mel = tiny_encoder.process_audio(audio)
    # 160000 samples / hop 256 -> 626 centred frames
    assert mel.shape == (2, 1, 256, 626)
    assert mel.dtype == torch.float32


def test_process_audio_accepts_float64_and_1d(tiny_encoder: Model) -> None:
    mel = tiny_encoder.process_audio(torch.randn(_SR, dtype=torch.float64))
    assert mel.shape[:2] == (1, 1)
    assert mel.dtype == torch.float32


def test_feature_mode_forward_shape(tiny_encoder: Model, audio: torch.Tensor) -> None:
    assert tiny_encoder.return_features_only is True
    assert tiny_encoder.classifier is None
    with torch.no_grad():
        out = tiny_encoder(audio)
    # stride 32 on both axes: 256 mel -> 8, 626 frames -> 19
    assert out.shape == (2, 64, 8, 19)


def test_padding_mask_is_ignored(tiny_encoder: Model, audio: torch.Tensor) -> None:
    mask = torch.zeros(audio.shape, dtype=torch.bool)
    with torch.no_grad():
        assert torch.equal(tiny_encoder(audio), tiny_encoder(audio, mask))


def test_classifier_mode_forward_shape(audio: torch.Tensor) -> None:
    m = Model(num_classes=7, pretrained=False, device="cpu", init_config=_TINY_ARCH)
    m.eval()
    assert m.num_classes == 7
    assert m.classifier is not None and m.classifier.out_features == 7
    with torch.no_grad():
        assert m(audio).shape == (2, 7)


def test_num_classes_none_falls_back_to_feature_mode() -> None:
    m = Model(num_classes=None, pretrained=False, device="cpu", init_config=_TINY_ARCH, return_features_only=False)
    assert m.return_features_only is True
    assert m.num_classes is None


def test_short_input_still_produces_a_time_frame(tiny_encoder: Model) -> None:
    """The output regression suite feeds 1 s at 16 kHz; the stride-32 encoder must cope."""
    with torch.no_grad():
        out = tiny_encoder(torch.randn(6, 16_000))
    assert out.shape == (6, 64, 8, 1)


def test_gradient_checkpointing_matches_plain_forward(audio: torch.Tensor) -> None:
    torch.manual_seed(1)
    m = Model(pretrained=False, device="cpu", init_config=_TINY_ARCH)
    m.train()
    plain = m(audio)
    m.enable_gradient_checkpointing()
    assert m.gradient_checkpointing is True
    checkpointed = m(audio)
    assert torch.allclose(plain, checkpointed)
    checkpointed.sum().backward()
    assert m.backbone.embeddings.patch_embeddings.weight.grad is not None


# --------------------------------------------------------------------------- #
#  Hooks and embeddings                                                         #
# --------------------------------------------------------------------------- #


def test_layer_discovery_lists_four_stages(tiny_encoder: Model) -> None:
    tiny_encoder.register_hooks_for_layers(["all"])
    assert tiny_encoder.get_model_layers() == [f"backbone.encoder.stages.{i}" for i in range(4)]
    tiny_encoder.deregister_all_hooks()


def test_last_layer_resolves_to_last_stage(tiny_encoder: Model, audio: torch.Tensor) -> None:
    resolved = tiny_encoder.register_hooks_for_layers(["last_layer"])
    assert resolved == ["backbone.encoder.stages.3"]
    emb = tiny_encoder.extract_embeddings(audio, aggregation="none")
    assert emb.shape == (2, 64, 8, 19)
    tiny_encoder.deregister_all_hooks()


def test_extract_embeddings_aggregation(tiny_encoder: Model, audio: torch.Tensor) -> None:
    tiny_encoder.register_hooks_for_layers(["all"])
    try:
        raw = tiny_encoder.extract_embeddings(audio, aggregation="none")
        assert isinstance(raw, list) and len(raw) == 4
        assert [t.shape[1] for t in raw] == [8, 16, 32, 64]

        mean = tiny_encoder.extract_embeddings(audio, aggregation="mean")
        # each stage reduced over time and flattened to channels * freq
        freq = [64, 32, 16, 8]
        assert mean.shape == (2, sum(c * f for c, f in zip([8, 16, 32, 64], freq, strict=True)))

        mx = tiny_encoder.extract_embeddings(audio, aggregation="max")
        assert mx.shape == mean.shape

        with pytest.raises(ValueError, match="Unsupported aggregation"):
            tiny_encoder.extract_embeddings(audio, aggregation="median")
    finally:
        tiny_encoder.deregister_all_hooks()


def test_extract_embeddings_without_hooks_raises(audio: torch.Tensor) -> None:
    m = Model(pretrained=False, device="cpu", init_config=_TINY_ARCH)
    with pytest.raises(ValueError, match="No hooks registered"):
        m.extract_embeddings(audio)


def test_extract_embeddings_accepts_dict_input(tiny_encoder: Model, audio: torch.Tensor) -> None:
    tiny_encoder.register_hooks_for_layers(["last_layer"])
    try:
        emb = tiny_encoder.extract_embeddings({"raw_wav": audio}, aggregation="mean")
        assert emb.shape == (2, 64 * 8)
    finally:
        tiny_encoder.deregister_all_hooks()


# --------------------------------------------------------------------------- #
#  Checkpoints                                                                  #
# --------------------------------------------------------------------------- #


def test_state_dict_layout_matches_published_checkpoint(tiny_encoder: Model) -> None:
    keys = set(tiny_encoder.state_dict())
    assert "mel_transform.spectrogram.window" in keys
    assert "mel_transform.mel_scale.fb" in keys
    assert all(k.startswith(("backbone.", "mel_transform.")) for k in keys)
    assert not any("classifier" in k for k in keys)


def test_load_checkpoint_roundtrip(tmp_path: Path, audio: torch.Tensor) -> None:
    """A backbone-only checkpoint with ``backbone.*`` keys loads through the public loader."""
    torch.manual_seed(2)
    src = Model(pretrained=False, device="cpu", init_config=_TINY_ARCH).eval()
    ckpt = tmp_path / "encoder.pt"
    torch.save({"model_state_dict": src.state_dict()}, ckpt)

    torch.manual_seed(3)
    dst = Model(pretrained=False, device="cpu", init_config=_TINY_ARCH).eval()
    with torch.no_grad():
        assert not torch.allclose(src(audio), dst(audio))

    _load_checkpoint(dst, str(ckpt), device="cpu", keep_classifier=False)
    with torch.no_grad():
        assert torch.equal(src(audio), dst(audio))


def _as_frame_detector_state_dict(model: Model, num_classes: int = 3, protos_per_class: int = 2) -> dict:
    """Wrap a wrapper state dict in the sound-event-detection ``FrameDetector`` layout.

    Returns
    -------
    dict
        ``encoder.backbone.*`` / ``encoder.mel_transform.*`` keys plus a prototype head.
    """
    sd = {}
    for k, v in model.state_dict().items():
        if k.startswith("backbone."):
            sd["encoder.backbone." + k[len("backbone.") :]] = v.clone()
        elif k.startswith("mel_transform."):
            sd["encoder.mel_transform." + k[len("mel_transform.") :]] = v.clone()
    hidden = model.backbone.config.hidden_sizes[-1]
    sd["encoder.prototype_vectors"] = torch.randn(num_classes * protos_per_class, hidden, 1, 1)
    sd["classifier.weight"] = torch.randn(num_classes, protos_per_class)
    sd["classifier.bias"] = torch.randn(num_classes)
    return sd


def test_load_state_dict_accepts_birdcode_layout(audio: torch.Tensor) -> None:
    torch.manual_seed(5)
    src = Model(pretrained=False, device="cpu", init_config=_TINY_ARCH).eval()
    detector_sd = _as_frame_detector_state_dict(src)

    torch.manual_seed(6)
    dst = Model(pretrained=False, device="cpu", init_config=_TINY_ARCH).eval()
    result = dst.load_state_dict(detector_sd, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    with torch.no_grad():
        assert torch.equal(src(audio), dst(audio))


def test_load_state_dict_handles_loader_added_backbone_prefix(audio: torch.Tensor) -> None:
    """``_load_checkpoint`` prefixes every key with ``backbone.`` when none carries it."""
    torch.manual_seed(7)
    src = Model(pretrained=False, device="cpu", init_config=_TINY_ARCH).eval()
    prefixed = {"backbone." + k: v for k, v in _as_frame_detector_state_dict(src).items()}

    dst = Model(pretrained=False, device="cpu", init_config=_TINY_ARCH).eval()
    result = dst.load_state_dict(prefixed, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    with torch.no_grad():
        assert torch.equal(src(audio), dst(audio))


def test_prototype_head_checkpoint_switches_classifier_model_to_feature_mode(audio: torch.Tensor) -> None:
    src = Model(pretrained=False, device="cpu", init_config=_TINY_ARCH).eval()
    detector_sd = _as_frame_detector_state_dict(src, num_classes=3)

    clf = Model(num_classes=3, pretrained=False, device="cpu", init_config=_TINY_ARCH).eval()
    assert clf.classifier is not None
    clf.load_state_dict(detector_sd, strict=True)
    assert clf.classifier is None
    assert clf.return_features_only is True
    assert clf.num_classes is None
    with torch.no_grad():
        assert torch.equal(src(audio), clf(audio))


def test_load_checkpoint_with_birdcode_layout_through_public_loader(tmp_path: Path, audio: torch.Tensor) -> None:
    """End-to-end: a FrameDetector-style .pt goes through ``_load_checkpoint`` unchanged."""
    torch.manual_seed(8)
    src = Model(pretrained=False, device="cpu", init_config=_TINY_ARCH).eval()
    ckpt = tmp_path / "best_model.pt"
    torch.save(_as_frame_detector_state_dict(src), ckpt)

    dst = Model(pretrained=False, device="cpu", init_config=_TINY_ARCH).eval()
    _load_checkpoint(dst, str(ckpt), device="cpu", keep_classifier=False)
    with torch.no_grad():
        assert torch.equal(src(audio), dst(audio))


def test_load_model_from_spec_with_backbone_checkpoint(tmp_path: Path, audio: torch.Tensor) -> None:
    """``load_model`` must auto-select embedding mode for a head-less checkpoint."""
    torch.manual_seed(4)
    src = Model(pretrained=False, device="cpu", init_config=_TINY_ARCH).eval()
    ckpt = tmp_path / "encoder.pt"
    torch.save({"model_state_dict": src.state_dict()}, ckpt)

    spec = ModelSpec(
        name="convnext",
        pretrained=False,
        device="cpu",
        init_config=_TINY_ARCH,
        audio_config=AudioConfig(sample_rate=_SR, representation="raw", normalize=False, target_length_seconds=5),
    )
    loaded = load_model(spec, device="cpu", checkpoint_path=str(ckpt))
    loaded.eval()
    assert isinstance(loaded, Model)
    assert loaded.return_features_only is True
    with torch.no_grad():
        assert torch.equal(src(audio), loaded(audio))


# --------------------------------------------------------------------------- #
#  Registry                                                                     #
# --------------------------------------------------------------------------- #


def test_registry_discovers_convnext_class() -> None:
    initialize_registry()
    assert get_model_class("convnext") is Model


def test_official_birdcode_encoder_spec_is_registered() -> None:
    initialize_registry()
    spec = get_model_spec("esp_aves2_sed_birdcode_convnext_encoder")
    assert spec is not None
    assert spec.name == "convnext"
    assert spec.pretrained is False
    assert spec.audio_config is not None and spec.audio_config.sample_rate == 32_000
    cfg = ConvNextInitConfig.from_mapping(spec.init_config)
    assert cfg.depths == (3, 3, 27, 3)
    assert cfg.hidden_sizes == (128, 256, 512, 1024)
    assert cfg.sample_rate == spec.audio_config.sample_rate


def test_build_model_from_official_spec_offline() -> None:
    initialize_registry()
    spec = get_model_spec("esp_aves2_sed_birdcode_convnext_encoder")
    assert spec is not None
    tiny = spec.model_copy(update={"init_config": {**spec.init_config, **_TINY_ARCH}})
    m = build_model_from_spec(tiny, device="cpu")
    assert isinstance(m, Model)
    assert m.init_config.n_mels == 256


# --------------------------------------------------------------------------- #
#  Review fixes                                                                 #
# --------------------------------------------------------------------------- #


def test_mel_front_end_is_independent_of_batch_composition(tiny_encoder: Model) -> None:
    """The 80 dB floor must be per clip, not taken from the loudest clip in the batch."""
    torch.manual_seed(9)
    quiet = torch.randn(1, _SR) * 1e-4
    loud = torch.randn(1, _SR) * 10.0
    alone = tiny_encoder.process_audio(quiet)
    batched = tiny_encoder.process_audio(torch.cat([quiet, loud]))[:1]
    # Only float rounding from the batched mel matmul remains (~1e-6); a
    # batch-wide floor would move the quiet clip by tens of dB (several units
    # after z-scoring).
    assert torch.allclose(alone, batched, atol=1e-5)


def test_mel_front_end_matches_upstream_at_batch_size_one(tiny_encoder: Model) -> None:
    """Per-clip floor is identical to the upstream (batch-wide) recipe for a single clip."""
    import torchaudio.transforms as T

    wav = torch.randn(1, _SR * _SECONDS)
    spec = T.Spectrogram(n_fft=2048, hop_length=256, power=2.0)
    mel = T.MelScale(n_mels=256, sample_rate=_SR, n_stft=1025)
    db = T.AmplitudeToDB(stype="power", top_db=80)
    upstream = ((db(mel(spec(wav))) - (-13.369)) / 13.162).unsqueeze(1)
    assert torch.equal(tiny_encoder.process_audio(wav), upstream)


def test_cls_token_aggregation_is_rejected(tiny_encoder: Model, audio: torch.Tensor) -> None:
    tiny_encoder.register_hooks_for_layers(["last_layer"])
    try:
        with pytest.raises(ValueError, match="no class token"):
            tiny_encoder.extract_embeddings(audio, aggregation="cls_token")
    finally:
        tiny_encoder.deregister_all_hooks()


def test_spec_declared_encoder_only_skips_num_classes_probe(
    tmp_path: Path, audio: torch.Tensor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``extra_config.return_features_only`` must avoid the extra checkpoint read and head build."""
    import avex.models.utils.load as load_mod

    src = Model(pretrained=False, device="cpu", init_config=_TINY_ARCH).eval()
    ckpt = tmp_path / "best_model.pt"
    torch.save(_as_frame_detector_state_dict(src, num_classes=3), ckpt)

    def _fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("num_classes probe must not run for an encoder-only spec")

    monkeypatch.setattr(load_mod, "_extract_num_classes_from_checkpoint", _fail)
    spec = ModelSpec(
        name="convnext",
        pretrained=False,
        device="cpu",
        init_config=_TINY_ARCH,
        extra_config={"return_features_only": True},
    )
    loaded = load_model(spec, device="cpu", checkpoint_path=str(ckpt)).eval()
    assert loaded.return_features_only is True and loaded.classifier is None
    with torch.no_grad():
        assert torch.equal(src(audio), loaded(audio))


def test_official_spec_declares_encoder_only() -> None:
    initialize_registry()
    spec = get_model_spec("esp_aves2_sed_birdcode_convnext_encoder")
    assert spec is not None and spec.extra_config == {"return_features_only": True, "weights_only": True}


def test_weights_only_is_opt_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only specs with ``extra_config.weights_only`` change how checkpoints are unpickled."""
    import avex.models.utils.load as load_mod

    assert load_mod._checkpoint_load_kwargs("cpu") == {"map_location": "cpu"}
    assert load_mod._checkpoint_load_kwargs("cpu", weights_only=True) == {"map_location": "cpu", "weights_only": True}

    src = Model(pretrained=False, device="cpu", init_config=_TINY_ARCH).eval()
    ckpt = tmp_path / "best_model.pt"
    torch.save(_as_frame_detector_state_dict(src), ckpt)

    seen: list[object] = []
    orig = load_mod.universal_torch_load

    def _spy(path: object, **kwargs: object) -> object:
        seen.append(kwargs.get("weights_only"))
        return orig(path, **kwargs)

    monkeypatch.setattr(load_mod, "universal_torch_load", _spy)
    for extra, expected in (
        ({"return_features_only": True}, None),
        ({"return_features_only": True, "weights_only": True}, True),
    ):
        seen.clear()
        spec = ModelSpec(name="convnext", pretrained=False, device="cpu", init_config=_TINY_ARCH, extra_config=extra)
        load_model(spec, device="cpu", checkpoint_path=str(ckpt))
        assert seen == [expected]
