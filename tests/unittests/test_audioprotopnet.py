"""Unit tests for AudioProtoPNet and plain ConvNeXt model wrappers.

AudioProtoPNet tests cover:
  - HuggingFace model loading (pretrained=True from hub)
  - Forward pass shape and dtype
  - Label mapping availability
  - Layer discovery for hook-based probing
  - API-level model loading via load_model("audioprotopnet_20")
  - Hook-based embedding extraction

Plain ConvNeXt tests cover:
  - Architecture init without a checkpoint (pretrained=False)
  - Forward pass shape
  - Layer discovery
  - Lightning checkpoint unwrapping + double-prefix remapping in load_state_dict
  - Mel params read from audio_config

AudioProtoPNet tests require network access to download from HuggingFace hub.
The smallest variant ("1" prototype/class) is used to keep load time short.
"""

import pytest
import torch

from avex.models.audioprotopnet import Model as AudioProtoPNetModel
from avex.models.convnext import Model as ConvNextModel

# Smallest AudioProtoPNet variant — fewest prototypes, fastest to download/load.
_FAST_MODEL_ID = "DBD-research-group/AudioProtoPNet-1-BirdSet-XCL"
_SAMPLE_RATE = 32_000
_CLIP_SECONDS = 5
_NUM_CLASSES = 9_736


# =========================================================================== #
#  Fixtures                                                                    #
# =========================================================================== #


@pytest.fixture(scope="module")
def audioprotopnet_model() -> AudioProtoPNetModel:
    """Load AudioProtoPNet (variant 1) once for the entire module.

    Returns
    -------
    AudioProtoPNetModel
        Loaded model in eval mode.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = AudioProtoPNetModel(pretrained=True, device=device, model_id=_FAST_MODEL_ID)
    m.eval()
    return m


@pytest.fixture
def dummy_audio(audioprotopnet_model: AudioProtoPNetModel) -> torch.Tensor:
    """Batch of two 5-second random waveforms at 32 kHz.

    Returns
    -------
    torch.Tensor
        Shape ``(2, sample_rate * clip_seconds)``.
    """
    n_samples = _SAMPLE_RATE * _CLIP_SECONDS
    return torch.randn(2, n_samples, device=audioprotopnet_model.device)


@pytest.fixture(scope="module")
def plain_convnext_model() -> ConvNextModel:
    """Plain ConvNeXt built from architecture only (pretrained=False), CPU.

    Returns
    -------
    ConvNextModel
        Randomly-initialised model in eval mode.
    """
    m = ConvNextModel(
        pretrained=False,
        device="cpu",
        model_id="facebook/convnext-base-224-22k",
        num_classes=_NUM_CLASSES,
    )
    m.eval()
    return m


@pytest.fixture(scope="module")
def plain_convnext_audio() -> torch.Tensor:
    """Batch of two 5-second random waveforms at 32 kHz for CPU ConvNeXt tests.

    Returns
    -------
    torch.Tensor
        Shape ``(2, sample_rate * clip_seconds)``.
    """
    return torch.randn(2, _SAMPLE_RATE * _CLIP_SECONDS)


# =========================================================================== #
#  AudioProtoPNet — model loading                                              #
# =========================================================================== #


def test_audioprotopnet_loads(audioprotopnet_model: AudioProtoPNetModel) -> None:
    assert audioprotopnet_model is not None
    assert audioprotopnet_model.num_classes == _NUM_CLASSES


def test_audioprotopnet_pretrained_false_requires_num_classes() -> None:
    """pretrained=False without num_classes must raise a clear error."""
    with pytest.raises(ValueError, match="num_classes"):
        AudioProtoPNetModel(pretrained=False, device="cpu")


def test_audioprotopnet_pretrained_false_custom_num_classes() -> None:
    """pretrained=False with num_classes creates a randomly-initialised model."""
    m = AudioProtoPNetModel(pretrained=False, device="cpu", num_classes=50)
    m.eval()
    assert m.num_classes == 50
    audio = torch.randn(1, _SAMPLE_RATE * _CLIP_SECONDS)
    with torch.no_grad():
        logits = m(audio)
    assert logits.shape == (1, 50)


def test_audioprotopnet_model_id_none_uses_default() -> None:
    """model_id=None must fall back to the AudioProtoPNet-20 default."""
    m = AudioProtoPNetModel(pretrained=True, device="cpu", model_id=None)
    assert "AudioProtoPNet" in m.model_id


# =========================================================================== #
#  AudioProtoPNet — forward pass                                               #
# =========================================================================== #


def test_audioprotopnet_forward_shape(audioprotopnet_model: AudioProtoPNetModel, dummy_audio: torch.Tensor) -> None:
    with torch.no_grad():
        logits = audioprotopnet_model(dummy_audio)
    assert logits.shape == (2, _NUM_CLASSES), f"Expected (2, {_NUM_CLASSES}), got {logits.shape}"


def test_audioprotopnet_forward_dtype(audioprotopnet_model: AudioProtoPNetModel, dummy_audio: torch.Tensor) -> None:
    with torch.no_grad():
        logits = audioprotopnet_model(dummy_audio)
    assert logits.dtype in (torch.float32, torch.float16, torch.bfloat16)


def test_audioprotopnet_forward_single_sample(audioprotopnet_model: AudioProtoPNetModel) -> None:
    audio = torch.randn(1, _SAMPLE_RATE * _CLIP_SECONDS, device=audioprotopnet_model.device)
    with torch.no_grad():
        logits = audioprotopnet_model(audio)
    assert logits.shape == (1, _NUM_CLASSES)


def test_audioprotopnet_padding_mask_ignored(
    audioprotopnet_model: AudioProtoPNetModel, dummy_audio: torch.Tensor
) -> None:
    mask = torch.zeros(2, dummy_audio.shape[1], dtype=torch.bool, device=audioprotopnet_model.device)
    with torch.no_grad():
        out_no_mask = audioprotopnet_model(dummy_audio)
        out_with_mask = audioprotopnet_model(dummy_audio, padding_mask=mask)
    assert torch.allclose(out_no_mask, out_with_mask)


# =========================================================================== #
#  AudioProtoPNet — label mapping                                              #
# =========================================================================== #


def test_audioprotopnet_label_mapping_exists(audioprotopnet_model: AudioProtoPNetModel) -> None:
    assert audioprotopnet_model.label_mapping is not None
    assert len(audioprotopnet_model.label_mapping) == _NUM_CLASSES


def test_audioprotopnet_label_mapping_values_are_common_names(
    audioprotopnet_model: AudioProtoPNetModel,
) -> None:
    """Values should be human-readable common names, not raw eBird codes."""
    sample = next(iter(audioprotopnet_model.label_mapping.values()))
    assert isinstance(sample, str)
    # Common names contain spaces (e.g. "Common Ostrich") or are multi-word;
    # raw eBird codes never contain spaces (e.g. "ostric2").
    assert " " in sample or sample[0].isupper(), f"Looks like a raw eBird code: {sample!r}"


def test_audioprotopnet_ebird_codes_accessible(
    audioprotopnet_model: AudioProtoPNetModel,
) -> None:
    """ebird_codes attribute exposes the raw eBird code for each class id."""
    assert len(audioprotopnet_model.ebird_codes) == _NUM_CLASSES
    sample_code = audioprotopnet_model.ebird_codes[0]
    assert isinstance(sample_code, str)
    assert " " not in sample_code, f"Expected eBird code, got: {sample_code!r}"


# =========================================================================== #
#  AudioProtoPNet — layer discovery                                            #
# =========================================================================== #


def test_audioprotopnet_discover_layers(audioprotopnet_model: AudioProtoPNetModel) -> None:
    audioprotopnet_model._layer_names = []
    audioprotopnet_model._discover_linear_layers()
    assert len(audioprotopnet_model._layer_names) == 4, f"Expected 4 stages, found: {audioprotopnet_model._layer_names}"


def test_audioprotopnet_layer_names_are_backbone_stages(
    audioprotopnet_model: AudioProtoPNetModel,
) -> None:
    audioprotopnet_model._layer_names = []
    audioprotopnet_model._discover_linear_layers()
    for name in audioprotopnet_model._layer_names:
        assert "backbone.encoder.stages" in name, f"Unexpected layer name: {name}"


# =========================================================================== #
#  AudioProtoPNet — API-level loading                                          #
# =========================================================================== #


def test_audioprotopnet_load_model_api() -> None:
    from importlib import resources

    from avex import load_model

    # The YAML lives in checkpoints/, not official_models/, so load by path.
    pkg_root = resources.files("avex.api.configs.checkpoints")
    yaml_path = pkg_root / "audioprotopnet_20.yml"
    with resources.as_file(yaml_path) as p:
        loaded = load_model(str(p), device="cpu")

    assert loaded is not None
    assert loaded.num_classes == _NUM_CLASSES

    audio = torch.randn(1, _SAMPLE_RATE * _CLIP_SECONDS)
    loaded.eval()
    with torch.no_grad():
        logits = loaded(audio)
    assert logits.shape == (1, _NUM_CLASSES)


# =========================================================================== #
#  AudioProtoPNet — hook-based embedding extraction                            #
# =========================================================================== #


def test_audioprotopnet_hook_embedding_extraction(
    audioprotopnet_model: AudioProtoPNetModel, dummy_audio: torch.Tensor
) -> None:
    audioprotopnet_model._layer_names = []
    audioprotopnet_model._discover_linear_layers()
    assert audioprotopnet_model._layer_names

    last_stage = audioprotopnet_model._layer_names[-1]
    audioprotopnet_model.register_hooks_for_layers([last_stage])

    with torch.no_grad():
        audioprotopnet_model(dummy_audio)

    assert last_stage in audioprotopnet_model._hook_outputs
    feat = audioprotopnet_model._hook_outputs[last_stage]
    assert feat.shape[0] == 2

    audioprotopnet_model.deregister_all_hooks()


# =========================================================================== #
#  Plain ConvNeXt — architecture and forward                                   #
# =========================================================================== #


def test_plain_convnext_loads(plain_convnext_model: ConvNextModel) -> None:
    assert plain_convnext_model.num_classes == _NUM_CLASSES
    assert plain_convnext_model.model_id == "facebook/convnext-base-224-22k"


def test_plain_convnext_init_config_overrides_defaults() -> None:
    """Architecture specified in init_config must override module-level defaults."""
    # Tiny variant: depths [2,2,6,2], hidden_sizes [96,192,384,768]
    m = ConvNextModel(
        pretrained=False,
        device="cpu",
        num_classes=10,
        init_config={"depths": [2, 2, 6, 2], "hidden_sizes": [96, 192, 384, 768]},
    )
    cfg = m.model.config
    assert cfg.depths == [2, 2, 6, 2]
    assert cfg.hidden_sizes == [96, 192, 384, 768]


def test_plain_convnext_architecture_matches_hf_pretrained() -> None:
    """Architecture built from init_config must match the HF pretrained one.

    Verifies that state dicts have the same keys and tensor shapes, so that a
    checkpoint trained with pretrained=True can be loaded into a pretrained=False
    model (and vice versa).
    """
    from transformers import ConvNextConfig, ConvNextForImageClassification

    from avex.models.convnext import _CHECKPOINT_CONFIGS_PKG, _CONVNEXT_BASE_YAML
    from avex.models.utils.registry import load_packaged_yaml_mapping

    yaml_data = load_packaged_yaml_mapping(package=_CHECKPOINT_CONFIGS_PKG, name=_CONVNEXT_BASE_YAML)
    arch = yaml_data["model_spec"]["init_config"]

    n_classes = 10
    # Local (via init_config loaded from packaged YAML)
    local_cfg = ConvNextConfig(
        depths=arch["depths"],
        hidden_sizes=arch["hidden_sizes"],
        num_labels=n_classes,
        num_channels=1,
    )
    local_model = ConvNextForImageClassification(local_cfg)

    # From HF (downloads config JSON only, no weights)
    hf_cfg = ConvNextConfig.from_pretrained("facebook/convnext-base-224-22k", num_labels=n_classes, num_channels=1)
    hf_model = ConvNextForImageClassification(hf_cfg)

    local_keys = set(local_model.state_dict().keys())
    hf_keys = set(hf_model.state_dict().keys())
    assert local_keys == hf_keys, f"Key mismatch: {local_keys.symmetric_difference(hf_keys)}"

    for k in local_keys:
        assert local_model.state_dict()[k].shape == hf_model.state_dict()[k].shape, f"Shape mismatch for {k}"


def test_plain_convnext_forward_shape(plain_convnext_model: ConvNextModel, plain_convnext_audio: torch.Tensor) -> None:
    with torch.no_grad():
        logits = plain_convnext_model(plain_convnext_audio)
    assert logits.shape == (2, _NUM_CLASSES)


def test_plain_convnext_discover_layers(plain_convnext_model: ConvNextModel) -> None:
    plain_convnext_model._layer_names = []
    plain_convnext_model._discover_linear_layers()
    assert len(plain_convnext_model._layer_names) == 4
    for name in plain_convnext_model._layer_names:
        assert "convnext.encoder.stages" in name, f"Unexpected: {name}"


def test_plain_convnext_load_state_dict_lightning_remapping(
    plain_convnext_model: ConvNextModel,
) -> None:
    """load_state_dict must unwrap Lightning dicts and strip double model prefix.

    Real Lightning checkpoints only contain self.model weights; torchaudio
    transform buffers (_spec_transform, _mel_scale) are not serialised.
    """
    real_sd = plain_convnext_model.model.state_dict()
    # Simulate Lightning double-prefix: model.model.<key>
    wrapped = {f"model.model.{k}": v for k, v in real_sd.items()}
    lightning_ckpt = {"state_dict": wrapped, "epoch": 0}

    result = plain_convnext_model.load_state_dict(lightning_ckpt, strict=False)
    # No unexpected keys after remapping; transform buffers are allowed missing
    assert not result.unexpected_keys


def test_plain_convnext_mel_params_from_convnext_cfg() -> None:
    """convnext_cfg overrides the ConvNextCfg defaults."""
    m = ConvNextModel(
        pretrained=False,
        device="cpu",
        num_classes=10,
        convnext_cfg={
            "sample_rate": 32000,
            "n_fft": 1024,
            "hop_length": 128,
            "n_mels": 128,
            "norm_mean": 0.0,
            "norm_std": 1.0,
        },
    )
    assert m._mel_params.n_fft == 1024
    assert m._mel_params.hop_length == 128
    assert m._mel_params.n_mels == 128
    assert m._mel_params.norm_mean == 0.0
    assert m._mel_params.norm_std == 1.0


# =========================================================================== #
#  ConvNextCfg                                                                 #
# =========================================================================== #


def test_convnext_cfg_defaults() -> None:
    """ConvNextCfg defaults match the BirdSet XCL setup."""
    from avex.configs import ConvNextCfg

    cfg = ConvNextCfg()
    assert cfg.sample_rate == 32000
    assert cfg.n_fft == 2048
    assert cfg.hop_length == 256
    assert cfg.n_mels == 256
    assert cfg.norm_mean == pytest.approx(-13.369)
    assert cfg.norm_std == pytest.approx(13.162)


def test_convnext_cfg_overrides() -> None:
    from avex.configs import ConvNextCfg

    cfg = ConvNextCfg(sample_rate=16000, n_fft=1024, hop_length=128, n_mels=128, norm_mean=0.0, norm_std=1.0)
    assert cfg.sample_rate == 16000
    assert cfg.n_fft == 1024
    assert cfg.n_mels == 128
    assert cfg.norm_mean == pytest.approx(0.0)
    assert cfg.norm_std == pytest.approx(1.0)
