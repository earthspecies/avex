"""AudioProtoPNet audio classification model.

Prototype-based classifier (ConvNeXt backbone + prototype head) loaded from
HuggingFace hub via ``AutoModelForSequenceClassification`` with
``trust_remote_code=True``.

Inherits Lightning checkpoint loading, gradient checkpointing, hook-based
probing, and embedding extraction from :class:`~avex.models.convnext.Model`.
Overrides model loading, audio preprocessing (via ``AutoFeatureExtractor``),
and layer-discovery paths.

HF repos: ``DBD-research-group/AudioProtoPNet-{1,5,10,20}-BirdSet-XCL``

YAML quick-start::

    model_spec:
      name: audioprotopnet
      pretrained: true
      model_id: "DBD-research-group/AudioProtoPNet-20-BirdSet-XCL"
"""

import logging
from typing import Optional

import torch

from avex.data.ebird_taxonomy import EbirdTaxonomyVersion
from avex.models.base_model import ModelBase
from avex.models.convnext import Model as ConvNextModel

logger = logging.getLogger(__name__)

_DEFAULT_AUDIOPROTOPNET_MODEL_ID = "DBD-research-group/AudioProtoPNet-20-BirdSet-XCL"
# BirdSet XCL species codes follow the eBird taxonomy v2021 release.
_EBIRD_TAXONOMY_VERSION: EbirdTaxonomyVersion = "v2021"


class Model(ConvNextModel):
    """AudioProtoPNet: ConvNeXt backbone + prototype classification head.

    Loaded from HuggingFace hub with ``trust_remote_code=True``.

    Inherits :meth:`load_state_dict`, :meth:`enable_gradient_checkpointing`,
    :meth:`extract_embeddings`, and :meth:`forward` from
    :class:`~avex.models.convnext.Model`.
    """

    def __init__(
        self,
        pretrained: bool = True,
        device: str = "cuda",
        audio_config: object = None,
        num_classes: Optional[int] = None,
        model_id: Optional[str] = None,
        ebird_taxonomy_version: EbirdTaxonomyVersion = _EBIRD_TAXONOMY_VERSION,
        init_config: Optional[dict] = None,  # unused; absorbed for API compatibility
        **kwargs: object,
    ) -> None:
        # Skip ConvNextModel.__init__ — we don't need the mel pipeline or
        # ConvNextForImageClassification.  Go straight to ModelBase.
        ModelBase.__init__(self, device=device, audio_config=audio_config)

        try:
            from transformers import (
                AutoConfig,
                AutoFeatureExtractor,
                AutoModelForSequenceClassification,
            )
        except ImportError as e:
            raise ImportError("transformers is required.  Install with: pip install transformers") from e

        self.gradient_checkpointing = False
        self.model_id = model_id or _DEFAULT_AUDIOPROTOPNET_MODEL_ID

        if pretrained:
            if num_classes is not None:
                logger.info(
                    "Loading AudioProtoPNet backbone from %s (overriding num_classes=%d)",
                    self.model_id,
                    num_classes,
                )
                hf_config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
                hf_config.num_classes = num_classes
                hf_config.id2label = {i: str(i) for i in range(num_classes)}
                hf_config.label2id = {str(i): i for i in range(num_classes)}
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_id,
                    config=hf_config,
                    trust_remote_code=True,
                    ignore_mismatched_sizes=True,
                )
                self.num_classes: int = num_classes
            else:
                logger.info("Loading AudioProtoPNet weights from %s", self.model_id)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id, trust_remote_code=True)
                self.num_classes = self.model.config.num_classes
        else:
            if num_classes is None:
                raise ValueError(
                    "num_classes is required when pretrained=False for AudioProtoPNet. "
                    "The model architecture is still fetched from the HuggingFace repo "
                    "(config + code only, no checkpoint weights)."
                )
            # NOTE: pretrained=False still requires network access (or a warm HF cache)
            # because the prototype head is implemented as remote custom code in the
            # DBD-research-group repo.  Tests using this path are therefore
            # network/cache-dependent.
            logger.info(
                "Creating AudioProtoPNet with random weights from %s (num_classes=%d)",
                self.model_id,
                num_classes,
            )
            config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
            config.num_classes = num_classes
            config.id2label = {i: str(i) for i in range(num_classes)}
            config.label2id = {str(i): i for i in range(num_classes)}
            self.model = AutoModelForSequenceClassification.from_config(config, trust_remote_code=True)
            self.num_classes = num_classes

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id, trust_remote_code=True)

        if hasattr(self.model.config, "id2label") and self.model.config.id2label:
            self.ebird_codes: dict[int, str] = {int(k): v for k, v in self.model.config.id2label.items()}
            self.label_mapping = self._build_label_mapping(self.ebird_codes, ebird_taxonomy_version)
        else:
            self.ebird_codes = {}
            self.label_mapping = None

        self.model = self.model.to(device)

    @staticmethod
    def _build_label_mapping(
        ebird_codes: dict[int, str],
        taxonomy_version: EbirdTaxonomyVersion = _EBIRD_TAXONOMY_VERSION,
    ) -> dict[int, str]:
        """Return ``{class_id: common_name}`` using the bundled eBird taxonomy.

        Falls back to the raw eBird code for any code not found in the taxonomy.

        Parameters
        ----------
        ebird_codes
            Class index to eBird species code.
        taxonomy_version
            eBird/Clements release used when the model was trained.

        Returns
        -------
        dict[int, str]
            Mapping of class index to common name (or raw eBird code as fallback).
        """
        from avex.data.ebird_taxonomy import load as load_taxonomy

        try:
            taxonomy = load_taxonomy(taxonomy_version)
        except Exception as exc:
            logger.warning(
                "Could not load eBird taxonomy %s, using raw eBird codes: %s",
                taxonomy_version,
                exc,
            )
            return dict(ebird_codes)

        return {idx: taxonomy[code]["common_name"] if code in taxonomy else code for idx, code in ebird_codes.items()}

    @property
    def backbone(self) -> "torch.nn.Module":
        """ConvNeXt backbone only (no prototype head) — used by freeze_backbone_epochs."""
        return self.model.model.backbone

    def _get_last_non_classification_layer(self) -> str:
        """Resolve ``last_layer`` to the prototype vector before the classifier.

        Returns
        -------
        str
            Virtual layer name used by AudioProtoPNet probing.
        """
        return "last_layer"

    def register_hooks_for_layers(self, layer_names: list[str]) -> list[str]:
        """Register probe targets.

        ``last_layer`` is virtual for AudioProtoPNet: the pre-classifier
        prototype activation vector is computed inside the head, not emitted by
        a submodule that can be hooked cleanly.

        Returns
        -------
        list[str]
            Resolved layer names.
        """
        if layer_names == ["last_layer"]:
            self.deregister_all_hooks()
            self._hook_layers = ["last_layer"]
            return ["last_layer"]
        return super().register_hooks_for_layers(layer_names)

    def _extract_pre_classifier_embeddings(self, x: torch.Tensor | dict) -> torch.Tensor:
        """Return pooled prototype activations passed directly to the classifier.

        Returns
        -------
        torch.Tensor
            Shape ``[batch, num_prototypes]``.
        """
        wav = x["raw_wav"] if isinstance(x, dict) else x
        processed = self.process_audio(wav)
        if isinstance(processed, dict):
            input_values = processed.get("input_values")
            if input_values is None:
                input_values = next(iter(processed.values()))
        else:
            input_values = processed

        backbone_outputs = self.model.model(input_values)
        last_hidden_state = backbone_outputs[0]
        _, info = self.model.head(last_hidden_state)
        return info[0]

    def extract_embeddings(
        self,
        x: torch.Tensor | dict,
        *,
        padding_mask: Optional[torch.Tensor] = None,
        aggregation: str = "none",
        freeze_backbone: bool = True,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Extract embeddings, with ``last_layer`` as pre-classifier activations.

        Returns
        -------
        torch.Tensor | list[torch.Tensor]
            Probe embeddings.
        """
        if self._hook_layers == ["last_layer"]:
            if freeze_backbone:
                with torch.no_grad():
                    return self._extract_pre_classifier_embeddings(x)
            return self._extract_pre_classifier_embeddings(x)
        return super().extract_embeddings(
            x,
            padding_mask=padding_mask,
            aggregation=aggregation,
            freeze_backbone=freeze_backbone,
        )

    def _discover_linear_layers(self) -> None:
        """Discover the four ConvNeXt encoder stages for hook-based probing.

        AudioProtoPNet path: ``model.model.backbone.encoder.stages.{0-3}``.
        """
        if self._layer_names:
            return

        self._layer_names = []
        for name, _ in self.named_modules():
            parts = name.split(".")
            if (
                len(parts) == 6
                and parts[:5] == ["model", "model", "backbone", "encoder", "stages"]
                and parts[5].isdigit()
            ):
                self._layer_names.append(name)

        logger.info(
            "Discovered %d AudioProtoPNet encoder stages: %s",
            len(self._layer_names),
            self._layer_names,
        )

    def process_audio(self, x: torch.Tensor) -> torch.Tensor:
        """Convert raw 32 kHz waveform using the HuggingFace feature extractor.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch_size, time_steps)``.

        Returns
        -------
        torch.Tensor
            Preprocessed tensor on ``self.device``.
        """
        samples = [x[i].cpu().numpy() for i in range(x.shape[0])]
        return self.feature_extractor(samples, return_tensors="pt").to(self.device)
