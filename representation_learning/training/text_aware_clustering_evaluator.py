"""
Text-aware clustering evaluation component for training.

This module extends the clustering evaluator to handle text-labeled datasets
by extracting meaningful semantic labels from the data.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from representation_learning.configs import ClusteringEvalConfig
from representation_learning.training.clustering_evaluator import ClusteringEvaluator

logger = logging.getLogger(__name__)


class TextAwareClusteringEvaluator(ClusteringEvaluator):
    """Clustering evaluator that handles text-labeled datasets."""

    def __init__(
        self,
        config: ClusteringEvalConfig,
        device: torch.device,
        text_label_strategy: str = "canonical_name",
    ) -> None:
        """Initialize the text-aware clustering evaluator.

        Parameters
        ----------
        config : ClusteringEvalConfig
            Configuration for clustering evaluation
        device : torch.device
            Device for computation
        text_label_strategy : str, optional
            Strategy for extracting labels from text datasets:
            - "canonical_name": Use canonical_name field (best for AnimalSpeak)
            - "hash_text": Hash text_label content to create pseudo-labels
            - "first_text": Use first element of text_label list
            - "labels_field": Use existing labels field (for AudioSet)
        """
        super().__init__(config, device)
        self.text_label_strategy = text_label_strategy
        self._label_cache = {}  # Cache for consistent label mapping

    def _extract_embeddings(
        self, model: torch.nn.Module, dataloader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract embeddings and meaningful labels from dataloader.

        This override extracts semantic labels from text datasets instead of
        using dummy labels.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple containing embeddings and semantic labels tensors.
        """
        model.eval()
        embeddings_list = []
        labels_list = []
        sample_count = 0

        # For building consistent label mappings
        all_raw_samples = []

        # Handle layer name resolution for models that need it
        layer_names = self.layer_names.copy()
        if "last_layer" in layer_names:
            # Find the actual last linear layer name
            linear_layers = [
                n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
            ]
            if linear_layers:
                layer_names = [
                    linear_layers[-1] if name == "last_layer" else name
                    for name in layer_names
                ]
            else:
                logger.warning("No linear layers found, using 'last_layer' as-is")

        with torch.no_grad():
            for batch in dataloader:
                if self.config.max_samples and sample_count >= self.config.max_samples:
                    break

                # Move batch to device
                wav = batch["raw_wav"].to(self.device)
                padding_mask = batch.get("padding_mask")
                if padding_mask is not None:
                    padding_mask = padding_mask.to(self.device)

                # Extract embeddings
                try:
                    if padding_mask is None:
                        embeddings = model.extract_embeddings(wav, layers=layer_names)
                    else:
                        inp = {"raw_wav": wav, "padding_mask": padding_mask}
                        embeddings = model.extract_embeddings(inp, layers=layer_names)

                    # Extract meaningful labels from text datasets
                    semantic_labels = self._extract_semantic_labels(batch)

                    if semantic_labels is not None:
                        # Store raw samples for consistent labeling
                        all_raw_samples.extend(self._extract_raw_samples(batch))

                        # Move to CPU for memory efficiency
                        embeddings_list.append(embeddings.cpu())
                        labels_list.append(semantic_labels.cpu())
                        sample_count += len(semantic_labels)

                        # Limit samples if configured
                        if (
                            self.config.max_samples
                            and sample_count >= self.config.max_samples
                        ):
                            # Truncate last batch if needed
                            excess = sample_count - self.config.max_samples
                            if excess > 0:
                                embeddings_list[-1] = embeddings_list[-1][:-excess]
                                labels_list[-1] = labels_list[-1][:-excess]
                                all_raw_samples = all_raw_samples[:-excess]
                            break
                    else:
                        logger.warning(
                            "No semantic labels could be extracted from batch"
                        )

                except Exception as e:
                    logger.warning(f"Failed to extract embeddings for batch: {e}")
                    continue

        if not embeddings_list:
            logger.warning("No embeddings were successfully extracted")
            return torch.empty(0), torch.empty(0)

        # Create consistent label mapping across all samples
        final_labels = self._create_consistent_labels(all_raw_samples)

        logger.info(f"Extracted embeddings from {sample_count} samples")
        logger.info(f"Found {len(set(final_labels.tolist()))} unique semantic labels")

        return torch.cat(embeddings_list), final_labels

    def _extract_semantic_labels(self, batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Extract semantic labels from batch based on strategy.

        Returns
        -------
        Optional[torch.Tensor]
            Tensor containing semantic labels or None if not available.
        """

        if self.text_label_strategy == "canonical_name":
            return self._extract_canonical_name_labels(batch)
        elif self.text_label_strategy == "hash_text":
            return self._extract_hash_text_labels(batch)
        elif self.text_label_strategy == "first_text":
            return self._extract_first_text_labels(batch)
        elif self.text_label_strategy == "labels_field":
            return self._extract_labels_field_labels(batch)
        else:
            logger.warning(f"Unknown text_label_strategy: {self.text_label_strategy}")
            return None

    def _extract_raw_samples(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract raw sample data for consistent labeling.

        Returns
        -------
        List[Dict[str, Any]]
            List of raw sample dictionaries.
        """
        samples = []
        batch_size = len(batch["raw_wav"])

        for i in range(batch_size):
            sample = {}
            for key, value in batch.items():
                if key == "raw_wav" or key == "padding_mask":
                    continue
                if isinstance(value, list):
                    sample[key] = value[i] if i < len(value) else None
                elif hasattr(value, "__getitem__") and not isinstance(value, str):
                    try:
                        sample[key] = value[i]
                    except (IndexError, TypeError):
                        sample[key] = None
                else:
                    sample[key] = value
            samples.append(sample)

        return samples

    def _create_consistent_labels(
        self, raw_samples: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """Create consistent numeric labels from raw samples.

        Returns
        -------
        torch.Tensor
            Tensor containing numeric labels for the samples.
        """
        if self.text_label_strategy == "canonical_name":
            return self._create_canonical_name_labels(raw_samples)
        elif self.text_label_strategy == "hash_text":
            return self._create_hash_text_labels(raw_samples)
        elif self.text_label_strategy == "first_text":
            return self._create_first_text_labels(raw_samples)
        elif self.text_label_strategy == "labels_field":
            return self._create_labels_field_labels(raw_samples)
        else:
            return torch.zeros(len(raw_samples), dtype=torch.long)

    def _create_canonical_name_labels(
        self, raw_samples: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """Create labels from canonical_name field (best for AnimalSpeak).

        Returns
        -------
        torch.Tensor
            Tensor containing numeric labels based on canonical names.
        """
        # Extract canonical names
        canonical_names = []
        for sample in raw_samples:
            canonical_name = sample.get("canonical_name")
            if canonical_name:
                canonical_names.append(str(canonical_name))
            else:
                canonical_names.append("unknown")

        # Create consistent mapping
        unique_names = sorted(set(canonical_names))
        name_to_idx = {name: idx for idx, name in enumerate(unique_names)}

        labels = [name_to_idx[name] for name in canonical_names]
        logger.info(
            f"Created canonical_name labels: {len(unique_names)} unique species"
        )

        return torch.tensor(labels, dtype=torch.long)

    def _create_hash_text_labels(
        self, raw_samples: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """Create labels by hashing text content.

        Returns
        -------
        torch.Tensor
            Tensor containing numeric labels based on text hash.
        """
        pseudo_labels = []
        for sample in raw_samples:
            text_label = sample.get("text_label")
            if text_label:
                # Use first text if it's a list
                if isinstance(text_label, list) and len(text_label) > 0:
                    text_key = str(text_label[0])
                else:
                    text_key = str(text_label)

                # Create consistent hash-based label
                hash_val = int(hashlib.md5(text_key.encode()).hexdigest(), 16)
                pseudo_labels.append(hash_val % 10000)  # Limit to reasonable range
            else:
                pseudo_labels.append(0)  # Unknown

        unique_labels = len(set(pseudo_labels))
        logger.info(f"Created hash_text labels: {unique_labels} unique hash labels")

        return torch.tensor(pseudo_labels, dtype=torch.long)

    def _create_first_text_labels(
        self, raw_samples: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """Create labels from first element of text_label.

        Returns
        -------
        torch.Tensor
            Tensor containing numeric labels based on first text element.
        """
        first_texts = []
        for sample in raw_samples:
            text_label = sample.get("text_label")
            if text_label:
                if isinstance(text_label, list) and len(text_label) > 0:
                    first_texts.append(str(text_label[0]))
                else:
                    first_texts.append(str(text_label))
            else:
                first_texts.append("unknown")

        # Create consistent mapping
        unique_texts = sorted(set(first_texts))
        text_to_idx = {text: idx for idx, text in enumerate(unique_texts)}

        labels = [text_to_idx[text] for text in first_texts]
        logger.info(
            f"Created first_text labels: {len(unique_texts)} unique first texts"
        )

        return torch.tensor(labels, dtype=torch.long)

    def _create_labels_field_labels(
        self, raw_samples: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """Create labels from existing labels field (for AudioSet).

        Returns
        -------
        torch.Tensor
            Tensor containing numeric labels from existing labels field.
        """
        label_lists = []
        for sample in raw_samples:
            labels = sample.get("labels", [])
            if labels and isinstance(labels, list):
                # Use first label for simplicity
                label_lists.append(str(labels[0]))
            else:
                label_lists.append("unknown")

        # Create consistent mapping
        unique_labels = sorted(set(label_lists))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

        labels = [label_to_idx[label] for label in label_lists]
        logger.info(f"Created labels_field labels: {len(unique_labels)} unique labels")

        return torch.tensor(labels, dtype=torch.long)

    # Helper methods for batch-level extraction (legacy compatibility)
    def _extract_canonical_name_labels(
        self, batch: Dict[str, Any]
    ) -> Optional[torch.Tensor]:
        """Extract canonical name labels from batch (legacy method).

        Returns
        -------
        Optional[torch.Tensor]
            Tensor containing canonical name labels or None if not available.
        """
        # This is a simplified version - the consistent labeling happens in
        # _create_consistent_labels
        if "canonical_name" in batch:
            return torch.zeros(len(batch["raw_wav"]), dtype=torch.long)  # Placeholder
        return None

    def _extract_hash_text_labels(
        self, batch: Dict[str, Any]
    ) -> Optional[torch.Tensor]:
        """Extract hash text labels from batch (legacy method).

        Returns
        -------
        Optional[torch.Tensor]
            Tensor containing hash text labels or None if not available.
        """
        if "text_label" in batch:
            return torch.zeros(len(batch["raw_wav"]), dtype=torch.long)  # Placeholder
        return None

    def _extract_first_text_labels(
        self, batch: Dict[str, Any]
    ) -> Optional[torch.Tensor]:
        """Extract first text labels from batch (legacy method).

        Returns
        -------
        Optional[torch.Tensor]
            Tensor containing first text labels or None if not available.
        """
        if "text_label" in batch:
            return torch.zeros(len(batch["raw_wav"]), dtype=torch.long)  # Placeholder
        return None

    def _extract_labels_field_labels(
        self, batch: Dict[str, Any]
    ) -> Optional[torch.Tensor]:
        """Extract labels field labels from batch (legacy method).

        Returns
        -------
        Optional[torch.Tensor]
            Tensor containing labels field labels or None if not available.
        """
        if "labels" in batch:
            return torch.zeros(len(batch["raw_wav"]), dtype=torch.long)  # Placeholder
        return None
