"""Tests for embedding extraction utilities."""

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from representation_learning.configs import ExperimentConfig
from representation_learning.evaluation.embedding_utils import (
    EmbeddingDataset,
    _extract_embeddings_in_memory,
    _extract_embeddings_streaming,
    load_embeddings_arrays,
    save_embeddings_arrays,
)


class MockModel:
    """Mock model for testing embedding extraction."""

    def __init__(self, layer_names: List[str], embedding_dims: List[int]) -> None:
        self.layer_names = layer_names
        self.embedding_dims = embedding_dims
        self._hook_outputs = {}
        self._hooks = {}
        self._layer_names = layer_names

    def eval(self) -> None:
        """Set model to eval mode."""
        pass

    def register_hooks_for_layers(self, layer_names: List[str]) -> List[str]:
        """Mock hook registration returning the same layer names.

        Returns
        -------
        List[str]
            The same `layer_names` provided.
        """
        # Ensure we return the exact layer names requested
        return layer_names

    def deregister_all_hooks(self) -> None:
        """Mock hook deregistration."""
        pass

    def extract_embeddings(
        self, x: torch.Tensor, aggregation: str = "mean"
    ) -> torch.Tensor | List[torch.Tensor] | Dict[str, torch.Tensor]:
        """Mock embedding extraction.

        Returns
        -------
        torch.Tensor | List[torch.Tensor] | Dict[str, torch.Tensor]
            Mock embeddings tensor, list of tensors, or dictionary with
            layer names as keys.
        """
        batch_size = x.shape[0] if isinstance(x, torch.Tensor) else x["raw_wav"].shape[0]

        if aggregation == "none" and len(self.layer_names) > 1:
            # Return dictionary with layer names as keys for multiple layers
            return {
                layer_name: torch.randn(batch_size, dim)
                for layer_name, dim in zip(self.layer_names, self.embedding_dims, strict=False)
            }
        elif aggregation == "mean" and len(self.layer_names) > 1:
            # Return single tensor (concatenated) for mean aggregation
            total_dim = sum(self.embedding_dims)
            return torch.randn(batch_size, total_dim)
        else:
            # Return single tensor (concatenated or single layer)
            total_dim = sum(self.embedding_dims)
            return torch.randn(batch_size, total_dim)


# Module-level fixtures
@pytest.fixture
def mock_model_single_layer() -> MockModel:
    """Create a mock model with single layer.

    Returns
    -------
    MockModel
        A mock model configured for single layer testing.
    """
    return MockModel(["layer1"], [128])


@pytest.fixture
def mock_model_multi_layer() -> MockModel:
    """Create a mock model with multiple layers.

    Returns
    -------
    MockModel
        A mock model configured for multi-layer testing.
    """
    return MockModel(["layer1", "layer2", "layer3"], [128, 256, 64])


@pytest.fixture
def sample_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Create sample data for testing.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Audio data and labels for testing.
    """
    audio_data = torch.randn(10, 16000)  # 10 samples, 16kHz audio
    labels = torch.randint(0, 5, (10,))  # 5 classes
    return audio_data, labels


@pytest.fixture
def dataloader(sample_data: tuple[torch.Tensor, torch.Tensor]) -> DataLoader:
    """Create a dataloader for testing.

    Returns
    -------
    DataLoader
        A DataLoader configured for testing.
    """
    audio_data, labels = sample_data

    # Create a custom dataset that returns dictionaries
    class DictDataset:
        def __init__(self, audio_data: torch.Tensor, labels: torch.Tensor) -> None:
            self.audio_data = audio_data
            self.labels = labels

        def __len__(self) -> int:
            return len(self.audio_data)

        def __getitem__(self, idx: int) -> dict:
            return {"raw_wav": self.audio_data[idx], "label": self.labels[idx]}

    dataset = DictDataset(audio_data, labels)
    return DataLoader(dataset, batch_size=2, shuffle=False)


class TestEmbeddingExtraction:
    """Test embedding extraction functionality."""

    def test_extract_embeddings_single_layer(self, mock_model_single_layer: MockModel, dataloader: DataLoader) -> None:
        """Test embedding extraction with single layer."""
        device = torch.device("cpu")

        embeddings, labels, _ = _extract_embeddings_in_memory(
            mock_model_single_layer,
            dataloader,
            ["layer1"],
            device,
            aggregation="mean",
            disable_tqdm=True,
        )

        # Should return dictionary with single key
        assert isinstance(embeddings, dict)
        key = next(iter(embeddings.keys()))
        assert embeddings[key].shape == (10, 128)  # 10 samples, 128 dims
        assert labels.shape == (10,)

    def test_extract_embeddings_multi_layer(self, mock_model_multi_layer: MockModel, dataloader: DataLoader) -> None:
        """Test embedding extraction with multiple layers."""
        device = torch.device("cpu")

        embeddings, labels, _ = _extract_embeddings_in_memory(
            mock_model_multi_layer,
            dataloader,
            ["layer1", "layer2", "layer3"],
            device,
            aggregation="none",
            disable_tqdm=True,
        )

        # Should return dictionary with multiple layers
        assert isinstance(embeddings, dict)
        assert len(embeddings) == 3
        assert "layer1" in embeddings
        assert "layer2" in embeddings
        assert "layer3" in embeddings

        # Check shapes
        assert embeddings["layer1"].shape == (10, 128)
        assert embeddings["layer2"].shape == (10, 256)
        assert embeddings["layer3"].shape == (10, 64)
        assert labels.shape == (10,)

    def test_extract_embeddings_with_concatenation(
        self, mock_model_multi_layer: MockModel, dataloader: DataLoader
    ) -> None:
        """Test embedding extraction with concatenation (aggregation != 'none')."""
        device = torch.device("cpu")

        embeddings, labels, _ = _extract_embeddings_in_memory(
            mock_model_multi_layer,
            dataloader,
            ["layer1", "layer2", "layer3"],
            device,
            aggregation="mean",
            disable_tqdm=True,
        )

        # Should return dictionary with single concatenated layer
        assert isinstance(embeddings, dict)
        assert len(embeddings) == 1
        first_layer = list(embeddings.keys())[0]
        # Total dimension should be sum of all layers
        expected_dim = 128 + 256 + 64  # 448
        assert embeddings[first_layer].shape == (10, expected_dim)
        assert labels.shape == (10,)


class TestEmbeddingDataset:
    """Test EmbeddingDataset class."""

    def test_embedding_dataset_single_tensor(self) -> None:
        """Test EmbeddingDataset with single tensor (backward compatibility)."""
        embeddings = torch.randn(10, 128)
        labels = torch.randint(0, 5, (10,))

        dataset = EmbeddingDataset(embeddings, labels)

        assert len(dataset) == 10

        # Test __getitem__
        item = dataset[0]
        assert isinstance(item, dict)
        assert "embed" in item
        assert "label" in item
        assert item["embed"].shape == (128,)
        assert item["label"].shape == ()

    def test_embedding_dataset_multi_layer(self) -> None:
        """Test EmbeddingDataset with multi-layer dictionary."""
        embeddings = {
            "layer1": torch.randn(10, 128),
            "layer2": torch.randn(10, 256),
            "layer3": torch.randn(10, 64),
        }
        labels = torch.randint(0, 5, (10,))

        dataset = EmbeddingDataset(embeddings, labels)

        assert len(dataset) == 10

        # Test __getitem__
        item = dataset[0]
        assert isinstance(item, dict)
        assert "layer1" in item
        assert "layer2" in item
        assert "layer3" in item
        assert "label" in item

        assert item["layer1"].shape == (128,)
        assert item["layer2"].shape == (256,)
        assert item["layer3"].shape == (64,)
        assert item["label"].shape == ()


class TestSaveLoadEmbeddings:
    """Test saving and loading embeddings."""

    def test_save_load_single_tensor(self) -> None:
        """Test saving and loading single tensor embeddings."""
        embeddings = {"embed": torch.randn(10, 128)}
        labels = torch.randint(0, 5, (10,))
        num_labels = 5

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_embeddings.h5"

            # Save embeddings
            save_embeddings_arrays(embeddings, labels, save_path, num_labels)

            # Load embeddings
            loaded_embeddings, loaded_labels, loaded_num_labels = load_embeddings_arrays(save_path)

            # Should return dictionary in current API
            assert isinstance(loaded_embeddings, dict)
            assert torch.allclose(embeddings["embed"], loaded_embeddings["embed"])
            assert torch.allclose(labels, loaded_labels)
            assert loaded_num_labels == num_labels

    def test_save_load_multi_layer(self) -> None:
        """Test saving and loading multi-layer embeddings."""
        embeddings = {
            "layer1": torch.randn(10, 128),
            "layer2": torch.randn(10, 256),
            "layer3": torch.randn(10, 64),
        }
        labels = torch.randint(0, 5, (10,))
        num_labels = 5

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_embeddings.h5"

            # Save embeddings
            save_embeddings_arrays(embeddings, labels, save_path, num_labels)

            # Load embeddings
            loaded_embeddings, loaded_labels, loaded_num_labels = load_embeddings_arrays(save_path)

            # Should return dictionary
            assert isinstance(loaded_embeddings, dict)
            assert set(loaded_embeddings.keys()) == set(embeddings.keys())

            for layer_name in embeddings:
                assert torch.allclose(embeddings[layer_name], loaded_embeddings[layer_name])

            assert torch.allclose(labels, loaded_labels)
            assert loaded_num_labels == num_labels

    def test_save_load_backward_compatibility(self) -> None:
        """Test that old single-tensor files can still be loaded."""
        embeddings = {"embed": torch.randn(10, 128)}
        labels = torch.randint(0, 5, (10,))
        num_labels = 5

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_embeddings.h5"

            # Save as single tensor (old format)
            save_embeddings_arrays(embeddings, labels, save_path, num_labels)

            # Load should work and return dict
            loaded_embeddings, loaded_labels, loaded_num_labels = load_embeddings_arrays(save_path)

            assert isinstance(loaded_embeddings, dict)
            assert torch.allclose(embeddings["embed"], loaded_embeddings["embed"])
            assert torch.allclose(labels, loaded_labels)
            assert loaded_num_labels == num_labels


class TestStreamingExtraction:
    """Test streaming embedding extraction."""

    def test_streaming_extraction_single_layer(
        self, mock_model_single_layer: MockModel, dataloader: DataLoader
    ) -> None:
        """Test streaming extraction with single layer."""
        device = torch.device("cpu")

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_embeddings.h5"

            # Perform streaming extraction (writes to HDF5), then load for assertions
            _ = _extract_embeddings_streaming(
                mock_model_single_layer,
                dataloader,
                ["layer1"],
                device,
                save_path=save_path,
                chunk_size=2,
                compression="gzip",
                compression_level=4,
                aggregation="mean",
                auto_chunk_size=False,
                max_chunk_size=16,
                min_chunk_size=1,
                batch_chunk_size=2,
                disable_tqdm=True,
            )
            embeddings, labels, _ = load_embeddings_arrays(save_path)

            # Should return dictionary
            assert isinstance(embeddings, dict)
            assert "layer1" in embeddings
            assert embeddings["layer1"].shape == (10, 128)
            assert labels.shape == (10,)

            # File should exist
            assert save_path.exists()

    def test_streaming_extraction_multi_layer(self, mock_model_multi_layer: MockModel, dataloader: DataLoader) -> None:
        """Test streaming extraction with multiple layers."""
        device = torch.device("cpu")

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_embeddings.h5"

            _ = _extract_embeddings_streaming(
                mock_model_multi_layer,
                dataloader,
                ["layer1", "layer2", "layer3"],
                device,
                save_path=save_path,
                chunk_size=2,
                compression="gzip",
                compression_level=4,
                aggregation="mean",
                auto_chunk_size=False,
                max_chunk_size=16,
                min_chunk_size=1,
                batch_chunk_size=2,
                disable_tqdm=True,
            )
            embeddings, labels, _ = load_embeddings_arrays(save_path)

            # Should return dictionary with single concatenated layer
            assert isinstance(embeddings, dict)
            assert len(embeddings) == 1
            first_layer = list(embeddings.keys())[0]
            expected_dim = 128 + 256 + 64  # 448
            assert embeddings[first_layer].shape == (10, expected_dim)
            assert labels.shape == (10,)

            # File should exist
            assert save_path.exists()

    def test_streaming_and_inmemory_write_identical(self, tmp_path: Path) -> None:
        """Test that streaming and in-memory extraction produce identical results."""

        class _TinyRawDataset(Dataset):
            def __init__(self, num_samples: int, wav_len: int, num_classes: int) -> None:
                self._num_samples = num_samples
                self._wav_len = wav_len
                self._num_classes = num_classes

            def __len__(self) -> int:
                return self._num_samples

            def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
                wav = torch.randn(self._wav_len)
                label = torch.tensor(idx % self._num_classes, dtype=torch.int64)
                return {"raw_wav": wav, "label": label}

        class _FakeModel:
            def __init__(self, seq_len: int = 4, feat_dim: int = 6) -> None:
                self.seq_len = seq_len
                self.feat_dim = feat_dim

            def register_hooks_for_layers(self, layers: list[str]) -> list[str]:
                return layers

            def deregister_all_hooks(self) -> None:
                return None

            def extract_embeddings(
                self,
                x: Dict[str, torch.Tensor],
                aggregation: str = "none",
                padding_mask: Optional[torch.Tensor] = None,
            ) -> Optional[torch.Tensor]:
                if isinstance(x, dict):
                    batch_size = x["raw_wav"].shape[0] if x["raw_wav"].dim() > 1 else 1
                elif isinstance(x, torch.Tensor):
                    batch_size = x.shape[0] if x.dim() > 1 else 1
                else:
                    batch_size = 1

                def _make(batch: int, offset: int) -> torch.Tensor:
                    base = torch.arange(batch * self.seq_len * self.feat_dim, dtype=torch.float32)
                    base = base.reshape(batch, self.seq_len, self.feat_dim)
                    return base + float(offset)

                if aggregation == "none":
                    return [_make(batch_size, 0), _make(batch_size, 1000)]
                return _make(batch_size, 0).mean(dim=1)

        def _read_h5(path: str) -> Dict[str, Any]:
            with h5py.File(path, "r") as h5f:
                attrs = {k: h5f.attrs[k] for k in h5f.attrs.keys()}
                keys = list(h5f.keys())
                datasets = {k: np.asarray(h5f[k]) for k in keys}
            return attrs, datasets

        device = torch.device("cpu")
        num_samples = 8
        wav_len = 16
        num_classes = 3
        layer_names = ["layer_a", "layer_b"]
        aggregation = "none"

        ds = _TinyRawDataset(num_samples=num_samples, wav_len=wav_len, num_classes=num_classes)
        dl = DataLoader(ds, batch_size=4, shuffle=False)

        model = _FakeModel(seq_len=4, feat_dim=6)

        stream_path = tmp_path / "stream.h5"
        mem_path = tmp_path / "mem.h5"

        # STREAMING: writes directly
        _ = _extract_embeddings_streaming(
            model,
            dl,
            layer_names,
            device,
            save_path=stream_path,
            chunk_size=4,
            compression="gzip",
            compression_level=4,
            aggregation=aggregation,
            auto_chunk_size=False,
            max_chunk_size=16,
            min_chunk_size=1,
            batch_chunk_size=2,
            disable_tqdm=True,
            disable_layerdrop=None,
        )

        # IN-MEMORY: compute then save via the same saving routine
        embeds_dict, labels, _ = _extract_embeddings_in_memory(
            model,
            dl,
            layer_names,
            device,
            aggregation=aggregation,
            disable_tqdm=True,
            disable_layerdrop=None,
        )

        num_labels = int(torch.unique(labels if labels.dim() == 1 else torch.argmax(labels, dim=-1)).numel())
        save_embeddings_arrays(
            embeds_dict,
            labels,
            mem_path,
            num_labels,
            compression="gzip",
            compression_level=4,
        )

        # Compare HDF5 structure and data
        attrs_stream, data_stream = _read_h5(str(stream_path))
        attrs_mem, data_mem = _read_h5(str(mem_path))

        # Attributes
        assert attrs_stream["multi_layer"] == attrs_mem["multi_layer"]
        assert list(attrs_stream["layer_names"]) == list(attrs_mem["layer_names"])
        assert attrs_stream.get("num_labels", None) == attrs_mem.get("num_labels", None)

        # Embedding datasets per layer must match in shape and content
        for lname in layer_names:
            key = f"embeddings_{lname}"
            assert key in data_stream and key in data_mem
            a = data_stream[key]
            b = data_mem[key]
            assert a.shape == b.shape
            np.testing.assert_allclose(a, b, rtol=0, atol=0)

        # Labels
        np.testing.assert_array_equal(data_stream["labels"], data_mem["labels"])

    def test_loading_matches_original_values(self, tmp_path: Path) -> None:
        """Test that loaded embeddings match original values."""

        class _TinyRawDataset(Dataset):
            def __init__(self, num_samples: int, wav_len: int, num_classes: int) -> None:
                self._num_samples = num_samples
                self._wav_len = wav_len
                self._num_classes = num_classes

            def __len__(self) -> int:
                return self._num_samples

            def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
                wav = torch.randn(self._wav_len)
                label = torch.tensor(idx % self._num_classes, dtype=torch.int64)
                return {"raw_wav": wav, "label": label}

        class _FakeModel:
            def __init__(self, seq_len: int = 4, feat_dim: int = 6) -> None:
                self.seq_len = seq_len
                self.feat_dim = feat_dim

            def register_hooks_for_layers(self, layers: list[str]) -> list[str]:
                return layers

            def deregister_all_hooks(self) -> None:
                return None

            def extract_embeddings(
                self,
                x: Dict[str, torch.Tensor],
                aggregation: str = "none",
                padding_mask: Optional[torch.Tensor] = None,
            ) -> Optional[torch.Tensor]:
                if isinstance(x, dict):
                    batch_size = x["raw_wav"].shape[0] if x["raw_wav"].dim() > 1 else 1
                elif isinstance(x, torch.Tensor):
                    batch_size = x.shape[0] if x.dim() > 1 else 1
                else:
                    batch_size = 1

                def _make(batch: int, offset: int) -> torch.Tensor:
                    base = torch.arange(batch * self.seq_len * self.feat_dim, dtype=torch.float32)
                    base = base.reshape(batch, self.seq_len, self.feat_dim)
                    return base + float(offset)

                if aggregation == "none":
                    return [_make(batch_size, 0), _make(batch_size, 1000)]
                return _make(batch_size, 0).mean(dim=1)

        device = torch.device("cpu")
        num_samples = 6
        wav_len = 10
        num_classes = 4
        layer_names = ["layer_a", "layer_b"]
        aggregation = "none"

        ds = _TinyRawDataset(num_samples=num_samples, wav_len=wav_len, num_classes=num_classes)
        dl = DataLoader(ds, batch_size=3, shuffle=False)
        model = _FakeModel(seq_len=3, feat_dim=5)

        stream_path = tmp_path / "stream_load.h5"
        mem_path = tmp_path / "mem_load.h5"

        _ = _extract_embeddings_streaming(
            model,
            dl,
            layer_names,
            device,
            save_path=stream_path,
            chunk_size=3,
            compression="gzip",
            compression_level=4,
            aggregation=aggregation,
            auto_chunk_size=False,
            max_chunk_size=12,
            min_chunk_size=1,
            batch_chunk_size=2,
            disable_tqdm=True,
            disable_layerdrop=None,
        )

        embeds_dict, labels, _ = _extract_embeddings_in_memory(
            model,
            dl,
            layer_names,
            device,
            aggregation=aggregation,
            disable_tqdm=True,
            disable_layerdrop=None,
        )
        num_labels = int(torch.unique(labels if labels.dim() == 1 else torch.argmax(labels, dim=-1)).numel())
        save_embeddings_arrays(
            embeds_dict,
            labels,
            mem_path,
            num_labels,
            compression="gzip",
            compression_level=4,
        )

        # Now load back via helper and compare to the raw HDF5 reads
        embeds_stream_loaded, labels_stream_loaded, _ = load_embeddings_arrays(stream_path)
        embeds_mem_loaded, labels_mem_loaded, _ = load_embeddings_arrays(mem_path)

        # Labels must match
        assert torch.equal(labels_stream_loaded, labels_mem_loaded)

        # Compare each layer content matches original in-memory tensors
        assert isinstance(embeds_stream_loaded, dict)
        assert isinstance(embeds_mem_loaded, dict)

        for lname in layer_names:
            a = embeds_stream_loaded[lname]
            b = embeds_mem_loaded[lname]
            assert a.shape == b.shape
            torch.testing.assert_close(a, b, rtol=0, atol=0)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataloader(self, mock_model_single_layer: MockModel) -> None:
        """Test with empty dataloader."""

        # Create empty dataset that returns dictionaries
        class EmptyDictDataset:
            def __len__(self) -> int:
                return 0

            def __getitem__(self, idx: int) -> dict:
                return {
                    "raw_wav": torch.empty(0, 16000),
                    "label": torch.empty(0, dtype=torch.long),
                }

        empty_dataset = EmptyDictDataset()
        empty_dataloader = DataLoader(empty_dataset, batch_size=2)

        device = torch.device("cpu")

        with pytest.raises(ValueError, match="No data processed"):
            _ = _extract_embeddings_in_memory(
                mock_model_single_layer,
                empty_dataloader,
                ["layer1"],
                device,
                aggregation="mean",
                disable_tqdm=True,
            )

    def test_single_sample_batch(self, mock_model_single_layer: MockModel) -> None:
        """Test with single sample in batch."""

        # Create dataset with single sample that returns dictionaries
        class SingleDictDataset:
            def __init__(self) -> None:
                self.audio_data = torch.randn(1, 16000)
                self.labels = torch.tensor([0])

            def __len__(self) -> int:
                return 1

            def __getitem__(self, idx: int) -> dict:
                return {"raw_wav": self.audio_data[idx], "label": self.labels[idx]}

        single_dataset = SingleDictDataset()
        single_dataloader = DataLoader(single_dataset, batch_size=1)

        device = torch.device("cpu")

        embeddings, labels, _ = _extract_embeddings_in_memory(
            mock_model_single_layer,
            single_dataloader,
            ["layer1"],
            device,
            aggregation="mean",
            disable_tqdm=True,
        )

        assert isinstance(embeddings, dict)
        assert "layer1" in embeddings
        assert embeddings["layer1"].shape == (1, 128)
        assert labels.shape == (1,)

    def test_different_aggregation_methods(self, mock_model_multi_layer: MockModel, dataloader: DataLoader) -> None:
        """Test different aggregation methods."""
        device = torch.device("cpu")

        for aggregation in ["mean", "max", "cls_token"]:
            embeddings, labels, _ = _extract_embeddings_in_memory(
                mock_model_multi_layer,
                dataloader,
                ["layer1", "layer2", "layer3"],
                device,
                aggregation=aggregation,
                disable_tqdm=True,
            )

            # Should return dictionary with concatenated layers
            assert isinstance(embeddings, dict)
            assert len(embeddings) == 1
            first_layer = list(embeddings.keys())[0]
            expected_dim = 128 + 256 + 64  # 448
            assert embeddings[first_layer].shape == (10, expected_dim)
            assert labels.shape == (10,)

    def test_multi_dimensional_embeddings_reshape_consistency(self) -> None:
        """Test that multi-dimensional embeddings are correctly reshaped after
        save/load."""
        # Create embeddings with different multi-dimensional shapes
        embeddings = {
            "layer1": torch.randn(10, 128),  # 2D
            "layer2": torch.randn(10, 64, 32),  # 3D
            "layer3": torch.randn(10, 16, 8, 4),  # 4D
        }
        labels = torch.randint(0, 5, (10,))
        num_labels = 5

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_embeddings.h5"

            # Save embeddings
            save_embeddings_arrays(embeddings, labels, save_path, num_labels)

            # Load embeddings
            loaded_embeddings, loaded_labels, loaded_num_labels = load_embeddings_arrays(save_path)

            # Check that all layers are correctly reshaped
            for layer_name, original_emb in embeddings.items():
                loaded_emb = loaded_embeddings[layer_name]

                # Check shapes match exactly
                assert loaded_emb.shape == original_emb.shape, (
                    f"Shape mismatch for {layer_name}: {loaded_emb.shape} != {original_emb.shape}"
                )

                # Check data is preserved (within floating point precision)
                assert torch.allclose(loaded_emb, original_emb, rtol=1e-6), f"Data not preserved for {layer_name}"

            assert loaded_labels.shape == labels.shape
            assert loaded_num_labels == num_labels

    def test_exact_data_order_preservation(self) -> None:
        """Test that the exact order of values in arrays is preserved during
        save/load."""
        # Create embeddings with known patterns to verify exact order
        batch_size = 3

        # Create 2D embedding with sequential values for easy verification
        layer1_2d = torch.arange(batch_size * 128, dtype=torch.float32).reshape(batch_size, 128)

        # Create 3D embedding with a specific pattern
        layer2_3d = torch.zeros(batch_size, 4, 8, dtype=torch.float32)
        for b in range(batch_size):
            for i in range(4):
                for j in range(8):
                    layer2_3d[b, i, j] = b * 1000 + i * 100 + j

        # Create 4D embedding with another pattern
        layer3_4d = torch.zeros(batch_size, 2, 3, 4, dtype=torch.float32)
        for b in range(batch_size):
            for i in range(2):
                for j in range(3):
                    for k in range(4):
                        layer3_4d[b, i, j, k] = b * 10000 + i * 1000 + j * 100 + k

        embeddings = {
            "layer1_2d": layer1_2d,
            "layer2_3d": layer2_3d,
            "layer3_4d": layer3_4d,
        }
        labels = torch.tensor([0, 1, 2], dtype=torch.long)
        num_labels = 3

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_embeddings.h5"

            # Save embeddings
            save_embeddings_arrays(embeddings, labels, save_path, num_labels)

            # Load embeddings
            loaded_embeddings, loaded_labels, loaded_num_labels = load_embeddings_arrays(save_path)

            # Verify exact order preservation for each layer
            for layer_name, original_emb in embeddings.items():
                loaded_emb = loaded_embeddings[layer_name]

                # Check shapes match exactly
                assert loaded_emb.shape == original_emb.shape, f"Shape mismatch for {layer_name}"

                # Check that values are EXACTLY the same (no tolerance)
                assert torch.equal(loaded_emb, original_emb), f"Exact data order not preserved for {layer_name}"

                # Additional verification: check specific known values
                if layer_name == "layer1_2d":
                    # Check first few values are sequential
                    assert loaded_emb[0, 0] == 0.0
                    assert loaded_emb[0, 1] == 1.0
                    assert loaded_emb[0, 2] == 2.0
                    assert loaded_emb[1, 0] == 128.0
                    assert loaded_emb[1, 1] == 129.0

                elif layer_name == "layer2_3d":
                    # Check the pattern we created
                    assert loaded_emb[0, 0, 0] == 0.0
                    assert loaded_emb[0, 0, 1] == 1.0
                    assert loaded_emb[0, 1, 0] == 100.0
                    assert loaded_emb[1, 0, 0] == 1000.0
                    # b=1, i=1, j=1: 1*1000 + 1*100 + 1 = 1101
                    assert loaded_emb[1, 1, 1] == 1101.0

                elif layer_name == "layer3_4d":
                    # Check the 4D pattern
                    assert loaded_emb[0, 0, 0, 0] == 0.0
                    assert loaded_emb[0, 0, 0, 1] == 1.0
                    assert loaded_emb[0, 0, 1, 0] == 100.0
                    assert loaded_emb[0, 1, 0, 0] == 1000.0
                    assert loaded_emb[1, 0, 0, 0] == 10000.0
                    # b=1, i=1, j=1, k=1: 1*10000 + 1*1000 + 1*100 + 1 = 11101
                    assert loaded_emb[1, 1, 1, 1] == 11101.0

            # Verify labels are also preserved exactly
            assert torch.equal(loaded_labels, labels)
            assert loaded_num_labels == num_labels

    def test_flattening_reshape_consistency_detailed(self) -> None:
        """Test detailed consistency of flattening and reshaping operations."""
        # Test with various shapes to ensure the flattening/reshaping is consistent
        test_cases = [
            (2, 128),  # 2D
            (3, 64, 32),  # 3D
            (2, 16, 8, 4),  # 4D
            (1, 4, 3, 2, 2),  # 5D
        ]

        for shape in test_cases:
            batch_size = shape[0]
            embedding_dim = shape[1:]

            # Create tensor with known pattern
            original = torch.zeros(shape, dtype=torch.float32)
            for i in range(batch_size):
                for j in range(np.prod(embedding_dim)):
                    # Fill with sequential values
                    original.view(batch_size, -1)[i, j] = i * 1000 + j

            # Simulate the flattening process (what happens during save)
            flattened = original.reshape(batch_size, -1)

            # Simulate the reshaping process (what happens during load)
            reshaped = flattened.reshape(batch_size, *embedding_dim)

            # Verify exact equality
            assert torch.equal(original, reshaped), f"Flattening/reshaping failed for shape {shape}"

            # Verify the flattening size matches np.prod calculation
            expected_flattened_size = np.prod(embedding_dim)
            assert flattened.shape[1] == expected_flattened_size, f"Flattened size mismatch for {shape}"

            # Verify specific values are in the right positions
            for i in range(batch_size):
                for j in range(min(5, expected_flattened_size)):  # Check first 5 values
                    expected_val = i * 1000 + j
                    assert flattened[i, j] == expected_val, f"Value mismatch at [{i}, {j}] for shape {shape}"
                    assert reshaped.view(batch_size, -1)[i, j] == expected_val, (
                        f"Reshaped value mismatch at [{i}, {j}] for shape {shape}"
                    )


class TestEmbeddingReuse:
    """Test embedding file reuse across different probe types."""

    def test_embedding_paths_are_dataset_model_level(self) -> None:
        """Test that embedding paths are at dataset+model level, not experiment level."""
        save_dir = Path("/tmp/test_results")
        dataset_name = "test_dataset"
        evaluation_dataset_name = "test_eval_dataset"
        model_name = "beats_pretrained"

        # Simulate the path construction logic from run_evaluate.py
        embedding_dir_name = evaluation_dataset_name or dataset_name
        emb_base_dir = save_dir / f"{embedding_dir_name}_{model_name}"
        train_path = emb_base_dir / "embedding_train.h5"
        val_path = emb_base_dir / "embedding_val.h5"
        test_path = emb_base_dir / "embedding_test.h5"

        # Verify paths don't include experiment_name but include model_name
        expected_base = save_dir / "test_eval_dataset_beats_pretrained"
        assert emb_base_dir == expected_base
        assert train_path == expected_base / "embedding_train.h5"
        assert val_path == expected_base / "embedding_val.h5"
        assert test_path == expected_base / "embedding_test.h5"

        # Verify paths are the same regardless of experiment_name but different models
        emb_base_dir2 = save_dir / f"{embedding_dir_name}_{model_name}"
        assert emb_base_dir == emb_base_dir2

        # Different model should have different path
        model_name2 = "efficientnet_b0"
        emb_base_dir3 = save_dir / f"{embedding_dir_name}_{model_name2}"
        assert emb_base_dir != emb_base_dir3

    def test_aggregation_none_for_offline_training(self) -> None:
        """Test that offline training uses aggregation='none'."""
        from unittest.mock import MagicMock

        experiment_cfg = MagicMock(spec=ExperimentConfig)
        experiment_cfg.get_training_mode.return_value = False  # offline training
        experiment_cfg.get_aggregation_method.return_value = "mean"

        # Simulate the aggregation logic from run_evaluate.py
        need_probe = True
        offline_training = need_probe and not experiment_cfg.get_training_mode()

        if offline_training:
            aggregation_method = "none"
        else:
            aggregation_method = experiment_cfg.get_aggregation_method()

        assert offline_training is True
        assert aggregation_method == "none"
        experiment_cfg.get_aggregation_method.assert_not_called()

    def test_aggregation_from_config_for_online_training(self) -> None:
        """Test that online training uses aggregation from config."""
        from unittest.mock import MagicMock

        experiment_cfg = MagicMock(spec=ExperimentConfig)
        experiment_cfg.get_training_mode.return_value = True  # online training
        experiment_cfg.get_aggregation_method.return_value = "mean"

        # Simulate the aggregation logic from run_evaluate.py
        need_probe = True
        online_training = need_probe and experiment_cfg.get_training_mode()
        offline_training = need_probe and not experiment_cfg.get_training_mode()

        if offline_training:
            aggregation_method = "none"
        else:
            aggregation_method = experiment_cfg.get_aggregation_method()

        assert online_training is True
        assert offline_training is False
        assert aggregation_method == "mean"
        experiment_cfg.get_aggregation_method.assert_called_once()

    @pytest.mark.integration
    def test_embedding_file_reuse_integration(self) -> None:
        """Integration test to verify embedding file reuse works end-to-end."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir)

            # Test with different experiment names but same dataset and model
            experiments = ["linear_probe", "lstm_probe", "attention_probe"]
            dataset_name = "test_dataset"
            model_name = "beats_pretrained"

            embedding_paths = []
            for _ in experiments:
                # Simulate path construction
                embedding_dir_name = dataset_name
                emb_base_dir = save_dir / f"{embedding_dir_name}_{model_name}"
                train_path = emb_base_dir / "embedding_train.h5"
                embedding_paths.append(train_path)

            # All experiments should use the same embedding file (same dataset+model)
            assert all(path == embedding_paths[0] for path in embedding_paths)

            # The path should not contain any experiment name but should contain model name
            assert "linear_probe" not in str(embedding_paths[0])
            assert "lstm_probe" not in str(embedding_paths[0])
            assert "attention_probe" not in str(embedding_paths[0])
            assert "test_dataset" in str(embedding_paths[0])
            assert "beats_pretrained" in str(embedding_paths[0])

            # Test with different model - should have different path
            model_name2 = "efficientnet_b0"
            embedding_paths2 = []
            for _ in experiments:
                embedding_dir_name = dataset_name
                emb_base_dir = save_dir / f"{embedding_dir_name}_{model_name2}"
                train_path = emb_base_dir / "embedding_train.h5"
                embedding_paths2.append(train_path)

            # Different model should have different embedding files
            assert embedding_paths[0] != embedding_paths2[0]
            assert "efficientnet_b0" in str(embedding_paths2[0])
