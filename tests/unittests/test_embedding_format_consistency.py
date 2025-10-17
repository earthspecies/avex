from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from representation_learning.evaluation.embedding_utils import (
    _extract_embeddings_in_memory,
    _extract_embeddings_streaming,
    load_embeddings_arrays,
    save_embeddings_arrays,
)


class _TinyRawDataset(Dataset):
    def __init__(self, num_samples: int, wav_len: int, num_classes: int) -> None:
        self._num_samples = num_samples
        self._wav_len = wav_len
        self._num_classes = num_classes

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Return raw wav and integer label
        wav = torch.randn(self._wav_len)
        label = torch.tensor(idx % self._num_classes, dtype=torch.int64)
        return {"raw_wav": wav, "label": label}


class _FakeModel:
    def __init__(self, seq_len: int = 4, feat_dim: int = 6) -> None:
        self.seq_len = seq_len
        self.feat_dim = feat_dim

    def register_hooks_for_layers(self, layers: list[str]) -> list[str]:
        # In test, just echo back
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

        # Produce deterministic embeddings per layer
        def _make(batch: int, offset: int) -> torch.Tensor:
            base = torch.arange(
                batch * self.seq_len * self.feat_dim, dtype=torch.float32
            )
            base = base.reshape(batch, self.seq_len, self.feat_dim)
            return base + float(offset)

        if aggregation == "none":
            # Two layers
            return [_make(batch_size, 0), _make(batch_size, 1000)]
        # Single tensor case (not used in this test)
        return _make(batch_size, 0).mean(dim=1)


def _read_h5(path: str) -> Dict[str, Any]:
    with h5py.File(path, "r") as h5f:
        attrs = {k: h5f.attrs[k] for k in h5f.attrs.keys()}
        keys = list(h5f.keys())
        datasets = {k: np.asarray(h5f[k]) for k in keys}
    return attrs, datasets


def test_streaming_and_inmemory_write_identical(tmp_path: torch.Tensor) -> None:
    device = torch.device("cpu")
    num_samples = 8
    wav_len = 16
    num_classes = 3
    layer_names = ["layer_a", "layer_b"]
    aggregation = "none"

    # Build tiny dataloader
    ds = _TinyRawDataset(
        num_samples=num_samples, wav_len=wav_len, num_classes=num_classes
    )
    dl = DataLoader(ds, batch_size=4, shuffle=False)

    model = _FakeModel(seq_len=4, feat_dim=6)

    # Paths
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

    num_labels = int(
        torch.unique(
            labels if labels.dim() == 1 else torch.argmax(labels, dim=-1)
        ).numel()
    )
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
    # layer_names stored as list; convert to list of str
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


def test_loading_matches_original_values(tmp_path: torch.Tensor) -> None:
    device = torch.device("cpu")
    num_samples = 6
    wav_len = 10
    num_classes = 4
    layer_names = ["layer_a", "layer_b"]
    aggregation = "none"

    ds = _TinyRawDataset(
        num_samples=num_samples, wav_len=wav_len, num_classes=num_classes
    )
    dl = DataLoader(ds, batch_size=3, shuffle=False)
    model = _FakeModel(seq_len=3, feat_dim=5)

    # Create both files
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
    num_labels = int(
        torch.unique(
            labels if labels.dim() == 1 else torch.argmax(labels, dim=-1)
        ).numel()
    )
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
        # Shapes equal
        assert a.shape == b.shape
        # Since our fake model produces deterministic embeddings, both should be equal
        torch.testing.assert_close(a, b, rtol=0, atol=0)
