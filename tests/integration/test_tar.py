import numpy as np

from representation_learning.data.dataset import (
    DatasetCollectionConfig,
    _build_datasets,
)


def test_iterate_build_datasets() -> None:
    path = "configs/data_configs/data_tar.yml"
    data_config = DatasetCollectionConfig.from_sources(path, cli_args=())
    train_ds, val_ds, test_ds = _build_datasets(
        data_config, postprocessors=[], label_type="arbit"
    )
    assert test_ds is None

    for sample in train_ds:
        assert isinstance(sample, dict)
        assert "audio" in sample
        assert isinstance(sample["audio"], np.ndarray)
        assert len(sample["audio"]) > 0
        assert "labels" in sample
        break

    for sample in val_ds:
        assert isinstance(sample, dict)
        assert "audio" in sample
        assert isinstance(sample["audio"], np.ndarray)
        assert len(sample["audio"]) > 0
        break
