import io
import json
from typing import Any, Callable, Iterator

import librosa
import numpy as np
import soundfile as sf
import webdataset as wds
from esp_data.dataset import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.io import AnyPathT, anypath, filesystem_from_path


def audio_decoder(
    data: dict, dtype: str = "float32", format: str = "FLAC"
) -> dict[str, Any]:
    """Decode audio data from a WebDataset sample.

    Parameters
    ----------
    data: dict
        The sample containing audio data in WebDataset format
    format: str
        The format of the audio data (default: "FLAC")

    Returns
    -------
        dict: Dictionary containing the decoded audio data and metadata.

    Raises
    ------
    ValueError
        If the sample does not contain an audio key ending with .flac, .wav, etc.
    """
    audio_key = next((k for k in data if k.endswith(f".{format.lower()}")), None)
    if not audio_key:
        raise ValueError(
            "Sample must contain an audio key ending with .flac, .wav, etc."
        )

    audio_buffer = io.BytesIO(data[audio_key])
    audio_data, samplerate = sf.read(audio_buffer, dtype=dtype)

    # Reconstruct sample
    sample = {}
    sample["audio"] = audio_data
    sample["sample_rate"] = samplerate
    md = json.loads(data.get("metadata.json", "{}").decode("utf-8"))
    sample.update(md)

    return sample


def audio_encoder(
    sample: dict[str, Any],
    sample_rate: int = 16000,
    dtype: str = "float32",
    format: str = "FLAC",
) -> dict[str, Any]:
    """Encode audio data in the sample to a specific format.

    Parameters
    ----------
    sample: dict[str, Any]
        The sample containing audio data
    sample_rate: int
        The sample rate of the audio data
    dtype: str
        The data type of the audio data (default: "float32")
    format: str
        The format to encode the audio data to (e.g., "WAV", "FLAC", "OGG")
        Default is "FLAC".

    Returns
    -------
        dict: Dictionary containing the encoded audio data and metadata
            in the WebDataset format.

    Raises
    ------
        ValueError: If the sample does not contain an "audio" key
            with audio data.
    """
    if "audio" not in sample:
        raise ValueError("Sample must contain 'audio' key with audio data")

    data_out = {}
    audio_buffer = io.BytesIO()
    # Convert audio data to the specified format
    if isinstance(sample["audio"], (list, tuple)):
        # If audio is a list or tuple, convert to numpy array
        sample["audio"] = np.array(sample["audio"], dtype=dtype)
    elif isinstance(sample["audio"], np.ndarray):
        # If audio is already a numpy array, ensure it's the correct dtype
        sample["audio"] = sample["audio"].astype(dtype)

    sf.write(audio_buffer, sample["audio"], sample_rate, format=format)

    data_out[f"audio.{format.lower()}"] = audio_buffer.getvalue()

    # Add metadata (without audio)
    sample = {
        k: v for k, v in sample.items() if k != "audio"
    }  # Remove audio key from metadata
    data_out["metadata.json"] = json.dumps(sample, indent=2).encode("utf-8")
    return data_out


def json_encoder(
    sample: dict[str, Any],
    indent: int = 2,
) -> dict[str, Any]:
    """Encode a sample to JSON format.

    Parameters
    ----------
    sample: dict[str, Any]
        The sample to encode
    indent: int
        Indentation level for JSON (default: 2)

    Returns
    -------
        dict: Dictionary containing the encoded sample in JSON format.
    """
    json_data = json.dumps(sample, indent=indent).encode("utf-8")
    return {"sample.json": json_data}


def json_decoder(
    data: dict[str, Any],
) -> dict[str, Any]:
    """Decode a sample from JSON format.

    Parameters
    ----------
    data: dict[str, Any]
        The sample containing JSON data

    Returns
    -------
        dict: Dictionary containing the decoded sample.

    Raises
    ------
        ValueError: If the sample does not contain a "sample.json" key.
    """
    if "sample.json" not in data:
        raise ValueError("Sample must contain 'sample.json' key with JSON data")

    json_data = json.loads(data["sample.json"].decode("utf-8"))
    return json_data


def make_file_opener_for_wds(
    file_path: str | AnyPathT,
    mode: str = "wb",
    block_size: int = 1024 * 1024 * 100,
) -> Callable:
    """Make a file opener function for WebDataset.
    If local path, create parent dirs if needed.

    Arguments
    ---------
    file_path: str | AnyPathT
        The file path to open
    mode: str
        The mode in which to open the file (default: "wb")
    block_size: int
        Block size for WebDataset (default: 100 MB)

    Returns
    -------
        Callable: A function that opens the file in the specified mode
        or a file object if the path is local.
    """

    path_obj = anypath(file_path)

    if path_obj.is_local:
        # Local filesystem - create parent dirs if needed
        parent_dir = path_obj.parent
        parent_dir.mkdir(parents=True, exist_ok=True)
        return open(str(path_obj), mode=mode)
    else:
        # Remote filesystem (GCS, R2, etc.)
        fs = filesystem_from_path(str(path_obj))
        return fs.open(str(path_obj.no_prefix), mode=mode, block_size=block_size)


def load_webdataset(
    path: str | AnyPathT,
    file_pattern: str = "shard*tar",
    data_processor: Callable | None = None,
    shuffle_size: int | None = None,
    batch_size: int | None = None,
    shard_shuffle: bool = False,
    shard_shuffle_size: int = 1000,
    split_by_worker: bool = False,
    batch_collate_fn: Callable | None = None,
    seed: int | None = 42,
) -> wds.WebDataset:
    """Create a pipeline for loading the dataset

    Arguments
    ---------
    path: str | AnyPath
            Path to the directory where the sharded dataset will be stored or
            is already stored.
    file_pattern: str, optional
        Pattern to match the shard files.
    data_processor: Callable, optional
        Function to process the data.
    shuffle_size: int, optional
        Size of the shuffle buffer.
    batch_size: int, optional
        Batch size for processing audio files.
    shard_shuffle: bool, optional
        Whether to shuffle the shards.
    shard_shuffle_size: int, optional
        Size of the shuffle buffer for shards.
    split_by_worker: bool, optional
        Whether to split the dataset by worker.
    batch_collate_fn: Callable, optional
        Function to collate the batch.
    seed : int | None, optional
        Seed for shuffling. Defaults to True, random seed. If None, means no shuffling!

    Returns
    -------
        wds.WebDataset: WebDataset object

    Raises
    ------
    FileNotFoundError
        If no shard files are found in the specified path.
    """
    path = anypath(path)
    shard_files = list([str(s) for s in path.glob(file_pattern)])

    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {path}")

    webds_kwargs = {"shardshuffle": shard_shuffle_size if shard_shuffle else False}
    if shard_shuffle and seed is not None:
        webds_kwargs["seed"] = seed
    if split_by_worker:
        webds_kwargs["workersplitter"] = wds.split_by_worker

    webds = wds.WebDataset(shard_files, **webds_kwargs)

    if shuffle_size:
        webds = webds.shuffle(shuffle_size, seed=seed if seed is not None else 42)
    if data_processor:
        webds = webds.map(data_processor)
    if batch_size is not None:
        webds = webds.batched(batch_size, collation_fn=batch_collate_fn)

    return webds


class TarDatasetConfig(DatasetConfig):
    """Configuration for TarDataset.

    Parameters
    ----------
    split : str
        The split to load. One of info.split_paths keys.
    output_take_and_give : dict[str, str], optional
        A dictionary mapping the original column names to the new column names.
        It acts as a filter as well.
    sample_rate : int, optional
        The sample rate to which audio files should be resampled.
    data_root : str | AnyPathT, optional
        The root directory for the dataset. This is optionally appended to the
        path item of a sample in the dataset.
    shard_pattern : str, default="*.tar"
        The pattern to match the shard files in the dataset directory.
    """

    dataset_name: str = "tar_dataset"
    split: str = "train"
    data_root: str | AnyPathT
    output_take_and_give: dict[str, str] | None = None
    sample_rate: int | None = None
    shard_pattern: str = "*.tar"
    within_shard_shuffle: bool = False
    within_shard_shuffle_size: int = 1000
    across_shard_shuffle: bool = False
    across_shard_shuffle_size: int = 1000
    seed: int | None = 42
    split_by_worker: bool = False
    transformations: list[Any] = []


@register_dataset
class TarDataset(Dataset):
    info = DatasetInfo(
        name="tar_dataset",
        owner="repr-learning",
        version="0.1.0",
        description="A dataset for loading tar files with audio data.",
        split_paths={},
        sources="unknown",
        license="unknown",
    )

    def __init__(
        self,
        split: str,
        data_root: str | AnyPathT,
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = None,
        data_processor: Callable | None = audio_decoder,
        shard_pattern: str = "*.tar",
        within_shard_shuffle: bool = False,
        within_shard_shuffle_size: int = 1000,
        across_shard_shuffle: bool = False,
        across_shard_shuffle_size: int = 1000,
        split_by_worker: bool = False,
        seed: int | None = 42,
    ) -> None:
        """Initialize the TarDataset.

        Parameters
        ----------
        split : str
            The split to load. One of info.split_paths keys.
        output_take_and_give : dict[str, str], optional
            A dictionary mapping the original column names to the new column names.
            It acts as a filter as well.
        data_root : str | AnyPathT, optional
            The root directory for the dataset. This is required!
        data_processor : Callable, optional
            Function to process the data. Defaults to audio_decoder.
        """
        super().__init__(output_take_and_give)

        self.split = split
        self._data: wds.WebDataset = load_webdataset(
            path=data_root,
            file_pattern=shard_pattern,
            data_processor=data_processor,
            shuffle_size=within_shard_shuffle_size if within_shard_shuffle else None,
            batch_size=None,
            shard_shuffle=across_shard_shuffle,
            shard_shuffle_size=across_shard_shuffle_size,
            split_by_worker=split_by_worker,
            seed=seed,
        )
        self.data_root = anypath(data_root)
        self.sample_rate = sample_rate
        self.streaming = True
        self._columns = None

    @property
    def columns(self) -> list[str]:
        """Return the columns of the dataset."""
        # iter once over the webdataset to get the columns
        if not self._columns:
            self._columns = list(next(iter(self._data)).keys())
        return self._columns

    @property
    def available_splits(self) -> list[str]:
        """Return the available splits of the dataset."""
        return list(self.info.split_paths.keys())

    def _load(self) -> None:
        pass

    @classmethod
    def from_config(
        cls, dataset_config: DatasetConfig
    ) -> tuple["TarDataset", dict[str, Any]]:
        """Create a Dataset instance from a configuration dictionary.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Configuration dictionary containing dataset parametesf

        Returns
        -------
        tuple[Dataset, dict[str, Any]]
            A tuple containing the dataset instance and metadata.
            If the dataset_config contains transformations, they will be applied
            and the metadata will be returned as dict, otherwise an empty dict.
        """
        cfg = dataset_config.model_dump(exclude=("dataset_name", "transformations"))

        split = cfg.get("split")
        kwargs = {
            "split": split,
            "output_take_and_give": cfg.get("output_take_and_give"),
            "sample_rate": cfg.get("sample_rate"),
            "data_root": cfg.get("data_root"),
        }
        if "shard_pattern" in cfg:
            kwargs["shard_pattern"] = cfg["shard_pattern"]
        if "within_shard_shuffle" in cfg:
            kwargs["within_shard_shuffle"] = cfg["within_shard_shuffle"]
        if "within_shard_shuffle_size" in cfg:
            kwargs["within_shard_shuffle_size"] = cfg["within_shard_shuffle_size"]
        if "across_shard_shuffle" in cfg:
            kwargs["across_shard_shuffle"] = cfg["across_shard_shuffle"]
        if "across_shard_shuffle_size" in cfg:
            kwargs["across_shard_shuffle_size"] = cfg["across_shard_shuffle_size"]
        if "seed" in cfg:
            kwargs["seed"] = cfg["seed"]
        if "sample_rate" in cfg:
            kwargs["sample_rate"] = cfg["sample_rate"]
        if "split_by_worker" in cfg:
            kwargs["split_by_worker"] = cfg["split_by_worker"]

        ds = cls(**kwargs)

        if dataset_config.transformations:
            raise NotImplementedError(
                "Transformations are not supported for NatureLMAudio (tar) dataset."
            )
        return ds, {}

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Raises
        -------
        TypeError
            Always raises TypeError because this dataset is iterable and
            does not support length calculation.
        """
        raise TypeError(
            "Length is not defined for NatureLMAudio tar dataset "
            "because it is an iterable dataset"
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a sample from the dataset by index.

        Parameters
        ----------
        idx : int
            The index of the sample to retrieve.

        Raises
        -------
        TypeError
            Always raises TypeError because this dataset is iterable
            and does not support indexing.
        """
        raise TypeError(
            "Indexing is not supported for NatureLMAudio tar dataset. "
            "because it is an iterable dataset"
        )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over the dataset.

        Yields
        -------
        dict[str, Any]: A dictionary containing the audio sample and its metadata.
        """
        for row in self._data:
            # Resample audio if needed
            sr = row["sample_rate"]
            if self.sample_rate is not None and sr != self.sample_rate:
                audio = librosa.resample(
                    y=row["audio"],
                    orig_sr=sr,
                    target_sr=self.sample_rate,
                    scale=True,
                    res_type="kaiser_best",
                )
                row["audio"] = audio

            # If output_take_and_give is defined, filter the keys
            if self.output_take_and_give:
                item = {}
                for key, value in self.output_take_and_give.items():
                    item[value] = row[key]
            else:
                item = row

            yield item

    def apply_transformations(self, transformations: list[Any]) -> None:
        """Apply transformations to the dataset.

        Parameters
        ----------
        transformations : list[RegisteredTransformConfigs]
            A list of transformation configs to apply to each sample.
            This method is not implemented for NatureLMAudio dataset.

        Raises
        -------
        TypeError
            Always raises TypeError because transformations are not supported
            for NatureLMAudio (tar) dataset.
        """
        raise NotImplementedError(
            "Transformations are not supported for NatureLMAudio (tar) dataset."
        )

    def __str__(self) -> str:
        """Return a string representation of the dataset.

        Returns
        -------
        str
            A string representation of the dataset including its name, version,
            and basic statistics if data is loaded.
        """
        base_info = f"{self.info.name} (v{self.info.version})"

        return (
            f"{base_info}\n"
            f"Description: {self.info.description}\n"
            f"Sources: {', '.join(self.info.sources)}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
