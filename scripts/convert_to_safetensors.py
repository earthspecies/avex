"""
Convert PyTorch model checkpoints to safetensors format.

This script converts existing PyTorch checkpoints (.pt files) to safetensors format
(.safetensors files). It handles both model-only checkpoints (state_dict) and full
training checkpoints (with optimizer, scheduler, etc.).

Features:
- Embeds lightweight metadata (string-only) directly in the .safetensors file
- Optionally saves public JSON metadata in separate *_metadata.json (safe for public distribution)
- Automatically handles shared memory tensors by cloning when needed
- Verifies converted files by loading and comparing with original
- Supports local and cloud storage paths (GCS, S3/R2)

For full training checkpoints:
- Model weights → .safetensors (with embedded metadata)
- Public metadata → *_metadata.json (JSON format, safe for public release, optional with --save-metadata)

Usage:
    # Convert a single checkpoint
    python scripts/convert_to_safetensors.py path/to/checkpoint.pt

    # Convert all checkpoints in a directory
    python scripts/convert_to_safetensors.py path/to/checkpoints/ --recursive

    # Convert with custom output directory
    python scripts/convert_to_safetensors.py path/to/checkpoint.pt --output-dir path/to/output/

    # Convert cloud storage paths
    python scripts/convert_to_safetensors.py gs://bucket/path/to/checkpoint.pt

    # Add custom metadata to safetensors file
    python scripts/convert_to_safetensors.py path/to/checkpoint.pt --meta model_id=my_model --meta license=MIT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from avex.io import AnyPathT, PureCloudPath, anypath, exists, filesystem_from_path
from avex.utils import universal_torch_load

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def is_state_dict_only(checkpoint: dict[str, Any]) -> bool:
    """Check if checkpoint contains only model state dict.

    Parameters
    ----------
    checkpoint : dict[str, Any]
        The loaded checkpoint dictionary

    Returns
    -------
    bool
        True if checkpoint contains only state dict, False otherwise
    """
    # If it has model_state_dict, it's a full checkpoint
    if "model_state_dict" in checkpoint:
        return False

    # Otherwise, check if all values are tensors (state dict only)
    return all(isinstance(v, torch.Tensor) for v in checkpoint.values())


def extract_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Extract model state dict from checkpoint.

    Parameters
    ----------
    checkpoint : dict[str, Any]
        The loaded checkpoint dictionary

    Returns
    -------
    dict[str, torch.Tensor]
        The model state dictionary
    """
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    # If it's already a state dict, return as is
    return checkpoint


def build_safetensors_metadata(
    *,
    input_path: str,
    checkpoint: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build metadata dictionary for safetensors file.

    Safetensors metadata must be dict[str, str] only (strings only).
    This is for lightweight descriptive metadata, not training state.

    Parameters
    ----------
    input_path : str
        Path to the original checkpoint file (will be sanitized to basename only)
    checkpoint : dict[str, Any]
        The loaded checkpoint dictionary
    state_dict : dict[str, torch.Tensor]
        The model state dictionary
    extra : dict[str, str] | None, optional
        Additional metadata to include, by default None

    Returns
    -------
    dict[str, str]
        Metadata dictionary with string values only
    """
    # Sanitize source_checkpoint to avoid leaking internal paths
    # Only keep the basename (filename) for public distribution
    if input_path.startswith(("gs://", "s3://", "r2://")):
        # Cloud path - extract filename after last slash
        last_slash = input_path.rfind("/")
        source_checkpoint_name = input_path[last_slash + 1 :] if last_slash != -1 else input_path
    else:
        # Local path - use basename
        source_checkpoint_name = Path(input_path).name

    # Keep values small and strings only
    meta: dict[str, str] = {
        "format": "safetensors",
        "source_checkpoint": source_checkpoint_name,  # Sanitized: only filename
        "pytorch_version": torch.__version__,
        "python_version": platform.python_version(),
        "num_tensors": str(len(state_dict)),
    }

    # Add a sample of keys (first 20) as JSON string to help with debugging
    keys_sample = list(state_dict.keys())[:20]
    if keys_sample:
        meta["keys_sample"] = json.dumps(keys_sample)

    # Optional: preserve a few common fields if present (convert to strings)
    for k in ["epoch", "num_classes", "best_val_acc"]:
        if k in checkpoint and checkpoint[k] is not None:
            meta[k] = str(checkpoint[k])

    # Add extra metadata if provided
    if extra:
        meta.update(extra)

    # Defensive: ensure all values are strings
    meta = {k: (v if isinstance(v, str) else str(v)) for k, v in meta.items()}
    return meta


def extract_public_metadata(
    checkpoint: dict[str, Any],
    extra_metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Extract JSON-serializable metadata from checkpoint.

    This extracts only safe, public metadata that can be stored in JSON format.
    Training state (optimizer, scheduler, etc.) is excluded.

    Parameters
    ----------
    checkpoint : dict[str, Any]
        The loaded checkpoint dictionary
    extra_metadata : dict[str, str] | None, optional
        Additional metadata to include (e.g., from CLI --meta flags), by default None

    Returns
    -------
    dict[str, Any]
        JSON-serializable metadata dictionary
    """
    # Fields that are safe to include in public metadata
    allow = {"epoch", "best_val_acc", "loss", "accuracy", "num_classes"}
    out: dict[str, Any] = {}

    for k in allow:
        if k in checkpoint:
            v = checkpoint[k]
            # Make it JSON-safe
            if isinstance(v, (int, float, str, bool)) or v is None:
                out[k] = v
            elif isinstance(v, torch.Tensor):
                # Convert single-element tensors to scalars
                if v.numel() == 1:
                    out[k] = v.item()
                else:
                    out[k] = str(v.shape)
            else:
                out[k] = str(v)

    # Include extra metadata (e.g., from CLI --meta flags)
    # Note: Default license is set by caller in base_extra, so we don't need to set it here
    if extra_metadata:
        out.update(extra_metadata)

    return out


def verify_safetensors(
    safetensors_path: str | AnyPathT,
    original_state_dict: dict[str, torch.Tensor],
    verify_mode: str = "full",
) -> bool:
    """Verify that a safetensors file can be loaded and matches the original state dict.

    Parameters
    ----------
    safetensors_path : str | AnyPathT
        Path to the safetensors file to verify
    original_state_dict : dict[str, torch.Tensor]
        Original state dictionary to compare against
    verify_mode : str, optional
        Verification mode: "none" (file exists and metadata readable),
        "fast" (keys/shapes/dtypes), "full" (allclose), by default "full"

    Returns
    -------
    bool
        True if verification succeeds, False otherwise

    Raises
    ------
    FileNotFoundError
        If safetensors file doesn't exist
    RuntimeError
        If verification fails (keys don't match, values don't match, etc.)
    """
    safetensors_path = anypath(safetensors_path)

    if not exists(safetensors_path):
        raise FileNotFoundError(f"Safetensors file not found: {safetensors_path}")

    # Handle "none" mode - just verify file exists and metadata is readable
    if verify_mode == "none":
        try:
            if isinstance(safetensors_path, PureCloudPath):
                # For cloud paths, download to temp file first
                import tempfile

                fs = filesystem_from_path(safetensors_path)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as tmp_file:
                    tmp_path = Path(tmp_file.name)
                try:
                    fs.get(str(safetensors_path), str(tmp_path))
                    # Read metadata only
                    with safe_open(str(tmp_path), framework="pt", device="cpu") as f:
                        meta = f.metadata() or {}
                    if meta.get("format") != "safetensors":
                        raise RuntimeError("Embedded metadata missing or invalid: format != safetensors")
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink()
            else:
                # Local path - just read metadata
                with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
                    meta = f.metadata() or {}
                if meta.get("format") != "safetensors":
                    raise RuntimeError("Embedded metadata missing or invalid: format != safetensors")
            logger.info("✓ Verification successful: file exists and metadata is readable (none)")
            return True
        except Exception as e:
            raise RuntimeError(f"Verification failed (none mode): {e}") from e

    logger.info(f"Verifying safetensors file: {safetensors_path}")

    # Helper to read embedded metadata
    def _read_embedded_metadata(local_path: Path) -> dict[str, str]:
        """Read embedded metadata from safetensors file.

        Returns:
            Dictionary containing metadata from the safetensors file, or empty dict if none.
        """
        with safe_open(str(local_path), framework="pt", device="cpu") as f:
            return f.metadata() or {}

    # Load safetensors file and verify embedded metadata
    local_path: Path | None = None
    loaded_state_dict: dict[str, torch.Tensor] | None = None
    is_cloud_path = isinstance(safetensors_path, PureCloudPath)

    try:
        if is_cloud_path:
            # For cloud paths, download to temp file first
            import tempfile

            fs = filesystem_from_path(safetensors_path)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as tmp_file:
                tmp_path = Path(tmp_file.name)
            fs.get(str(safetensors_path), str(tmp_path))
            local_path = tmp_path
            loaded_state_dict = load_file(tmp_path)
        else:
            # Local path
            local_path = Path(safetensors_path)
            loaded_state_dict = load_file(local_path)

        # Verify embedded metadata
        if local_path:
            meta = _read_embedded_metadata(local_path)
            if meta.get("format") != "safetensors":
                raise RuntimeError("Embedded metadata missing or invalid: format != safetensors")
            if meta.get("num_tensors") and meta["num_tensors"] != str(len(original_state_dict)):
                raise RuntimeError(
                    f"Embedded metadata num_tensors mismatch: {meta['num_tensors']} != {len(original_state_dict)}"
                )
    except Exception as e:
        raise RuntimeError(f"Failed to load safetensors file: {e}") from e
    finally:
        # Clean up temp file for cloud paths (always, even on error)
        if is_cloud_path and local_path and local_path.exists():
            local_path.unlink()

    # Ensure loaded_state_dict was successfully assigned
    if loaded_state_dict is None:
        raise RuntimeError("Failed to load safetensors file: loaded_state_dict is None")

    # Verify keys match
    original_keys = set(original_state_dict.keys())
    loaded_keys = set(loaded_state_dict.keys())

    if original_keys != loaded_keys:
        missing_keys = original_keys - loaded_keys
        extra_keys = loaded_keys - original_keys
        error_msg = "Key mismatch in safetensors file:\n"
        if missing_keys:
            error_msg += f"  Missing keys: {sorted(missing_keys)}\n"
        if extra_keys:
            error_msg += f"  Extra keys: {sorted(extra_keys)}\n"
        raise RuntimeError(error_msg)

    # Verify values match (shape and values)
    mismatched_keys: list[str] = []
    for key in original_keys:
        original_tensor = original_state_dict[key]
        loaded_tensor = loaded_state_dict[key]

        # Check shape
        if original_tensor.shape != loaded_tensor.shape:
            mismatched_keys.append(f"{key} (shape mismatch: {original_tensor.shape} vs {loaded_tensor.shape})")
            continue

        # Check dtype
        if original_tensor.dtype != loaded_tensor.dtype:
            mismatched_keys.append(f"{key} (dtype mismatch: {original_tensor.dtype} vs {loaded_tensor.dtype})")
            continue

        # Only check values if verify_mode is "full"
        if verify_mode == "full":
            # Check values (use allclose for floating point comparison)
            # equal_nan=True handles NaN cases (NaN != NaN normally)
            if not torch.allclose(original_tensor, loaded_tensor, rtol=1e-5, atol=1e-8, equal_nan=True):
                mismatched_keys.append(f"{key} (value mismatch)")

    if mismatched_keys:
        error_msg = f"Mismatch in {len(mismatched_keys)} tensor(s):\n"
        for key in mismatched_keys[:10]:  # Show first 10 mismatches
            error_msg += f"  - {key}\n"
        if len(mismatched_keys) > 10:
            error_msg += f"  ... and {len(mismatched_keys) - 10} more\n"
        raise RuntimeError(error_msg)

    if verify_mode == "full":
        logger.info(f"✓ Verification successful: {len(original_keys)} tensors match (full)")
    elif verify_mode == "fast":
        logger.info(f"✓ Verification successful: {len(original_keys)} tensors match (fast: keys/shapes/dtypes)")
    else:
        logger.info("✓ Verification successful: file exists and can be loaded")
    return True


def convert_checkpoint_to_safetensors(
    input_path: str | AnyPathT,
    output_path: str | AnyPathT | None = None,
    save_metadata: bool = False,
    extra_metadata: dict[str, str] | None = None,
    verify_mode: str = "full",
) -> tuple[str, str | None]:
    """Convert a PyTorch checkpoint to safetensors format.

    Parameters
    ----------
    input_path : str | AnyPathT
        Path to input checkpoint file (.pt)
    output_path : str | AnyPathT | None, optional
        Path to output safetensors file. If None, replaces .pt with .safetensors
    save_metadata : bool, optional
        Whether to save metadata separately for full checkpoints, by default False
    extra_metadata : dict[str, str] | None, optional
        Additional metadata to include in safetensors file (must be string values), by default None
    verify_mode : str, optional
        Verification mode: "none" (file exists), "fast" (keys/shapes/dtypes), "full" (allclose), by default "full"

    Returns
    -------
    tuple[str, str | None]
        Tuple of (safetensors_path, metadata_path). metadata_path is None if no metadata
        was saved.

    Raises
    ------
    FileNotFoundError
        If input file doesn't exist
    ValueError
        If checkpoint format is invalid
    RuntimeError
        If verification fails after conversion
    """
    input_path = anypath(input_path)

    if not exists(input_path):
        raise FileNotFoundError(f"Checkpoint not found: {input_path}")

    logger.info(f"Loading checkpoint from: {input_path}")

    # Load checkpoint
    checkpoint = universal_torch_load(input_path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must be a dictionary, got {type(checkpoint)}")

    # Extract state dict
    state_dict = extract_state_dict(checkpoint)

    if not state_dict:
        raise ValueError("Checkpoint does not contain a valid state dictionary")

    # Convert tensors to CPU, detach, and make contiguous (safer for distribution)
    logger.debug("Converting tensors to CPU and making contiguous...")
    state_dict = {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}

    # Determine output path and model name
    # Extract model name from input path (without extension)
    input_path_str = str(input_path)
    if isinstance(input_path, PureCloudPath):
        # For cloud paths, extract name using string operations (not Path)
        # Find last slash to get filename
        last_slash = input_path_str.rfind("/")
        if last_slash != -1:
            filename = input_path_str[last_slash + 1 :]
        else:
            filename = input_path_str
        # Remove extension
        if filename.endswith(".pt"):
            model_name = filename[:-3]
        elif filename.endswith(".pth"):
            model_name = filename[:-4]
        else:
            model_name = filename
    else:
        model_name = Path(input_path).stem  # Removes extension (.pt or .pth)

    # Determine output directory (folder named after model)
    if output_path is None:
        # Create folder with model name in the same directory as input
        if isinstance(input_path, PureCloudPath):
            # For cloud paths, use string operations
            last_slash = input_path_str.rfind("/")
            if last_slash != -1:
                input_dir_str = input_path_str[:last_slash]
            else:
                input_dir_str = input_path_str
            output_dir_str = f"{input_dir_str}/{model_name}"
            output_path = anypath(f"{output_dir_str}/{model_name}.safetensors")
        else:
            input_dir = Path(input_path).parent
            output_dir = input_dir / model_name
            output_path = output_dir / f"{model_name}.safetensors"
    else:
        # If output_path is specified, extract directory and model name
        output_path = anypath(output_path)
        if isinstance(output_path, PureCloudPath):
            # For cloud paths, extract model name using string operations
            output_path_str = str(output_path)
            last_slash = output_path_str.rfind("/")
            if last_slash != -1:
                filename = output_path_str[last_slash + 1 :]
            else:
                filename = output_path_str
            model_name = filename.replace(".safetensors", "")
            # Ensure output is in a folder named after the model
            if last_slash != -1:
                output_dir_str = output_path_str[:last_slash]
            else:
                output_dir_str = output_path_str
            if not output_dir_str.endswith(f"/{model_name}"):
                output_dir_str = f"{output_dir_str}/{model_name}"
            output_path = anypath(f"{output_dir_str}/{model_name}.safetensors")
        else:
            output_path = Path(output_path)
            # If output_path is a directory, create model subfolder
            if output_path.is_dir() or not output_path.suffix:
                output_dir = output_path / model_name
                output_path = output_dir / f"{model_name}.safetensors"
            else:
                # Extract model name from filename
                model_name = output_path.stem.replace(".safetensors", "")
                output_dir = output_path.parent / model_name
                output_path = output_dir / f"{model_name}.safetensors"

    # Determine if this is a full checkpoint or just state dict
    is_full_checkpoint = not is_state_dict_only(checkpoint)
    json_metadata_path: str | None = None
    manifest_sha256: str | None = None

    # Build safetensors metadata (string-only, lightweight)
    base_extra = {"converted_by": "convert_to_safetensors.py", "license": "CC-BY-NC-SA"}
    if extra_metadata:
        base_extra.update(extra_metadata)  # CLI --meta can override default license

    # Use base_extra for both safetensors and JSON metadata (includes default license)
    safetensors_metadata = build_safetensors_metadata(
        input_path=str(input_path),
        checkpoint=checkpoint,
        state_dict=state_dict,
        extra=base_extra,
    )

    # Helper function to save with shared tensor handling
    def save_safetensors_with_retry(
        state_dict_to_save: dict[str, torch.Tensor],
        path: Path | str,
        *,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Save safetensors file, handling shared memory tensors by cloning if needed.

        Parameters
        ----------
        state_dict_to_save : dict[str, torch.Tensor]
            State dictionary to save
        path : Path | str
            Output path for safetensors file
        metadata : dict[str, str] | None, optional
            Metadata to embed in safetensors file (must be string values only), by default None

        Raises
        ------
        RuntimeError
            If saving fails even after cloning shared memory tensors
        """
        try:
            save_file(state_dict_to_save, path, metadata=metadata)
        except Exception as e:
            error_msg = str(e)
            # Check for shared memory tensor errors
            if any(phrase in error_msg.lower() for phrase in ["share memory", "duplicate memory", "tensors share"]):
                logger.warning("Detected shared memory tensors, cloning all tensors before saving...")
                # Clone all tensors to avoid shared memory issues
                cloned_dict = {key: tensor.clone() for key, tensor in state_dict_to_save.items()}
                try:
                    save_file(cloned_dict, path, metadata=metadata)
                    logger.info("Successfully saved after cloning shared tensors")
                except Exception as e2:
                    raise RuntimeError(f"Failed to save even after cloning tensors: {e2}. Original error: {e}") from e2
            else:
                raise

    # Ensure output directory exists
    if not isinstance(output_path, PureCloudPath):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {output_path.parent}")

    # Save state dict as safetensors
    logger.info(f"Saving model weights to: {output_path}")

    # Handle cloud paths
    if isinstance(output_path, PureCloudPath):
        # For cloud paths, we need to save to a temporary local file first
        import tempfile

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as tmp_file:
                tmp_path = Path(tmp_file.name)
            save_safetensors_with_retry(state_dict, tmp_path, metadata=safetensors_metadata)
            # Upload to cloud
            fs = filesystem_from_path(output_path)
            fs.put(str(tmp_path), str(output_path))
            logger.info(f"Uploaded safetensors to: {output_path}")
        finally:
            # Clean up temp file
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()
    else:
        # Local path (directory already created above)
        save_safetensors_with_retry(state_dict, output_path, metadata=safetensors_metadata)
        logger.info(f"Saved safetensors to: {output_path}")

        # Compute SHA256 for manifest (only for local paths)
        # This will be included in the manifest
        try:
            manifest_sha256 = compute_file_hash(output_path)
        except Exception as e:
            logger.debug(f"Could not compute file hash for manifest: {e}")
            manifest_sha256 = None

    # Save metadata if this is a full checkpoint
    if is_full_checkpoint and save_metadata:
        # Save public JSON metadata (safe for public distribution)
        # Use base_extra (includes default license) so CLI --meta flags are included in JSON
        public_metadata = extract_public_metadata(checkpoint, extra_metadata=base_extra)
        if public_metadata:
            # Determine JSON metadata path
            json_metadata_path_str = str(output_path)
            if json_metadata_path_str.endswith(".safetensors"):
                json_metadata_path_str = json_metadata_path_str[:-12] + "_metadata.json"
            else:
                json_metadata_path_str = json_metadata_path_str + "_metadata.json"
            json_metadata_path = json_metadata_path_str
            json_metadata_path_obj = anypath(json_metadata_path)

            logger.info(f"Saving public JSON metadata to: {json_metadata_path}")

            if isinstance(json_metadata_path_obj, PureCloudPath):
                # For cloud paths, save to temp file first
                import tempfile

                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as tmp_file:
                        tmp_path = Path(tmp_file.name)
                    # Write JSON content
                    tmp_path.write_text(json.dumps(public_metadata, indent=2, sort_keys=True))
                    # Upload to cloud
                    fs = filesystem_from_path(json_metadata_path_obj)
                    fs.put(str(tmp_path), str(json_metadata_path_obj))
                    logger.info(f"Uploaded public JSON metadata to: {json_metadata_path}")
                finally:
                    # Clean up temp file
                    if tmp_path and tmp_path.exists():
                        tmp_path.unlink()
            else:
                # Local path - use atomic write (temp file then rename)
                json_metadata_path_obj = Path(json_metadata_path_obj)
                json_metadata_path_obj.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = json_metadata_path_obj.with_suffix(".json.tmp")
                tmp_path.write_text(json.dumps(public_metadata, indent=2, sort_keys=True), encoding="utf-8")
                tmp_path.replace(json_metadata_path_obj)
                logger.info(f"Saved public JSON metadata to: {json_metadata_path}")

    # Verify the safetensors file (verify_safetensors now handles "none" mode)
    try:
        verify_safetensors(output_path, state_dict, verify_mode=verify_mode)
        logger.info(f"✓ Verification passed for: {output_path}")
    except Exception as e:
        logger.error(f"✗ Verification failed for {output_path}: {e}")
        raise RuntimeError(f"Verification failed after conversion: {e}") from e

    logger.info(f"Successfully converted checkpoint: {input_path} -> {output_path}")
    if json_metadata_path:
        logger.info(f"  Public JSON metadata saved to: {json_metadata_path}")

    # Return manifest_sha256 for use in manifest creation (local paths only)
    return (
        str(output_path),
        json_metadata_path,
        manifest_sha256 if not isinstance(output_path, PureCloudPath) else None,
    )


def find_checkpoint_files(
    directory: str | AnyPathT,
    recursive: bool = False,
) -> list[AnyPathT]:
    """Find all checkpoint files in a directory.

    Parameters
    ----------
    directory : str | AnyPathT
        Directory to search
    recursive : bool, optional
        Whether to search recursively, by default False

    Returns
    -------
    list[AnyPathT]
        List of checkpoint file paths

    Raises
    ------
    FileNotFoundError
        If directory doesn't exist
    """
    directory = anypath(directory)

    if not exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    checkpoint_files: list[AnyPathT] = []
    checkpoint_extensions = {".pt", ".pth"}

    if isinstance(directory, PureCloudPath):
        # For cloud paths, use filesystem API
        fs = filesystem_from_path(directory)
        dir_str = str(directory)
        # Ensure directory path ends with / for proper listing
        if not dir_str.endswith("/"):
            dir_str = dir_str + "/"

        def is_checkpoint_file(path: str) -> bool:
            return any(path.endswith(ext) for ext in checkpoint_extensions)

        # List files
        try:
            if recursive:
                # Find all files recursively
                files = fs.find(dir_str)
            else:
                # List files in directory only
                files = fs.ls(dir_str, detail=False)
                # Filter to only files (not directories)
                files = [f for f in files if not fs.isdir(f)]

            for file_path in files:
                file_path_str = str(file_path)
                if is_checkpoint_file(file_path_str):
                    checkpoint_files.append(anypath(file_path_str))
        except Exception as e:
            logger.warning(f"Error listing files in {directory}: {e}")
    else:
        # Local path
        directory = Path(directory)
        pattern = "**/*" if recursive else "*"
        for ext in checkpoint_extensions:
            checkpoint_files.extend(directory.glob(f"{pattern}{ext}"))

    return sorted(checkpoint_files)


def compute_file_hash(file_path: Path | str) -> str:
    """Compute SHA256 hash of a file.

    Parameters
    ----------
    file_path : Path | str
        Path to the file

    Returns
    -------
    str
        SHA256 hash in hexadecimal format
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def create_manifest(
    safetensors_path: str,
    json_metadata_path: str | None = None,
    output_dir: Path | None = None,
    safetensors_sha256: str | None = None,
) -> Path | None:
    """Create a manifest.json file for public distribution.

    Parameters
    ----------
    safetensors_path : str
        Path to the safetensors file
    json_metadata_path : str | None, optional
        Path to JSON metadata file if it exists, by default None
    output_dir : Path | None, optional
        Directory to save manifest.json. If None, uses safetensors file directory, by default None

    Returns
    -------
    Path | None
        Path to created manifest file, or None if creation failed
    """
    try:
        safetensors_path_obj = Path(safetensors_path)
        if not safetensors_path_obj.exists():
            logger.warning(f"Cannot create manifest: safetensors file not found: {safetensors_path}")
            return None

        # Determine output directory (use safetensors file's directory)
        if output_dir is None:
            output_dir = safetensors_path_obj.parent
        else:
            output_dir = Path(output_dir)

        # Compute file hashes and sizes
        safetensors_size = safetensors_path_obj.stat().st_size
        # Use provided hash if available (from conversion), otherwise compute it
        if safetensors_sha256:
            safetensors_hash = safetensors_sha256
        else:
            safetensors_hash = compute_file_hash(safetensors_path_obj)

        manifest_entry: dict[str, Any] = {
            "filename": safetensors_path_obj.name,
            "sha256": safetensors_hash,
            "size": safetensors_size,
        }

        # Add embedded metadata from safetensors file if available
        try:
            from safetensors import safe_open

            with safe_open(safetensors_path, framework="pt") as f:
                embedded_meta = f.metadata()
                if embedded_meta:
                    # Extract key fields for manifest
                    for key in ["model_id", "license", "format", "pytorch_version", "epoch"]:
                        if key in embedded_meta:
                            manifest_entry[key] = embedded_meta[key]
        except Exception as e:
            logger.debug(f"Could not read embedded metadata: {e}")

        # Add JSON metadata if available
        if json_metadata_path and Path(json_metadata_path).exists():
            json_metadata_path_obj = Path(json_metadata_path)
            json_size = json_metadata_path_obj.stat().st_size
            json_hash = compute_file_hash(json_metadata_path_obj)
            manifest_entry["metadata_file"] = json_metadata_path_obj.name
            manifest_entry["metadata_sha256"] = json_hash
            manifest_entry["metadata_size"] = json_size

        # Create manifest file
        manifest_path = output_dir / "manifest.json"
        manifest_data: dict[str, Any] = {"files": [manifest_entry]}

        # If manifest already exists, update or append to it
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    existing_manifest = json.load(f)
                if "files" in existing_manifest:
                    # Check if this file is already in manifest
                    existing_files = existing_manifest["files"]
                    file_index = next(
                        (i for i, f in enumerate(existing_files) if f.get("filename") == safetensors_path_obj.name),
                        None,
                    )
                    if file_index is not None:
                        # Update existing entry with new metadata (in case file was re-converted)
                        existing_files[file_index] = manifest_entry
                        manifest_data = existing_manifest
                        logger.debug(f"Updated existing manifest entry for {safetensors_path_obj.name}")
                    else:
                        # Append new entry
                        existing_files.append(manifest_entry)
                        manifest_data = existing_manifest
            except Exception as e:
                logger.warning(f"Could not read existing manifest, creating new one: {e}")

        # Write manifest (atomic write)
        tmp_manifest = manifest_path.with_suffix(".json.tmp")
        tmp_manifest.write_text(json.dumps(manifest_data, indent=2, sort_keys=True), encoding="utf-8")
        tmp_manifest.replace(manifest_path)
        logger.info(f"Created manifest: {manifest_path}")
        return manifest_path

    except Exception as e:
        logger.warning(f"Failed to create manifest: {e}")
        return None


def main() -> None:
    """Main entry point for the conversion script."""
    parser = argparse.ArgumentParser(
        description="Convert PyTorch checkpoints to safetensors format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input checkpoint file or directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for converted files (default: same as input)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for checkpoints in directories",
    )
    parser.add_argument(
        "--save-metadata",
        action="store_true",
        help="Save public JSON metadata separately for full checkpoints",
    )
    parser.add_argument(
        "--verify",
        type=str,
        choices=["none", "fast", "full"],
        default="full",
        help="Verification mode: none (file exists), fast (keys/shapes/dtypes), full (allclose) (default: full)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be converted without actually converting",
    )
    parser.add_argument(
        "--meta",
        action="append",
        metavar="KEY=VALUE",
        help=(
            "Add custom metadata to safetensors file (can be repeated, "
            "e.g., --meta model_id=my_model --meta license=MIT)"
        ),
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Don't create manifest.json file (manifest is created by default)",
    )

    args = parser.parse_args()

    input_path = anypath(args.input)

    # Check if input is a file or directory
    if not exists(input_path):
        logger.error(f"Input path does not exist: {input_path}")
        return

    # Determine if input is a directory
    is_directory = False
    if isinstance(input_path, PureCloudPath):
        fs = filesystem_from_path(input_path)
        try:
            # Try to list it as a directory
            fs.ls(str(input_path))
            is_directory = True
        except Exception:
            # Assume it's a file
            is_directory = False
    else:
        input_path = Path(input_path)
        is_directory = input_path.is_dir()

    # Get list of files to convert
    if is_directory:
        checkpoint_files = find_checkpoint_files(input_path, recursive=args.recursive)
        logger.info(f"Found {len(checkpoint_files)} checkpoint files to convert")
    else:
        checkpoint_files = [input_path]

    if not checkpoint_files:
        logger.warning("No checkpoint files found to convert")
        return

    # Convert each file
    successful = 0
    failed = 0

    for checkpoint_file in checkpoint_files:
        try:
            # Determine output path
            output_path = None
            if args.output_dir:
                output_dir = anypath(args.output_dir)
                if isinstance(checkpoint_file, PureCloudPath):
                    # For cloud paths, preserve relative structure
                    checkpoint_name = Path(str(checkpoint_file)).name
                    if checkpoint_name.endswith(".pt"):
                        checkpoint_name = checkpoint_name[:-3] + ".safetensors"
                    elif checkpoint_name.endswith(".pth"):
                        checkpoint_name = checkpoint_name[:-4] + ".safetensors"
                    else:
                        checkpoint_name = checkpoint_name + ".safetensors"
                    output_path = anypath(str(output_dir) + "/" + checkpoint_name)
                else:
                    checkpoint_file = Path(checkpoint_file)
                    checkpoint_name = checkpoint_file.name
                    if checkpoint_name.endswith(".pt"):
                        checkpoint_name = checkpoint_name[:-3] + ".safetensors"
                    elif checkpoint_name.endswith(".pth"):
                        checkpoint_name = checkpoint_name[:-4] + ".safetensors"
                    else:
                        checkpoint_name = checkpoint_name + ".safetensors"
                    output_path = anypath(str(output_dir) + "/" + checkpoint_name)

            # Parse extra metadata from CLI
            extra_metadata: dict[str, str] | None = None
            if args.meta:
                extra_metadata = {}
                for meta_arg in args.meta:
                    if "=" not in meta_arg:
                        logger.warning(f"Invalid metadata format (expected KEY=VALUE): {meta_arg}")
                        continue
                    key, value = meta_arg.split("=", 1)
                    extra_metadata[key.strip()] = value.strip()

            if args.dry_run:
                logger.info(f"[DRY RUN] Would convert: {checkpoint_file} -> {output_path or 'auto'}")
                if extra_metadata:
                    logger.info(f"[DRY RUN] With metadata: {extra_metadata}")
            else:
                safetensors_path, json_metadata_path, manifest_sha256 = convert_checkpoint_to_safetensors(
                    checkpoint_file,
                    output_path=output_path,
                    save_metadata=args.save_metadata,
                    extra_metadata=extra_metadata,
                    verify_mode=args.verify,
                )
                logger.info(f"✓ Converted: {checkpoint_file}")
                logger.info(f"  → {safetensors_path}")
                if json_metadata_path:
                    logger.info(f"  → {json_metadata_path}")

                # Create manifest by default (unless --no-manifest is specified)
                # Note: Only works for local paths, not cloud paths
                if not args.no_manifest:
                    safetensors_path_any = anypath(safetensors_path)
                    if isinstance(safetensors_path_any, PureCloudPath):
                        logger.info("Skipping manifest generation for cloud output (not supported)")
                    else:
                        # Manifest goes in the same directory as the safetensors file
                        manifest_output_dir = Path(safetensors_path).parent

                        # Use pre-computed SHA256 if available, otherwise compute it
                        safetensors_sha256_for_manifest = manifest_sha256
                        if not safetensors_sha256_for_manifest:
                            try:
                                safetensors_sha256_for_manifest = compute_file_hash(safetensors_path)
                            except Exception as e:
                                logger.debug(f"Could not compute SHA256 for manifest: {e}")

                        manifest_path = create_manifest(
                            safetensors_path,
                            json_metadata_path=json_metadata_path,
                            output_dir=manifest_output_dir,
                            safetensors_sha256=safetensors_sha256_for_manifest,
                        )
                        if manifest_path:
                            logger.info(f"  → {manifest_path}")

            successful += 1
        except Exception as e:
            logger.error(f"✗ Failed to convert {checkpoint_file}: {e}")
            failed += 1

    # Summary
    logger.info("=" * 60)
    logger.info(f"Conversion complete: {successful} successful, {failed} failed")
    if args.dry_run:
        logger.info("(Dry run - no files were actually converted)")


if __name__ == "__main__":
    main()
