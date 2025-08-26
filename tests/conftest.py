import sys
import types

import torch

# -----------------------------------------------------------------------------
# Stub **optional** third-party packages so that imports succeed even when the
# actual libraries are not installed in the minimal CI environment.
# -----------------------------------------------------------------------------

# 1. bitsandbytes – only required when using the 8-bit Adam optimiser variant.
#    We replace it with a minimal stub exposing the attribute accessed by the
#    code-base (PagedAdamW8bit -> torch.optim.AdamW).
if "bitsandbytes" not in sys.modules:
    bnb_stub = types.ModuleType("bitsandbytes")
    optim_stub = types.ModuleType("optim")
    optim_stub.PagedAdamW8bit = torch.optim.AdamW  # type: ignore[attr-defined]
    bnb_stub.optim = optim_stub  # type: ignore[attr-defined]
    sys.modules["bitsandbytes"] = bnb_stub

# 2. google-cloud-storage – only needed for logging/remote paths.  Provide a
#    dummy `Client` class so `from google.cloud.storage.client import Client`
#    succeeds.
# if "google" not in sys.modules:
#     google_mod = types.ModuleType("google")
#     cloud_mod = types.ModuleType("google.cloud")
#     storage_mod = types.ModuleType("google.cloud.storage")
#     client_mod = types.ModuleType("google.cloud.storage.client")

#     class _DummyClient:  # pylint: disable=too-few-public-methods
#         """Bare-bones replacement for GCS Client used only for type checks."""

#         def __init__(self, *args, **kwargs) -> None:  # noqa: D401, ANN001, ANN002, ANN003, E501
#             pass

#     client_mod.Client = _DummyClient  # type: ignore[attr-defined]
#     # Register hierarchy in sys.modules so nested imports resolve correctly
#     sys.modules.update(
#         {
#             "google": google_mod,
#             "google.cloud": cloud_mod,
#             "google.cloud.storage": storage_mod,
#             "google.cloud.storage.client": client_mod,
#         }
#     )
#     google_mod.cloud = cloud_mod  # type: ignore[attr-defined]
#     cloud_mod.storage = storage_mod  # type: ignore[attr-defined]
#     storage_mod.client = client_mod  # type: ignore[attr-defined]

# 3. esp_data – the original codebase relies on a separate *esp-data* package
#    for path helpers (anypath, AnyPathT).  For unit tests we provide a
#    lightweight substitute that fulfils the import-time requirements.
# if "esp_data" not in sys.modules:
#     esp_mod = types.ModuleType("esp_data")
#     io_mod = types.ModuleType("esp_data.io")

#     class _DummyPath(str):  # type: ignore[misc]
#         """String subclass acting as a stand-in for AnyPathT."""

#         def __truediv__(self, other) -> "_DummyPath":  # noqa: D401, ANN001
#             return _DummyPath(f"{self}/{other}")

#         def exists(self) -> bool:  # noqa: D401
#             return False

#         @property
#         def is_cloud(self) -> bool:  # noqa: D401
#             return str(self).startswith(("gs://", "r2://", "s3://"))

#         @property
#         def is_local(self) -> bool:  # noqa: D401
#             return not self.is_cloud

#     def _anypath(p) -> _DummyPath:  # noqa: D401, ANN001
#         return _DummyPath(p)

#     # Set up the io module with the expected exports
#     io_mod.anypath = _anypath  # type: ignore[attr-defined]
#     io_mod.AnyPathT = _DummyPath  # type: ignore[attr-defined]

#     # Register hierarchy
#     sys.modules.update(
#         {
#             "esp_data": esp_mod,
#             "esp_data.io": io_mod,
#         }
#     )
#     esp_mod.io = io_mod  # type: ignore[attr-defined]

# No fixtures are defined at the moment – the stubs above are imported eagerly
# by Pytest before test collection, ensuring all subsequent imports succeed.
