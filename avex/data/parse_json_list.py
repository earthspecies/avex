"""A dataset transform that parses a JSON-encoded string column into a real list column.

Some datasets (e.g. BirdSet's ``ebird_code_multilabel``) store multi-label targets as
JSON strings like ``'["amecro","daejun"]'`` or ``'[]'``. The multi-label label builder
(``labels_from_features``) expects actual lists, otherwise it treats the whole JSON string
as one opaque label and never drops the empty ``'[]'`` rows. This transform decodes those
strings into lists so downstream multi-label handling (and no-label dropping) works.

Registered at import time; import this module for the ``parse_json_list`` transform type to
be available (wired in :pyfile:`avex/data/dataset.py`).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from pydantic import BaseModel

try:  # installed package is ``alp_data`` (renamed from ``esp_data``)
    from alp_data.transforms import register_transform
except ModuleNotFoundError:  # pragma: no cover - fallback for the old package name
    from esp_data.transforms import register_transform

logger = logging.getLogger("alp_data")


def _polars_backend_cls() -> type:
    try:
        from alp_data.backends.polars_backend import PolarsBackend
    except ModuleNotFoundError:  # pragma: no cover
        from esp_data.backends.polars_backend import PolarsBackend
    return PolarsBackend


def _pandas_backend_cls() -> type:
    try:
        from alp_data.backends.pandas_backend import PandasBackend
    except ModuleNotFoundError:  # pragma: no cover
        from esp_data.backends.pandas_backend import PandasBackend
    return PandasBackend


class ParseJsonListConfig(BaseModel):
    """Config for the ``parse_json_list`` transform (decode a JSON-string list column)."""

    type: Literal["parse_json_list"]
    feature: str


class ParseJsonList:
    """Decode a JSON-string column into a list column, in-place (by name)."""

    def __init__(self, *, feature: str) -> None:
        self.feature = feature

    @classmethod
    def from_config(cls, cfg: ParseJsonListConfig) -> "ParseJsonList":
        return cls(**cfg.model_dump(exclude={"type"}))

    def __call__(self, backend: Any) -> tuple[Any, dict]:  # noqa: ANN401
        if "Polars" in type(backend).__name__:
            return self._apply_polars(backend)
        return self._apply_pandas(backend)

    def _apply_polars(self, backend: Any) -> tuple[Any, dict]:  # noqa: ANN401
        import polars as pl

        PolarsBackend = _polars_backend_cls()
        df = backend._df
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        if self.feature not in df.columns:
            logger.warning("ParseJsonList: column %r not present; pass-through", self.feature)
            return backend, {"parsed": 0}
        if df.schema[self.feature].base_type() == pl.List:
            return backend, {"parsed": 0}
        col = pl.col(self.feature)
        decoded = (
            pl.when(col.is_null() | (col.cast(pl.Utf8).str.strip_chars() == ""))
            .then(pl.lit("[]"))
            .otherwise(col.cast(pl.Utf8))
            .str.json_decode(pl.List(pl.Utf8))
            .alias(self.feature)
        )
        df_out = df.with_columns(decoded)
        return PolarsBackend(df_out, streaming=False), {"parsed": len(df_out)}

    def _apply_pandas(self, backend: Any) -> tuple[Any, dict]:  # noqa: ANN401
        PandasBackend = _pandas_backend_cls()
        df = backend._df
        if self.feature not in df.columns:
            logger.warning("ParseJsonList: column %r not present; pass-through", self.feature)
            return backend, {"parsed": 0}
        df = df.copy()

        def _parse(v: Any) -> list:  # noqa: ANN401
            if isinstance(v, list):
                return v
            if v is None or not isinstance(v, str) or v.strip() == "":
                return []
            try:
                out = json.loads(v)
                return out if isinstance(out, list) else [out]
            except (ValueError, TypeError):
                return []

        df[self.feature] = df[self.feature].apply(_parse)
        return PandasBackend(df, streaming=False), {"parsed": len(df)}


register_transform(ParseJsonListConfig, ParseJsonList)
