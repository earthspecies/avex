"""
Transform to generate textual captions (text_label) from one or more feature columns.

This mirrors esp-data's LabelFromFeature / MultiLabelFromFeatures but instead of
producing numeric class indices it keeps the raw text so that CLIP / CLAP style
models can use them directly.

Works with both Polars and Pandas backends from esp-data.
"""

from __future__ import annotations

import logging
from typing import Any, List, Literal

from esp_data.transforms import register_transform
from pydantic import BaseModel

logger = logging.getLogger("esp_data")


class TextLabelFromFeaturesConfig(BaseModel):
    """Pydantic configuration for :class:`TextLabelFromFeatures`."""

    type: Literal["text_label_from_features"]

    features: str | List[str]

    output_feature: str = "text_label"

    override: bool = False

    listify: bool = True

    join_with: str = ", "


class TextLabelFromFeatures:
    """Generate a textual caption from one or more source columns.

    Examples
    --------
    >>> cfg = TextLabelFromFeaturesConfig(
    ...     type="text_label_from_features",
    ...     features=["caption", "species_common", "labels"],
    ... )
    >>> transform = TextLabelFromFeatures.from_config(cfg)
    """

    def __init__(
        self,
        *,
        features: List[str],
        output_feature: str = "text_label",
        override: bool = False,
        listify: bool = True,
        join_with: str = ", ",
    ) -> None:
        self.features = list(features)
        self.output_feature = output_feature
        self.override = override
        self.listify = listify
        self.join_with = join_with

    @classmethod
    def from_config(cls, cfg: TextLabelFromFeaturesConfig) -> "TextLabelFromFeatures":
        return cls(**cfg.model_dump(exclude=("type",)))

    def _apply_polars(self, backend: Any) -> tuple[Any, dict]:  # noqa: ANN401
        """Polars implementation using struct + map_elements.

        Returns
        -------
        tuple[Any, dict]
            (resulting PolarsBackend, metadata dict).
        """
        import polars as pl

        df = backend._df
        if isinstance(df, pl.LazyFrame):
            df = df.collect()

        existing = [c for c in self.features if c in df.columns]
        if not existing:
            logger.warning("TextLabelFromFeatures: none of %s found in data", self.features)
            from esp_data.backends.polars_backend import PolarsBackend

            empty_df = df.head(0).with_columns(pl.lit(None).alias(self.output_feature))
            return PolarsBackend(empty_df, streaming=False), {
                "text_label_source": self.features,
                "output_feature": self.output_feature,
                "num_samples_with_text": 0,
            }

        listify = self.listify
        join_with = self.join_with

        def _process_struct(row: dict) -> list[str] | str | None:
            parts: list[str] = []
            for col in existing:
                val = row.get(col)
                if val is None:
                    continue
                s = str(val).strip()
                if s:
                    parts.append(s)
            if not parts:
                return None
            return parts if listify else join_with.join(parts)

        return_dtype = pl.List(pl.Utf8) if self.listify else pl.Utf8

        df_result = df.with_columns(
            pl.struct(existing).map_elements(_process_struct, return_dtype=return_dtype).alias(self.output_feature)
        )
        df_final = df_result.drop_nulls(subset=[self.output_feature])

        from esp_data.backends.polars_backend import PolarsBackend

        result_backend = PolarsBackend(df_final, streaming=False)
        metadata = {
            "text_label_source": self.features,
            "output_feature": self.output_feature,
            "num_samples_with_text": len(df_final),
        }
        return result_backend, metadata

    def _apply_pandas(self, backend: Any) -> tuple[Any, dict]:  # noqa: ANN401
        """Pandas implementation (fallback).

        Returns
        -------
        tuple[Any, dict]
            (resulting PandasBackend, metadata dict).
        """
        import pandas as pd

        df = backend._df

        existing = [c for c in self.features if c in df.columns]
        if not existing:
            logger.warning("TextLabelFromFeatures: none of %s found in data", self.features)

        def _row_to_caption(row: pd.Series) -> list[str] | str | None:
            parts: List[str] = []
            for col in existing:
                val = row.get(col)
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    continue
                if isinstance(val, list):
                    parts.extend([str(v) for v in val if str(v).strip()])
                else:
                    s = str(val).strip()
                    if s:
                        parts.append(s)
            if not parts:
                return None
            return parts if self.listify else self.join_with.join(parts)

        df = df.copy()
        df[self.output_feature] = df.apply(_row_to_caption, axis="columns")
        df_clean = df.dropna(subset=[self.output_feature])

        from esp_data.backends.pandas_backend import PandasBackend

        result_backend = PandasBackend(df_clean)
        metadata = {
            "text_label_source": self.features,
            "output_feature": self.output_feature,
            "num_samples_with_text": len(df_clean),
        }
        return result_backend, metadata

    def __call__(self, backend: Any) -> tuple[Any, dict]:  # noqa: ANN401
        """Apply transform to an esp-data backend (Polars or Pandas).

        Returns
        -------
        tuple[Any, dict]
            (resulting backend with the new column, metadata dict).

        Raises
        ------
        AssertionError
            If the output feature already exists and ``override=False``.
        """
        if self.output_feature in backend.columns and not self.override:
            raise AssertionError(f"Feature '{self.output_feature}' already exists. Set `override=True` to replace it.")

        missing_cols = [c for c in self.features if c not in backend.columns]
        if missing_cols:
            logger.warning(
                "TextLabelFromFeatures: columns %s not present – they will be ignored.",
                missing_cols,
            )

        # Dispatch to backend-specific implementation
        backend_type = type(backend).__name__
        if "Polars" in backend_type:
            return self._apply_polars(backend)
        return self._apply_pandas(backend)


# Register the transform with esp-data so that it can be instantiated from YAML
register_transform(TextLabelFromFeaturesConfig, TextLabelFromFeatures)
