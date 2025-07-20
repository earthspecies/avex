"""
Transform to generate textual captions (text_label) from one or more feature columns.

This mirrors esp-data's LabelFromFeature / MultiLabelFromFeatures but instead of
producing numeric class indices it keeps the raw text so that CLIP / CLAP style
models can use them directly.
"""

from __future__ import annotations

from typing import List, Literal

import pandas as pd
from esp_data.transforms import register_transform
from pydantic import BaseModel


class TextLabelFromFeaturesConfig(BaseModel):
    """Pydantic configuration for :class:`TextLabelFromFeatures`."""

    type: Literal["text_label_from_features"]

    # Columns to pull the textual information from.  Either a single string or a
    # list of strings.
    features: str | List[str]

    # Name of the generated column.  Default matches the expectation of our
    # Collater / TrainingStrategy.
    output_feature: str = "text_label"

    # If *False* and the output column already exists in the dataset an
    # AssertionError is raised.
    override: bool = False

    # Whether to always store the caption as a *list* (so that the collater can
    # randomly select one element).  If *False*, the individual parts will be
    # joined with *join_with*.
    listify: bool = True

    # Delimiter for joining when *listify* is False.
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
        # Normalise to list[str]
        self.features = list(features)
        self.output_feature = output_feature
        self.override = override
        self.listify = listify
        self.join_with = join_with

    # ---------------------------------------------------------------------
    # Factory helper
    # ---------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: TextLabelFromFeaturesConfig) -> "TextLabelFromFeatures":
        return cls(**cfg.model_dump(exclude=("type",)))

    # ---------------------------------------------------------------------
    # Transform logic
    # ---------------------------------------------------------------------
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:  # noqa: ANN001
        # Safety checks ----------------------------------------------------
        if self.output_feature in df and not self.override:
            raise AssertionError(
                f"Feature '{self.output_feature}' already exists in DataFrame. "
                "Set `override=True` to replace it."
            )

        missing_cols = [c for c in self.features if c not in df.columns]
        if missing_cols:
            # Not fatal: just log – esp_data codebase uses a logger named "esp_data".
            import logging

            logger = logging.getLogger("esp_data")
            logger.warning(
                "TextLabelFromFeatures: columns %s not present in DataFrame – "
                "they will be ignored.",
                missing_cols,
            )

        # Build caption for each row --------------------------------------
        def _row_to_caption(row: pd.Series) -> str:  # noqa: ANN001
            parts: List[str] = []
            for col in self.features:
                if col not in row:
                    continue

                val = row[col]

                # Handle pandas NA/None values safely
                try:
                    is_na = pd.isna(val)
                    # Handle case where is_na might be an array
                    if hasattr(is_na, "__len__") and not isinstance(is_na, str):
                        # If it's an array, check if all values are NA
                        is_na = is_na.all() if len(is_na) > 0 else True
                except (ValueError, TypeError):
                    # If pd.isna fails (e.g., on complex objects), assume it's not NA
                    is_na = False

                if is_na:
                    continue

                if isinstance(val, list):
                    parts.extend([str(v) for v in val if str(v).strip()])
                else:
                    s = str(val).strip()
                    if s:
                        parts.append(s)
            if not parts:
                return None
            if self.listify:
                return parts
            return self.join_with.join(parts)

        df[self.output_feature] = df.apply(_row_to_caption, axis="columns")

        # Drop rows without captions --------------------------------------
        df_clean = df.dropna(subset=[self.output_feature])

        # Minimal metadata – helpful for inspection/debugging --------------
        metadata = {
            "text_label_source": self.features,
            "output_feature": self.output_feature,
            "num_samples_with_text": len(df_clean),
        }

        return df_clean, metadata


# Register the transform with esp-data so that it can be instantiated from YAML
register_transform(TextLabelFromFeaturesConfig, TextLabelFromFeatures)
