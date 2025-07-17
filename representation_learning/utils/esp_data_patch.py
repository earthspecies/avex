"""
Temporary monkey patch for esp_data pandas indexing bug.

This is a throwaway file containing patches that can be applied when needed.
Currently not used - kept for reference if the patch is needed again.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def patch_esp_data() -> None:
    """Apply monkey patch to fix the pandas indexing bug in esp_data.transforms."""
    try:
        from esp_data.transforms.label_from_feature import LabelFromFeature

        def patched_call(self: Any, df: pd.DataFrame) -> tuple:  # noqa: ANN401
            """Patched version of LabelFromFeature.__call__ with fixed pandas indexing.

            Returns
            -------
            tuple
                (processed_dataframe, metadata_dict)

            Raises
            ------
            AssertionError
                If output feature exists and override is False
            """
            import logging

            logger = logging.getLogger("esp_data")

            if self.output_feature in df and not self.override:
                raise AssertionError(
                    f"Feature '{self.output_feature}' already exists in DataFrame. "
                    "Set `override=True` to replace it."
                )

            df_clean = df.dropna(subset=[self.feature])
            if len(df_clean) != len(df):
                logger.warning(
                    f"Dropped {len(df) - len(df_clean)} rows with {self.feature}=NaN"
                )

            # Auto-generate label_map if not provided (restore original logic)
            if self.label_map is None:
                uniques = sorted(df_clean[self.feature].unique())
                label_map = {lbl: idx for idx, lbl in enumerate(uniques)}
            else:
                label_map = self.label_map

            # Map feature values to labels using the label_map
            # Fixed: Using scalar indexing instead of list indexing to avoid
            # pandas broadcasting error
            mapped_labels = df_clean[self.feature].map(label_map)

            # Check for unmapped values
            if mapped_labels.isna().any():
                unmapped_values = df_clean.loc[
                    mapped_labels.isna(), self.feature
                ].unique()
                logger.warning(
                    f"Found {len(unmapped_values)} unmapped values in "
                    f"feature '{self.feature}': {unmapped_values}"
                )

            # Use scalar indexing instead of list indexing to fix
            # pandas broadcasting error
            df_clean.loc[:, self.output_feature] = mapped_labels

            metadata = {
                "label_feature": self.feature,
                "label_map": label_map,
                "num_classes": len(label_map),
            }

            return df_clean, metadata

        # Apply the patch
        LabelFromFeature.__call__ = patched_call
        print("Applied esp_data patch to fix pandas indexing bug")

    except ImportError as e:
        print(f"Could not apply esp_data patch: {e}")
