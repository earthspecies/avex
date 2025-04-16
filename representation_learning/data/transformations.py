"""
Data transformation utilities for filtering and subsampling datasets.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np

from representation_learning.configs import FilterConfig, SubsampleConfig

class DataTransform(ABC):
    """Base class for data transformations."""
    
    @abstractmethod
    def __call__(self, data: Any) -> Any:
        """Apply the transformation to the data."""
        pass

class Filter(DataTransform):
    """Filter data based on property values."""
    
    def __init__(self, config: FilterConfig):
        """
        Initialize the filter.
        
        Args:
            config: Filter configuration
        """
        self.config = config
        self.values = set(config.values)
        
        if config.operation not in ["include", "exclude"]:
            raise ValueError(f"Operation must be 'include' or 'exclude', got {config.operation}")
    
    def __call__(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> Union[pd.DataFrame, Dict[str, Any]]:
        """Filter the data based on property values."""
        if isinstance(data, pd.DataFrame):
            return self._filter_dataframe(data)
        elif isinstance(data, dict):
            return self._filter_dict(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
    
    def _filter_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter a pandas DataFrame."""
        if self.config.operation == "include":
            return df[df[self.config.property].isin(self.values)]
        else:
            return df[~df[self.config.property].isin(self.values)]
    
    def _filter_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter a dictionary of data."""
        if self.config.operation == "include":
            return {k: v for k, v in data.items() if v[self.config.property] in self.values}
        else:
            return {k: v for k, v in data.items() if v[self.config.property] not in self.values}

class Subsample(DataTransform):
    """Subsample data based on property ratios."""
    
    def __init__(self, config: SubsampleConfig):
        """
        Initialize the subsampler.
        
        Args:
            config: Subsample configuration
        """
        self.config = config
        
        if config.operation != "subsample":
            raise ValueError(f"Operation must be 'subsample', got {config.operation}")
        
        if not all(0 <= r <= 1 for r in config.ratios.values()):
            raise ValueError("All ratios must be between 0 and 1")
    
    def __call__(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> Union[pd.DataFrame, Dict[str, Any]]:
        """Subsample the data based on property ratios."""
        if isinstance(data, pd.DataFrame):
            return self._subsample_dataframe(data)
        elif isinstance(data, dict):
            return self._subsample_dict(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
    
    def _subsample_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Subsample a pandas DataFrame."""
        result = []
        for value, ratio in self.config.ratios.items():
            value_data = df[df[self.config.property] == value]
            if ratio < 1.0:
                n_samples = int(len(value_data) * ratio)
                value_data = value_data.sample(n=n_samples, random_state=42)
            result.append(value_data)
        
        # Handle "other" value if specified
        if "other" in self.config.ratios:
            other_values = set(self.config.ratios.keys()) - {"other"}
            other_data = df[~df[self.config.property].isin(other_values)]
            if self.config.ratios["other"] < 1.0:
                n_samples = int(len(other_data) * self.config.ratios["other"])
                other_data = other_data.sample(n=n_samples, random_state=42)
            result.append(other_data)
        
        return pd.concat(result, ignore_index=True)
    
    def _subsample_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Subsample a dictionary of data."""
        result = {}
        for value, ratio in self.config.ratios.items():
            value_data = {k: v for k, v in data.items() if v[self.config.property] == value}
            if ratio < 1.0:
                n_samples = int(len(value_data) * ratio)
                keys = np.random.choice(list(value_data.keys()), size=n_samples, replace=False)
                value_data = {k: value_data[k] for k in keys}
            result.update(value_data)
        
        # Handle "other" value if specified
        if "other" in self.config.ratios:
            other_values = set(self.config.ratios.keys()) - {"other"}
            other_data = {k: v for k, v in data.items() if v[self.config.property] not in other_values}
            if self.config.ratios["other"] < 1.0:
                n_samples = int(len(other_data) * self.config.ratios["other"])
                keys = np.random.choice(list(other_data.keys()), size=n_samples, replace=False)
                other_data = {k: other_data[k] for k in keys}
            result.update(other_data)
        
        return result

def build_transforms(transform_configs: List[Dict[str, Any]]) -> List[DataTransform]:
    """
    Build a list of transformations from configuration.
    
    Args:
        transform_configs: List of transformation configurations
        
    Returns:
        List of DataTransform instances
    """
    transforms = []
    for config in transform_configs:
        transform_type = next(iter(config))
        params = config[transform_type]
        
        if transform_type == "filter":
            transforms.append(Filter(FilterConfig(**params)))
        elif transform_type == "subsample":
            transforms.append(Subsample(SubsampleConfig(**params)))
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")
    
    return transforms 