"""
Unit tests for data transformations.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from representation_learning.data.transformations import Filter, Subsample, build_transforms
from representation_learning.configs import FilterConfig, SubsampleConfig

def test_filter_dataframe():
    """Test filtering a pandas DataFrame."""
    # Create test data
    df = pd.DataFrame({
        'source': ['xeno-canto', 'iNaturalist', 'Watkins', 'other'],
        'class': ['birds', 'mammals', 'amphibians', 'reptiles'],
        'value': [1, 2, 3, 4]
    })
    
    # Test include operation
    config = FilterConfig(
        property='source',
        values=['xeno-canto', 'iNaturalist'],
        operation='include'
    )
    filter_transform = Filter(config)
    filtered_df = filter_transform(df)
    
    assert len(filtered_df) == 2
    assert set(filtered_df['source']) == {'xeno-canto', 'iNaturalist'}
    
    # Test exclude operation
    config = FilterConfig(
        property='source',
        values=['xeno-canto', 'iNaturalist'],
        operation='exclude'
    )
    filter_transform = Filter(config)
    filtered_df = filter_transform(df)
    
    assert len(filtered_df) == 2
    assert set(filtered_df['source']) == {'Watkins', 'other'}

def test_filter_dict():
    """Test filtering a dictionary of data."""
    # Create test data
    data = {
        '1': {'source': 'xeno-canto', 'class': 'birds', 'value': 1},
        '2': {'source': 'iNaturalist', 'class': 'mammals', 'value': 2},
        '3': {'source': 'Watkins', 'class': 'amphibians', 'value': 3},
        '4': {'source': 'other', 'class': 'reptiles', 'value': 4}
    }
    
    # Test include operation
    config = FilterConfig(
        property='source',
        values=['xeno-canto', 'iNaturalist'],
        operation='include'
    )
    filter_transform = Filter(config)
    filtered_data = filter_transform(data)
    
    assert len(filtered_data) == 2
    assert set(k for k, v in filtered_data.items() if v['source'] in ['xeno-canto', 'iNaturalist']) == {'1', '2'}
    
    # Test exclude operation
    config = FilterConfig(
        property='source',
        values=['xeno-canto', 'iNaturalist'],
        operation='exclude'
    )
    filter_transform = Filter(config)
    filtered_data = filter_transform(data)
    
    assert len(filtered_data) == 2
    assert set(k for k, v in filtered_data.items() if v['source'] in ['Watkins', 'other']) == {'3', '4'}

def test_subsample_dataframe():
    """Test subsampling a pandas DataFrame."""
    # Create test data with known class distribution
    df = pd.DataFrame({
        'class': ['birds'] * 100 + ['mammals'] * 100 + ['amphibians'] * 100,
        'value': range(300)
    })
    
    # Test subsampling with different ratios
    config = SubsampleConfig(
        property='class',
        operation='subsample',
        ratios={
            'birds': 0.5,
            'mammals': 0.3,
            'amphibians': 0.7
        }
    )
    subsample_transform = Subsample(config)
    subsampled_df = subsample_transform(df)
    
    # Check that the ratios are approximately correct
    class_counts = subsampled_df['class'].value_counts()
    assert abs(class_counts['birds'] / 100 - 0.5) < 0.1
    assert abs(class_counts['mammals'] / 100 - 0.3) < 0.1
    assert abs(class_counts['amphibians'] / 100 - 0.7) < 0.1
    
    # Test with 'other' class
    config = SubsampleConfig(
        property='class',
        operation='subsample',
        ratios={
            'birds': 0.5,
            'other': 0.2
        }
    )
    subsample_transform = Subsample(config)
    subsampled_df = subsample_transform(df)
    
    # Check that 'other' class (mammals + amphibians) is subsampled correctly
    other_count = len(subsampled_df[subsampled_df['class'].isin(['mammals', 'amphibians'])])
    assert abs(other_count / 200 - 0.2) < 0.1

def test_subsample_dict():
    """Test subsampling a dictionary of data."""
    # Create test data with known class distribution
    data = {
        str(i): {'class': 'birds', 'value': i} for i in range(100)
    }
    data.update({
        str(i): {'class': 'mammals', 'value': i} for i in range(100, 200)
    })
    data.update({
        str(i): {'class': 'amphibians', 'value': i} for i in range(200, 300)
    })
    
    # Test subsampling with different ratios
    config = SubsampleConfig(
        property='class',
        operation='subsample',
        ratios={
            'birds': 0.5,
            'mammals': 0.3,
            'amphibians': 0.7
        }
    )
    subsample_transform = Subsample(config)
    subsampled_data = subsample_transform(data)
    
    # Check that the ratios are approximately correct
    class_counts = {
        'birds': sum(1 for v in subsampled_data.values() if v['class'] == 'birds'),
        'mammals': sum(1 for v in subsampled_data.values() if v['class'] == 'mammals'),
        'amphibians': sum(1 for v in subsampled_data.values() if v['class'] == 'amphibians')
    }
    assert abs(class_counts['birds'] / 100 - 0.5) < 0.1
    assert abs(class_counts['mammals'] / 100 - 0.3) < 0.1
    assert abs(class_counts['amphibians'] / 100 - 0.7) < 0.1

def test_build_transforms():
    """Test building transformations from configuration."""
    # Test building a single filter transform
    configs = [{
        'filter': {
            'property': 'source',
            'values': ['xeno-canto', 'iNaturalist'],
            'operation': 'include'
        }
    }]
    transforms = build_transforms(configs)
    assert len(transforms) == 1
    assert isinstance(transforms[0], Filter)
    
    # Test building a single subsample transform
    configs = [{
        'subsample': {
            'property': 'class',
            'operation': 'subsample',
            'ratios': {'birds': 0.5}
        }
    }]
    transforms = build_transforms(configs)
    assert len(transforms) == 1
    assert isinstance(transforms[0], Subsample)
    
    # Test building multiple transforms
    configs = [
        {
            'filter': {
                'property': 'source',
                'values': ['xeno-canto'],
                'operation': 'include'
            }
        },
        {
            'subsample': {
                'property': 'class',
                'operation': 'subsample',
                'ratios': {'birds': 0.5}
            }
        }
    ]
    transforms = build_transforms(configs)
    assert len(transforms) == 2
    assert isinstance(transforms[0], Filter)
    assert isinstance(transforms[1], Subsample)
    
    # Test invalid transform type
    configs = [{'invalid': {}}]
    with pytest.raises(ValueError):
        build_transforms(configs) 