import pandas as pd
import yaml
from torch.utils.data import DataLoader
import pandas as pd
import cloudpathlib
from google.cloud.storage.client import Client
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable
from io import StringIO

from representation_learning.data.transformations import build_transforms

ANIMALSPEAK_PATH = "gs://animalspeak2/splits/v1/animalspeak_train_v1.3.csv"
ANIMALSPEAK_PATH_EVAL = "gs://animalspeak2/splits/v1/animalspeak_eval_v1.3.csv"

def get_dataset_from_name(name: str, validation = False):
    name = name.lower().strip()

    if name == "animalspeak":
        anaimspeak_path = ANIMALSPEAK_PATH_EVAL if validation else ANIMALSPEAK_PATH
        if ANIMALSPEAK_PATH.startswith("gs://"):
            csv_path = GSPath(anaimspeak_path)
        else:
            csv_path = Path(anaimspeak_path)

        # Read CSV content
        csv_text = csv_path.read_text(encoding="utf-8")
        df = pd.read_csv(StringIO(csv_text))
        df["gs_path"] = df["local_path"].apply(lambda x: "gs://" + x) # AnimalSpeak missing gs path
        return df
    else:
        raise NotImplementedError("Only AnimalSpeak dataset supported")

def balance_by_attribute(dataset, attribute, strategy='undersample', target_count=None, random_state=42):
    """
    Balance a pandas DataFrame by the specified attribute.
    
    Parameters:
    -----------
    dataset : pandas.DataFrame
        The input dataset to be balanced
    attribute : str
        The column name to balance by
    strategy : str, optional (default='undersample')
        Strategy for balancing:
        - 'undersample': Reduce all classes to the size of the smallest class
        - 'oversample': Increase all classes to the size of the largest class
        - 'target': Set all classes to a specific count defined by target_count
    target_count : int, optional (default=None)
        Target sample count per class when using 'target' strategy
    random_state : int, optional (default=42)
        Random seed for reproducibility
    
    Returns:
    --------
    pandas.DataFrame
        Balanced dataset
    """
    if attribute not in dataset.columns:
        raise ValueError(f"Attribute '{attribute}' not found in dataset columns")
    
    # Get value counts
    value_counts = dataset[attribute].value_counts()
    min_count = value_counts.min()
    max_count = value_counts.max()
    
    # Determine target counts based on strategy
    if strategy == 'undersample':
        target_counts = {val: min_count for val in value_counts.index}
    elif strategy == 'oversample':
        target_counts = {val: max_count for val in value_counts.index}
    elif strategy == 'target':
        if target_count is None:
            raise ValueError("target_count must be specified when using 'target' strategy")
        target_counts = {val: target_count for val in value_counts.index}
    else:
        raise ValueError("Strategy must be one of: 'undersample', 'oversample', 'target'")
    
    # Create empty DataFrame to hold balanced data
    balanced_data = pd.DataFrame(columns=dataset.columns)
    
    # Balance each class
    for val, count in target_counts.items():
        class_data = dataset[dataset[attribute] == val]
        
        if len(class_data) > count:
            # Undersample
            balanced_class = class_data.sample(n=count, random_state=random_state)
        elif len(class_data) < count:
            # Oversample with replacement
            balanced_class = class_data.sample(n=count, replace=True, random_state=random_state)
        else:
            # Already at target count
            balanced_class = class_data
            
        balanced_data = pd.concat([balanced_data, balanced_class], ignore_index=True)
    
    # Shuffle the final dataset
    return balanced_data.sample(frac=1, random_state=random_state).reset_index(drop=True)

def resample():
    pass

@lru_cache(maxsize=1)
def _get_client():
    return cloudpathlib.GSClient(storage_client=Client())

class GSPath(cloudpathlib.GSPath):
    """
    A wrapper for the cloudpathlib GSPath that provides a default client.
    This avoids issues when the GOOGLE_APPLICATION_CREDENTIALS variable is not set.
    """
    def __init__(self, client_path, client=_get_client()):
        super().__init__(client_path, client=client)
