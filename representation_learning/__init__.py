# Ensure custom esp-data transforms are registered before any configs are imported.
# This must execute before `representation_learning.configs` is imported, otherwise
# Pydantic will not recognise the extra transform type in the DatasetCollectionConfig
# union.
from importlib import import_module

# Trigger side-effects only; ignore if esp_data is missing (unit-test contexts)
try:
    import_module("representation_learning.data.text_label_from_features")
    import_module("representation_learning.data.require_features")
except ModuleNotFoundError:
    pass
