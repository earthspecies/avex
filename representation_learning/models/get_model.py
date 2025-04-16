from representation_learning.models.efficientnetb0 import Model as EfficientNetB0

def get_model(model_config, num_classes):
    """
    Factory function to obtain a model instance based on a static list of supported models.
    Model implementations are expected to reside in their own modules (e.g. efficientnetb0.py)
    and define a class (always called 'Model'). This function currently supports the 'efficientnetb0' model.

    Args:
        model_config (str): The model configuration name (e.g., "efficientnetb0").
        num_classes (int): The number of classes to be used in the model.

    Returns:
        An instance of the corresponding model configured with the provided num_classes.

    Raises:
        NotImplementedError: If the model_config does not match any supported models.
    """
    if model_config.lower() == "efficientnetb0":
        return EfficientNetB0(num_classes=num_classes)
    else:
        raise NotImplementedError(
            f"Model '{model_config}' is not implemented. Supported models: 'efficientnetb0'."
        )
