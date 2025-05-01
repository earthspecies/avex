import torch.nn as nn

def _build_criterion(loss_name: str) -> nn.Module:
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss_name in {"bce", "binary_cross_entropy_with_logits"}:
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown loss_function: {loss_name}")