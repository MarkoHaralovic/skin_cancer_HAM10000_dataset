import torch.nn as nn
import torchvision.models as models

def _get_torchvision_weights(name: str, pretrained: bool):
    if not pretrained:
        return None
    try:
        weights_enum = models.get_model_weights(name)
    except (AttributeError, ValueError):
        return "DEFAULT"

    return weights_enum.DEFAULT

def get_backbone(name: str, num_classes: int, pretrained: bool = True):
    name = name.lower()

    # -------------------
    # ResNet family
    # -------------------
    if name in ["resnet18", "resnet50", "resnet101"]:
        weights = _get_torchvision_weights(name, pretrained)

        if name == "resnet18":
            model = models.resnet18(weights=weights)
        elif name == "resnet50":
            model = models.resnet50(weights=weights)
        elif name == "resnet101":
            model = models.resnet101(weights=weights)

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    # -------------------
    # EfficientNet
    # -------------------
    if name.startswith("efficientnet"):
        if hasattr(models, name):
            model_fn = getattr(models, name)
            model = model_fn(weights=_get_torchvision_weights(name, pretrained))

            if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(in_features, num_classes)
                return model

        raise ValueError(f"Unsupported torchvision EfficientNet classifier layout for: {name}")

    # -------------------
    # ConvNeXt
    # -------------------
    if name.startswith("convnext"):
        weights = _get_torchvision_weights(name, pretrained)

        model_fn = getattr(models, name)
        model = model_fn(weights=weights)

        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported backbone: {name}")
