import torch.nn as nn
import torchvision.models as models


def get_backbone(name: str, num_classes: int, pretrained: bool = True):
    name = name.lower()

    # -------------------
    # ResNet family
    # -------------------
    if name in ["resnet18", "resnet50", "resnet101"]:
        weights = "IMAGENET1K_V1" if pretrained else None

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
        weights = "IMAGENET1K_V1" if pretrained else None

        model_fn = getattr(models, name)
        model = model_fn(weights=weights)

        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    # -------------------
    # ConvNeXt
    # -------------------
    if name.startswith("convnext"):
        weights = "IMAGENET1K_V1" if pretrained else None

        model_fn = getattr(models, name)
        model = model_fn(weights=weights)

        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported backbone: {name}")