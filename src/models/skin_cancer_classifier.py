import torch.nn as nn
from .backbone import get_backbone

class SkinLesionClassifier(nn.Module):
    def __init__(self, backbone_name='resnet50', num_classes=7, pretrained=True):
        """
        Args:
            backbone_name: Name of the backbone model to use
            num_classes: Number of output classes (7 for HAM10000 classification)
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        self.model = get_backbone(backbone_name, num_classes, pretrained)
     
    def forward(self, x):
        return self.model(x)