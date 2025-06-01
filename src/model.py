import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes, model_name="efficientnet_b0", pretrained=True):
    """
    Load a pre-trained CNN and modify the classifier head.
    
    Args:
        num_classes (int): Number of target classes
        model_name (str): Pretrained model to use
        pretrained (bool): Whether to load ImageNet pretrained weights
    
    Returns:
        model (torch.nn.Module): Modified model ready for training
    """
    if model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=pretrained)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=pretrained)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")

    return model
