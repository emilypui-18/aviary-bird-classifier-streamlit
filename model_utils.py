"""Shared PyTorch model architecture and preprocessing from the training project.

The FastAPI server uses these exact layers and evaluation transforms.
"""
from torch import nn
from torchvision import models, transforms
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

def get_transforms(img_size, augment=True):
    eval_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size), transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    if not augment: return eval_transform, eval_transform
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.65, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.RandomRotation(12),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    return train_transform, eval_transform


def build_model(name, num_classes, pretrained=True, dropout=0.4):
    """Return CNN and its replaceable classifier head."""
    if name == 'resnet50':
        w = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = models.resnet50(weights=w)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, num_classes))
        head = m.fc
    elif name == 'mobilenet_v2':
        w = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.mobilenet_v2(weights=w)
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.classifier[1].in_features, num_classes))
        head = m.classifier
    elif name == 'efficientnet_b0':
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b0(weights=w)
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.classifier[1].in_features, num_classes))
        head = m.classifier
    elif name == 'convnext_tiny':
        w = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.convnext_tiny(weights=w)
        m.classifier[2] = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.classifier[2].in_features, num_classes))
        head = m.classifier[2]
    else: raise ValueError(f'Unknown architecture: {name}')
    return m,head

