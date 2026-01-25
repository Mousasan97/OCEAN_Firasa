"""
ResNet-based personality regressor model
Extracted from original script for better modularity
"""
import torch
import torch.nn as nn
from torchvision import models
from typing import Literal


class ResNetRegressor(nn.Module):
    """ResNet-based personality trait regressor"""

    def __init__(
        self,
        backbone: Literal["resnet18", "resnet50"] = "resnet18",
        out_dim: int = 5,
        pretrained: bool = False,
        dropout: float = 0.1
    ):
        """
        Initialize ResNet regressor

        Args:
            backbone: ResNet architecture (resnet18 or resnet50)
            out_dim: Number of output dimensions (personality traits)
            pretrained: Whether to use ImageNet pretrained weights
            dropout: Dropout rate before final layer
        """
        super().__init__()

        # Load backbone
        if backbone == "resnet50":
            net = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            )
        elif backbone == "resnet18":
            net = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Extract feature dimension
        in_feats = net.fc.in_features

        # Replace classifier with identity
        net.fc = nn.Identity()

        # Store backbone
        self.backbone = net

        # Regression head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            Personality trait predictions [B, out_dim]
        """
        features = self.backbone(x)
        return self.head(features)

    @property
    def num_parameters(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
