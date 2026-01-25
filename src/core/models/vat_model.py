"""
VAT (Video Attention Transformer) Model Architecture
=====================================================
ResNet-50 backbone + Transformer tail for video-based personality prediction.

Configured for:
- 32 frames per video
- 224x224 input resolution
- 5 OCEAN personality traits
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable


class FeedForward(nn.Module):
    """Feed-forward network used in transformer blocks"""

    def __init__(self, d_model, d_ff=2048, dropout=0.3):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        nn.init.normal_(self.linear_1.weight, std=0.001)
        nn.init.normal_(self.linear_2.weight, std=0.001)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class Norm(nn.Module):
    """Layer normalization with optional trainable parameters"""

    def __init__(self, d_model, eps=1e-6, trainable=False):
        super(Norm, self).__init__()
        self.size = d_model
        if trainable:
            self.alpha = nn.Parameter(torch.ones(self.size))
            self.bias = nn.Parameter(torch.zeros(self.size))
        else:
            self.alpha = nn.Parameter(torch.ones(self.size), requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(self.size), requires_grad=False)
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding for temporal sequence"""

    def __init__(self, d_model, max_seq_len=80):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        batch_size = x.size(0)
        num_feature = x.size(2)
        spatial_h = x.size(3)
        spatial_w = x.size(4)
        z = Variable(self.pe[:, :seq_len], requires_grad=False)
        z = z.unsqueeze(-1).unsqueeze(-1)
        z = z.expand(batch_size, seq_len, num_feature, spatial_h, spatial_w)
        x = x + z
        return x


def attention(q, k, v, d_k, mask=None, dropout=None):
    """Scaled dot-product attention"""
    scores = torch.sum(q * k, -1) / math.sqrt(d_k)
    scores = F.softmax(scores, dim=-1)
    scores = scores.unsqueeze(-1).expand(scores.size(0), scores.size(1), v.size(-1))
    output = scores * v
    output = torch.sum(output, 1)
    if dropout:
        output = dropout(output)
    return output


class TX(nn.Module):
    """Single transformer block with attention and feed-forward"""

    def __init__(self, d_model=64, dropout=0.3):
        super(TX, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.ff = FeedForward(d_model, d_ff=int(d_model / 2))

    def forward(self, q, k, v, mask=None):
        b = q.size(0)
        t = k.size(1)
        dim = q.size(1)
        q_temp = q.unsqueeze(1)
        q_temp = q_temp.expand(b, t, dim)
        A = attention(q_temp, k, v, self.d_model, mask, self.dropout)
        q_ = self.norm_1(A + q)
        new_query = self.norm_2(q_ + self.dropout_2(self.ff(q_)))
        return new_query


class BlockHead(nn.Module):
    """Multi-layer transformer block (3 TX layers)"""

    def __init__(self, d_model=64, dropout=0.3):
        super(BlockHead, self).__init__()
        self.T1 = TX()
        self.T2 = TX()
        self.T3 = TX()

    def forward(self, q, k, v, mask=None):
        q = self.T1(q, k, v)
        q = self.T2(q, k, v)
        q = self.T3(q, k, v)
        return q


class Tail(nn.Module):
    """Transformer tail for temporal aggregation of video features"""

    def __init__(self, num_classes, num_frames, head=16, return_feature=False, spatial_size=7):
        super(Tail, self).__init__()
        self.spatial_h = spatial_size
        self.spatial_w = spatial_size
        self.head = head
        self.num_features = 2048
        self.num_frames = num_frames
        self.return_feature = return_feature
        self.d_model = int(self.num_features / 2)
        self.d_k = self.d_model // self.head
        self.bn1 = nn.BatchNorm2d(self.num_features)
        self.bn2 = Norm(self.d_model, trainable=False)

        self.pos_embedding = PositionalEncoder(self.num_features, self.num_frames)
        self.Qpr = nn.Conv2d(self.num_features, self.d_model,
                             kernel_size=(spatial_size, spatial_size), stride=1, padding=0, bias=False)

        # Note: named "list_layers" to match checkpoint saved from original training code
        self.list_layers = nn.ModuleList([BlockHead() for _ in range(self.head)])
        self.classifier = nn.Linear(self.d_model, num_classes)

        nn.init.kaiming_normal_(self.Qpr.weight, mode='fan_out')
        nn.init.normal_(self.classifier.weight, std=0.001)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)

    def forward(self, x, b, t):
        x = self.bn1(x)
        x = x.view(b, t, self.num_features, self.spatial_h, self.spatial_w)
        x = self.pos_embedding(x)
        x = x.view(-1, self.num_features, self.spatial_h, self.spatial_w)
        x = F.relu(self.Qpr(x))
        x = x.view(-1, t, self.d_model)
        x = self.bn2(x)

        q = x[:, int(t / 2), :]
        v = x
        k = x

        q = q.view(b, self.head, self.d_k)
        k = k.view(b, t, self.head, self.d_k)
        v = v.view(b, t, self.head, self.d_k)

        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        outputs = []
        for i in range(self.head):
            outputs.append(self.list_layers[i](q[:, i], k[:, i], v[:, i]))

        f = torch.cat(outputs, 1)
        f = F.normalize(f, p=2, dim=1)
        y = self.classifier(f)

        if self.return_feature:
            return y, f
        return y


class SemiTransformer(nn.Module):
    """
    VAT Model: ResNet-50 backbone + Transformer tail

    Architecture:
        - ResNet-50 pretrained on ImageNet (without avgpool and fc)
        - Transformer tail with 16-head multi-head attention
        - Configured for 224x224 input with 32 frames

    Input:
        x: Tensor of shape [batch, num_frames, 3, height, width]
           where height=width=224 and num_frames=32

    Output:
        Tensor of shape [batch, num_classes] with personality trait predictions
        Trait order: openness, conscientiousness, extraversion, agreeableness, neuroticism
    """

    def __init__(self, num_classes=5, seq_len=32, return_feature=False):
        super(SemiTransformer, self).__init__()
        self.return_feature = return_feature

        # For 224x224 input: spatial_size = 224/32 = 7
        spatial_size = 7

        # ResNet-50 backbone (without avgpool and fc)
        resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])

        # Transformer tail
        self.tail = Tail(num_classes, seq_len, return_feature=return_feature, spatial_size=spatial_size)

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        return self.tail(x, b, t)
