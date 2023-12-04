# Gem pooling implementation
# use adaptiveavg-pooling to fit the resnet50

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeMPooling(nn.Module):
    def __init__(self, feature_size, output_size=(1, 1), init_norm=3.0, eps=1e-6, normalize=False, **kwargs):
        super(GeMPooling, self).__init__(**kwargs)
        self.feature_size = feature_size  # Final layer channel size, the pow calc at -1 axis
        self.init_norm = init_norm
        self.p = torch.nn.Parameter(torch.ones(self.feature_size) * self.init_norm, requires_grad=True)
        self.p.data.fill_(init_norm)
        self.normalize = normalize
        # the adaptiveavgpool2d in the end of resnet50,get(batch_size, 2048, 15, 20)
        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size)
        self.eps = eps

    def forward(self, features):
        # filter invalid value: set minimum to 1e-6
        # features-> (B, C, H, W)
        features = features.clamp(min=self.eps)
        features = features.permute((0, 2, 3, 1))
        features = features.pow(self.p)

        features = features.permute((0, 3, 1, 2))
        features = self.avg_pooling(features)
        features = features.reshape(features.shape[0], -1)

        features = torch.pow(features, (1.0 / self.p))
        # unit vector
        if self.normalize:
            features = F.normalize(features, dim=-1, p=2)
        return features


if __name__ == '__main__':
    x = torch.randn((1, 2048, 15, 20)) * 0.02
    # x = torch.randn((8, 7, 7, 768)) * 0.02

    gem = GeMPooling(2048, output_size=(1, 1), init_norm=3.0)

    # print("input : ", x)
    print("=========================")
    print(gem(x).shape)