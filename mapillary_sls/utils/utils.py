#  Copyright (c) Facebook, Inc. and its affiliates.

from torchvision import transforms
import torch

def configure_transform(image_dim, meta):

	# transforms.Normalize用均值和标准差归一化张量图像
	normalize = transforms.Normalize(mean=meta['mean'], std=meta['std'])

	# 一般用Compose把多个步骤整合到一起
	transform = transforms.Compose([
		transforms.Resize(image_dim),    # 把给定的图片resize到given size
		transforms.ToTensor(),           # ToTensor()能够把灰度范围从0-255变换到0-1之间
		normalize,                       # 后面的transform.Normalize()则把0-1变换到(-1,1)
	])

	return transform
