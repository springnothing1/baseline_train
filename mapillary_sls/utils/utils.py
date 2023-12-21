#  Copyright (c) Facebook, Inc. and its affiliates.

from torchvision import transforms
import torch

from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
    
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


def _convert_image_to_rgb(image):
    return image.convert("RGB")

def clip_transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])