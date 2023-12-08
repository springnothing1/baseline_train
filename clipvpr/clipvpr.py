import hashlib
import os
import urllib
import warnings
from typing import Any, Union, List
from pkg_resources import packaging
import clip
import llama
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
from .utils import _llama_download,_clip_download
from .model import build_model

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


if packaging.version.parse(torch.__version__) < packaging.version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")


# __all__ = ["available_models", "load", "tokenize"]

_CLIP_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}

_LLAMA_MODELS = {
    "BIAS-7B": "https://github.com/OpenGVLab/LLaMA-Adapter/releases/download/v.2.0.0/7fa55208379faf2dd862565284101b0e4a2a72114d6490a95e432cf9d9b6c813_BIAS-7B.pth",
    "LORA-BIAS-7B": "https://github.com/OpenGVLab/LLaMA-Adapter/releases/download/v.2.0.0/1bcbffc43484332672092e0024a8699a6eb5f558161aebf98a7c6b1db67224d1_LORA-BIAS-7B.pth",
    "CAPTION-7B": "https://github.com/OpenGVLab/LLaMA-Adapter/releases/download/v.2.0.0/5088aeb63a89746b90bcfd5cb819e1c7411b2771b267c6d131ce73e250a8abf0_CAPTION-7B.pth",
    "LORA-BIAS-7B-v21": "https://github.com/OpenGVLab/LLaMA-Adapter/releases/download/v.2.1.0/d26d107eec32127ac86ef1997cf7169de1c56a59c539fc1258c6798b969e289c_LORA-BIAS-7B-v21.pth",
    # "LORA16-7B": "",
    # "PARTIAL-7B": ""
}


def load(clip_name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False,
         clip_download_root: str = None,  llama_name = "BIAS-7B", llama_dir='./path/to/LLaMA/', llama_type="7B", 
         llama_download_root='ckpts', max_seq_len=512, phase="finetune", prompt="Please introduce this painting"):
    """Load a CLIP model

    Parameters
    ----------
    clip_name : str
        A model clip_name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    Returns
    -------
    model : torch.nn.Module
        The CLIP model

    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    # load clip
    if clip_name in _CLIP_MODELS:
        model_path = _clip_download(_CLIP_MODELS[clip_name], clip_download_root or os.path.expanduser("./cache/clip"))
    elif os.path.isfile(clip_name):
        model_path = clip_name
    else:
        raise RuntimeError(f"Model {clip_name} not found; available models = {clip.available_models()}")

    with open(model_path, 'rb') as opened_file:
        # clip_state_dict = torch.load(opened_file, map_location="cpu")
        try:
            # loading JIT archive
            clip_model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
            clip_state_dict = None
        except RuntimeError:
            # loading saved state dict
            if jit:
                warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                jit = False
            clip_state_dict = torch.load(opened_file, map_location="cpu")

    # load llama
    if llama_name in _LLAMA_MODELS:
        model_path = _llama_download(_LLAMA_MODELS[llama_name], llama_download_root)
    elif os.path.isfile(llama_name):
        model_path = llama_name
    else:
        return RuntimeError(f"Model {llama_name} not found; available models = {llama.available_models()}"), None
    
    llama_ckpt_dir = os.path.join(llama_dir, llama_type)
    llama_tokenzier_path = os.path.join(llama_dir, 'tokenizer.model')
    
    ckpt = torch.load(model_path, map_location='cpu')
    model_cfg = ckpt.get('config', {})

    model = build_model(clip_model.state_dict() if clip_state_dict is None else clip_state_dict,
                        prompt, llama_ckpt_dir,llama_tokenzier_path,model_cfg, max_seq_len, phase)
    if str(device) == "cpu":
        model.float()

    return model.to(device)#, clip._transform(model.visual.input_resolution)

    
