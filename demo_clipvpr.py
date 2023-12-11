import cv2
import torch
import llama
import clipvpr
from PIL import Image

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, process = clipvpr.load(clip_name="ViT-B/16", llama_name="BIAS-7B", llama_dir='./path/to/LLaMA/', llama_type="7B", 
            llama_download_root='ckpts', max_seq_len=512, phase="finetune", 
            prompt=['Is the scene in the picture urban or rural? How many lanes are there on the road in the photo? Is there a residential building in the picture? If so, which side of the road is it located on? Are there vegetation and trees in the photo? If so, which side of the road is it located on?'])
    
    model.eval().to(device)
    
    # prompts_list = 2 * ['Is there a sidewalk on the road in the picture?']
    # prompts = [llama.format_prompt(prompt) for prompt in prompts_list]
    
    img1 = Image.fromarray(cv2.imread("./data/image1.jpg"))
    img2 = Image.fromarray(cv2.imread("./data/image2.jpg"))
    imgs = [img1, img2]
    img_load = torch.stack([process(img).to(device) for img in imgs], dim=0)
    #print(img_load.shape)
    
    result = model(img_load)
    
    print(result.shape)
    print(model)


if __name__ == "__main__":
    main()