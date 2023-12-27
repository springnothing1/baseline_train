import cv2
import llama
import torch
from PIL import Image
import clipvpr
    

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)

    llama_dir = "./path/to/LLaMA/"

    # choose from BIAS-7B, LORA-BIAS-7B, CAPTION-7B.pth
    model, preprocess = llama.load("BIAS-7B", llama_dir, device=device)
    model.eval()

    #prompts_list = ['Please introduce this painting.', 'Please introduce this painting.']
    prompts_list = 2 * ['Is there a sidewalk on the road in the picture?']

    prompts = [llama.format_prompt(prompt) for prompt in prompts_list]
    img1 = Image.fromarray(cv2.imread("./data/image1.jpg"))
    img2 = Image.fromarray(cv2.imread("./data/image2.jpg"))
    imgs = [img1, img2]
    img = torch.stack([preprocess(img).to(device) for img in imgs], dim=0)
    print(img.shape)
    result = model.generate(img, prompts, max_gen_len=77)

    print(result)

    """class RoomClassifier:
        def __init__(self,possible_rooms=INTEREST_ROOMS,caption_checkpoint=CAPTION_CHECKPOINT_PATH,llama_checkpoint=LLAMA_CHECKPOINT_PATH,device="cuda:0"):
            self.device=device
            self.possible_rooms = possible_rooms
            self.model,self.preprocess = llama.load(caption_checkpoint,llama_checkpoint,device=device)
            self.model.eval()
        
        def predict(self,image):
            prompt = ["Classify the room shown in the image. The possible room types are %s"%self.possible_rooms]
            answer = self.image_caption(image,prompt)
            answer = answer[0].replace(" room","room")
            for room in self.possible_rooms:
                if room in answer:
                    return room
            return None
            
        def image_caption(self,image,prompts):
            load_prompts = [llama.format_prompt(p) for p in prompts]
            load_image = Image.fromarray(image)
            load_image = self.preprocess(load_image).unsqueeze(0).to(self.device).tile((len(prompts),1,1,1))
            result = self.model.generate(load_image,load_prompts,temperature=0.0,top_p=1.0)
            return result"""


if __name__ == "__main__":
    main()
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #model = clipvpr.load(clip_name="ViT-B/16", device = device, llama_name="BIAS-7B", llama_dir='./path/to/LLaMA/', llama_type="7B", 
    #        llama_download_root='ckpts', max_seq_len=512, phase="finetune")
        