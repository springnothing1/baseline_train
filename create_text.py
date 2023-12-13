import cv2
import os
import llama
import torch
from PIL import Image
import torch
import predict
import evaluate
import argparse
import torchvision
from torch import nn
from pathlib import Path
from mapillary_sls.datasets.msls import MSLS
from mapillary_sls.utils.utils import configure_transform
from torch.utils.data import DataLoader


def data_iter(batch_size, data):
    # number of iterator
    num = len(data) // batch_size + 1
    for i in range(num):
        # get batch_size data every time
        start = i * batch_size
        end = min((i + 1) * batch_size, len(data))
        yield data[start: end]

def llama_create_save(data_iter, city, transform, model, path, prompt_load, device, num):
    # len_iter = sum([1 for _ in data_iter])
    count = 0
    results = []
    print("\n=====>city:" + city + " query start:")
    for i, images in enumerate(data_iter):
        imgs = torch.stack([transform(Image.fromarray(cv2.imread(im))) for im in images], dim=0).to(device)
        prompts = imgs.shape[0] * prompt_load
        results.extend(model.generate(imgs, prompts, max_gen_len=77))
        print(results[0])
        count += imgs.shape[0]
        if count % 100 == 0:
            print(f"======create {count}/{num} images description already!!")
            
    with open(path, 'w', encoding='utf-8') as f:
        for line in results:
            f.write(line + '\n')
    print(f"========>create {count}/{num} images description for {city} query totally!!\n")

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    llama_dir = "./path/to/LLaMA/"

    # choose from BIAS-7B, LORA-BIAS-7B, CAPTION-7B.pth
    model, transform = llama.load("BIAS-7B", llama_dir, device=device)
    model.eval().to(device)
    prompt = ['Is the scene in the picture urban or rural? How many lanes are there on the road in the photo? Is there a residential building in the picture? If so, which side of the road is it located on? Are there vegetation and trees in the photo? If so, which side of the road is it located on?']
    prompt_load = [llama.format_prompt(prompt)]
    
    #cities = "trondheim,london,boston,melbourne,amsterdam,helsinki,tokyo,toronto,saopaulo,moscow,zurich,paris,bangkok,budapest,austin,berlin,ottawa,phoenix,goa,amman,nairobi,manila,cph,sf".split(",")
    cities = "amsterdam,helsinki,tokyo,toronto,saopaulo,moscow,zurich,paris,bangkok,budapest,austin,berlin,ottawa,phoenix,goa,amman,nairobi,manila,cph,sf".split(",")
    root_dir = Path('/root/autodl-tmp/msls').absolute()
    
    for city in cities:
        print("\n=====>city:" + city + " start:")
        # return the train dataset
        train_dataset = MSLS(root_dir, cities = city, transform = transform, mode = 'train', 
                            task = "im2im", seq_length = 1,negDistThr = 25, 
                            posDistThr = 5, nNeg = 5, cached_queries = 60, 
                            cached_negatives = 60, positive_sampling = True)
        
        q_data_iter = data_iter(batch_size, train_dataset.qImages)
        db_data_iter = data_iter(batch_size, train_dataset.dbImages)
        query_path = os.path.join(root_dir, 'train_val', city, "query", "descriptions.txt")
        database_path = os.path.join(root_dir, 'train_val', city, "database", "descriptions.txt")
        # create text description and save 
        llama_create_save(q_data_iter, city, transform, model, query_path, prompt_load, device, len(train_dataset.qImages))
        llama_create_save(db_data_iter, city, transform, model, database_path, prompt_load, device, len(train_dataset.dbImages))
        
        print("=====>city:" + city + " end:\n")         
                

if __name__ == "__main__":
    main()