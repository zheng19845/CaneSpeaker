import torch
import random
import os
import re
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from dataset import VLN_Dataset
from model import MM_Model
from config import IGNORE_INDEX, STOP_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
_weights_path = r"/home/data/zyy/Mp3D/weights/llava/"
_eval_path = r"/home/zyy/Matterport3D/Matterport3DSimulator/tasks/llava/eval/"
from args_parser import parse_args
 
 
def eval_item(model, prompt, images, instr, task_type, **kwargs):

    prompt = model.format_prompt(prompt)
    # print(prompt)

    image_features = model.encode_images(images)

    input_ids = model.tokenize(prompt)
    return model.generate(input_ids = input_ids, image_features_pre_proj = image_features, **kwargs)
    
    

@torch.no_grad()
def eval(args):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # args.device = "cpu"
    # args.weights_path = r"/home/zheng/Documents/Matterport3D/Matterport3DSimulator/tasks/llava/weights_best_new/"
    # args.resume = "0_24000_R2R_RxR_REVERIE_"
    args.resume = "best_"
    # args.resume = "1_17000_R2R_RxR_REVERIEobj_SOON_Caption_Rx2R_"
    # args.resume = "None" 
    
    args.arch = 'linear'
    args.seed = -1
    if args.seed != -1:
        random.seed(args.seed)
    
    # tasks = ["Caption"]
    # split = "val"
    # save_im = True
    
    tasks = ["R2R"]
    split = "val_seen"
    save_im = False
    # save_im = True
    task_type = "RxR"
    
    # prompt = "Give detailed description of the couch and the room."
    prompt = None
    # sent = "in at least 4 sentences."


    model = MM_Model(args).to(args.device)
    
    model.eval()
    # model.train()
    dataset = VLN_Dataset(args, splits= [split], tasks=tasks)
    
    
    test_id =  5177
    for idx in range(len(dataset.data)):
        if dataset.data[idx]['id'] == test_id:
            test_id = idx
            print(idx)
            break
    
    print(dataset.data[test_id]['id'])
    dataset.data[test_id]['type'] = task_type
        
    sample = dataset.__getitem__(test_id, prompt_idx = prompt, pano = 0, obj = 0)
    prompt, images, instr, task_type = sample 
    # save_im = False
    # result = eval_item(model, prompt, images, instr, task_type)
    result = eval_item(model, prompt, images, instr, task_type, 
                       max_new_tokens = 100, 
                       do_sample=True, 
                    #    top_k = 10, 
                       top_p = 0.6, 
                       temperature = 1.2, 
                       no_repeat_ngram_size = 5, 
                       num_return_sequences = 5)
    # result = eval_item(model, prompt, images, instr, task_type, 
    #                     max_new_tokens = 320, 
    #                     do_sample=False, 
    #                     num_beams = 5,
    #                     no_repeat_ngram_size = 5, 
    #                     num_return_sequences = 1)
    print(prompt)
    # print(dataset.data[test_id]["id"])
    print("Reference: ")
    if task_type == 'R2R':
        for i in range(3):
            sample = dataset.__getitem__((test_id//3)*3 + i, prompt_idx=0)
            _, _, instr, task_type = sample 
            print(instr)
    else:
        print(instr)

    print("Generated: ")
    print(result)
    
    if save_im:
        import numpy as np
        import cv2
        for c in range(len(images)):
            # images[c].save(args.out_path + str(c)+".jpg")
            image = np.array(images[c])
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow('img', image)
            cv2.waitKey(0)

    

if __name__ == "__main__":
    args = parse_args()
    eval(args)
