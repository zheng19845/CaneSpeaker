import torch
import os
import re
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from dataset import VLN_Dataset
from model import MM_Model
from config import IGNORE_INDEX, STOP_TOKEN_INDEX
import json
from args_parser import parse_args
 
 
def eval_item(model, prompt, images, instr, task_type, **kwargs):
    prompt = model.format_prompt(prompt)

    image_features = model.encode_images(images)

    input_ids = model.tokenize(prompt)
    return model.generate(input_ids = input_ids, image_features_pre_proj = image_features, **kwargs)
    
    

@torch.no_grad()
def generate(args, model, dataset, task, split, save_file = None):
    if save_file == None:
        save_file = task + "_" + split +".json"
    instructions = []
    ground_truth = {}
    done_ids = {}
    for idx, item in enumerate(tqdm(dataset)):
        item = dataset.data[idx]
        if item["id"] in done_ids:
            ground_truth[item["id"]].append(item["instruction"])
            continue

        prompt, images, instr, task_type = dataset.__getitem__(idx, prompt_idx = 0, pano = 1, obj = 1 )
        # result = eval_item(model, prompt, images, instr, task_type)
        # result = eval_item(model, prompt, images, instr, task_type, 
        #                max_new_tokens = 180, 
        #                do_sample=True, 
        #             #    top_k = 50, 
        #                top_p = 0.6, 
        #                temperature = 1.2, 
        #                no_repeat_ngram_size = 5, 
        #                num_return_sequences = 3)
        result = eval_item(model, prompt, images, instr, task_type, 
                           max_new_tokens = 320, 
                           do_sample=False,  
                           num_beams = 5,
                           no_repeat_ngram_size = 5, 
                           num_return_sequences = 1)

        done_ids[item["id"]] = result
        ground_truth[item["id"]] = [item["instruction"]]

        instructions.append({
            "scan" : item["scan"],  
            "path" : item["path"],
            "heading" : item["heading"],
            "path_id" : item["id"],
            "instructions" : result,
            # "distance" : item["distance"]
        })

            
    for item in instructions:
        item["references"] = ground_truth[item["path_id"]]

        
    with open(args.out_path + save_file,"w") as file:
        json.dump(instructions, file, indent=4)


    




if __name__ == "__main__":
    args = parse_args()
    
    tasks = args.val_dataset
    splits = args.val_split
    
    for task in tasks:
        for split in splits:
            model = MM_Model(args).to(args.device)
            model.eval()
            dataset = VLN_Dataset(args, splits= [split], tasks=[task])
            generate(args, model, dataset, task, split)
            del dataset
            del model
