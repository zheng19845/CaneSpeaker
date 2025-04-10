import sys
import torch
import copy
import cv2
import json
import numpy as np
import re
from PIL import Image
from tqdm import tqdm
import argparse

from pycocoevalcap.eval import Bleu, Cider, Rouge, Meteor, Spice, PTBTokenizer
from tokenization_clip import SimpleTokenizer




def score(save_file,):
    with open(save_file) as f:
        cur_data = json.load(f)
    

    scorers = [
        (Bleu(4), ["Bleu_1","Bleu_2","Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        (Spice(), "Spice"),
    ]

    ref = {}
    gen = {}

    tokenizer = SimpleTokenizer()
    for item in cur_data:
        ground_truth_instr = [tmp for tmp in item["references"]]
        instr = item['instructions'][0]
        
        ref[item["path_id"]] = copy.deepcopy([{'caption': tmp} for tmp in ground_truth_instr])
        gen[item["path_id"]] = copy.deepcopy([{'caption': instr}])

    
    max_len = 200
    for i in ref:
        # print("ref[i]: ", ref[i])
        tmp = [item['caption'] for item in ref[i]]
        # print("tmp: ", tmp)
        # print()
        tmp = [' '.join(tokenizer.split_sentence(sent)) for sent in tmp]
        for j in range(len(tmp)):
            if tmp[j].__len__()>max_len:
                
                tmp[j] = tmp[j][:max_len]
        ref[i] = tmp
    for i in gen:
        # print("ref[i]: ", ref[i])
        tmp = [item['caption'] for item in gen[i]]
        # print("tmp: ", tmp)
        # print()
        tmp = [' '.join(tokenizer.split_sentence(sent)) for sent in tmp]
        for j in range(len(tmp)):
            if tmp[j].__len__()>max_len:

                tmp[j] = tmp[j][:max_len]
        gen[i] = tmp
        # print(gen[i])
    
    results = {}
    for scorer, method in scorers:
        print('computing %s score...'%(scorer.method()))
        score, scores = scorer.compute_score(ref, gen)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                print(m)
                print(sc)
                results[m] = sc

        else:
            print(method)
            print(score)
            results[method] = score

        print()
    
    print()
    print()
    return results
    


def parse_args():
    parser = argparse.ArgumentParser(description="")
    
    parser.add_argument("--paths", nargs = '+', type = str, default = '')
    args, _ = parser.parse_known_args()

    if type(args.paths)!=list:
        args.paths = [args.paths]

    return args




if __name__=="__main__":
    args = parse_args()
    for task in args.val_dataset:
        for split in args.val_split:
            path = args.out_path + task +'_%s.json' % split
            result = score(save_file=path)
            print(path)
            print(result)
            with open(path.split('.json')[0] + '_score.json', 'w') as f:
                json.dump(result, f, indent=4)
        