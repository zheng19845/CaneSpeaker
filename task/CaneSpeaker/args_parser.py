import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="")
    
    parser.add_argument('--root_path', type = str, default = r"./")
    parser.add_argument('--weights_path', type = str, default = r"./tasks/CaneSpeaker/weights/")
    parser.add_argument('--out_path', type = str, default = r"./tasks/CaneSpeaker/eval/")
    parser.add_argument('--dataset_path', type = str, default = r"./tasks/data/")
    parser.add_argument('--base_model_path', type = str, default = 'jiajunlong/TinyLLaVA-OpenELM-450M-SigLIP-0.89B')
    
    parser.add_argument("--dataset", nargs = '+', type = str, default = 'R2R') # R2R RxR
    parser.add_argument("--split", nargs = '+', type = str, default = 'train')
    parser.add_argument('--dataset_sample', type = bool, default = False)
    parser.add_argument("--max_path", type = int, default = 16)
    
    parser.add_argument("--val_dataset", nargs = '+', type = str, default = 'R2R')
    parser.add_argument("--val_split", nargs = '+', type = str, default = 'val_unseen')
    parser.add_argument('--val_dataset_sample', type = float, default = 0.1)
    
    parser.add_argument("--resume", type = str, default = "None")
    
    
    parser.add_argument("--batch_size", type = int, default = 4)
    parser.add_argument("--accumulate_size", type = int, default = 1)
    parser.add_argument("--epoch", type = int, default = 1)
    
    parser.add_argument('--lr', nargs = '+', type = float, default = 5e-5)  # peft, connector
    parser.add_argument('--decay', type = float, default = 0.99)
    
    parser.add_argument('--arch', type = str, default = 'linear')
    parser.add_argument('--dropout', type = float, default = 0.3)

    parser.add_argument('--device', type = str, default = 'default')
    
    parser.add_argument('--save', type = int, default = 1000)
    
    parser.add_argument('--seed', type = int, default = -1)
    
    args, _ = parser.parse_known_args()

    args = postprocess_args(args)

    return args


def postprocess_args(args):
    if args.device == "default":
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if type(args.dataset)!=list:
        args.dataset = [args.dataset]
    if type(args.val_dataset)!=list:
        args.val_dataset = [args.val_dataset]
        
    if type(args.split)!=list:
        args.split = [args.split]
    if type(args.val_split)!=list:
        args.val_split = [args.val_split]
    return args