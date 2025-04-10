import torch
import os
import re
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from dataset import VLN_Dataset
from model import MM_Model
from config import IGNORE_INDEX, STOP_TOKEN_INDEX
from args_parser import parse_args
import random



def my_collate(batch):
    return batch


def generate_inputs_instrs(input_ids, instr_ids):
    input_ids = input_ids[0]
    instr_ids = instr_ids[0]
    
    
    input_ids_list = []
    instr_ids_list = []
    for i in range(instr_ids.shape[0]):
        if i == 0:
            input_ids_list.append(input_ids.unsqueeze(0))
            instr_ids_list.append(instr_ids[i].unsqueeze(0))
        else:
            input_ids_list.append(torch.cat([input_ids, instr_ids[:i]]).unsqueeze(0))
            instr_ids_list.append(instr_ids[i].unsqueeze(0))
        
    return input_ids_list, instr_ids_list
    

def pad(input_ids_list):
    pad_token_id = IGNORE_INDEX
    max_len = max([item.shape[1] for item in input_ids_list])
    # print(max_len)
    attention_masks = torch.zeros((len(input_ids_list), max_len))
    for idx in range(len(input_ids_list)):
        item = input_ids_list[idx]
        pad_length = max_len - item.shape[1]
        pad_ids = torch.tensor([pad_token_id] * pad_length).to(int).to("cuda").unsqueeze(0)
        # Concatenate the embeddings
        input_ids_list[idx] = torch.cat((item, pad_ids), dim=1)
        attention_masks[idx, :item.shape[1]] = 1
        # print(pad_length)
        # print(input_list[idx].shape)
        # print(attention_masks[idx])
    return torch.cat(input_ids_list, dim = 0), attention_masks



def train_epoch(model, dataloader, optimizer, scheduler, loss_fn, epoch, start_batch_id, task, val_envs, args):
    batch_num = 0
    batch_input = []
    batch_output = []
    batch_image_features = []
    
    accumulate_num = 0
    accumulate_loss = 0
    
    predicted = []
    ground_truth = []
    score = []
    
    start_time = time.time()
    
    optimizer.zero_grad()
    length = len(dataloader)

    for batch_idx, batch_sample in enumerate(tqdm(dataloader),start = start_batch_id):
        # if batch_idx ==len(dataloader):
        #     break
        print(f"Batch: {batch_idx}/{len(dataloader)}")
        if batch_idx!=start_batch_id:
            cur_run_time = int(time.time()-start_time)
            est_run_time = int(cur_run_time*(length - batch_idx + start_batch_id)/(batch_idx-start_batch_id))
            print(f"Running: {cur_run_time//3600} h {(cur_run_time%3600)//60} min {cur_run_time%60} sec, Estimating: {est_run_time//3600} h {(est_run_time%3600)//60} min {est_run_time%60}")
        print("Current lr (PEFT): ", f"{optimizer.param_groups[0]['lr']:.8f}")
        print()
        
        for k in range(len(batch_sample)):
            sample = batch_sample[k]
            prompt, images, instr, task_type = sample 
            print(task_type)
            print(instr)
            
            instr_ids = torch.cat([torch.tensor([model.module.tokenizer(instr, add_special_tokens=False).input_ids], device="cuda"), torch.tensor([[STOP_TOKEN_INDEX]], device = 'cuda')], dim = 1)
            # print(instr_ids.shape[1])
            if instr_ids.shape[1] > 180:
                print(instr_ids.shape[1], " : Too long, Pass.")
                continue
            
            prompt = model.module.format_prompt(prompt)
            input_ids = model.module.tokenize(prompt)
            
            with torch.no_grad():
                image_features = model.module.encode_images(images)

            input_ids_list, instr_ids_list = generate_inputs_instrs(input_ids, instr_ids)
            
            for input_ids, instr_ids in zip(input_ids_list, instr_ids_list):
                batch_input.append(input_ids)
                batch_output.append(instr_ids)
                batch_image_features.append(image_features)
                batch_num+=1


                # if batch_num >= batch_size or (batch_idx == len(dataloader)-1 and p == len(input_ids_list) - 1):
                if batch_num >= args.batch_size:
                    batch_input , attention_mask = pad(batch_input)
                    batch_output = torch.cat(batch_output)

                    batch_image_features = torch.cat(batch_image_features, dim = 0)

                    output, attention_mask = model(input_ids = batch_input, image_features_pre_proj = batch_image_features, attention_mask = attention_mask)
                    logits = []
                    for m in range(output.shape[0]):
                        idx = torch.nonzero(attention_mask[m]).squeeze()[-1].item()
                        # print(idx)
                        logits.append(output[m][idx].unsqueeze(0))
                        # print(attention_mask[i][idx], attention_mask[i][idx+1])
                    logits = torch.cat(logits)
                
                    # print(logits)
                    cur_loss = loss_fn(logits,batch_output)
                    accumulate_loss += cur_loss
                    accumulate_num += 1
                
                    cur_ground_truth = batch_output.detach().cpu().numpy().tolist()
                    predicted = predicted + torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
                    ground_truth = ground_truth + cur_ground_truth
                    logits_softmax = torch.nn.functional.softmax(logits, dim = 1)
                
                    for m in range(len(cur_ground_truth)):
                        score.append(logits_softmax[m][cur_ground_truth[m]])
                
                    # if accumulate_num >= accumulate_size or (batch_idx == len(dataloader)-1 and p == len(pairs) - 1):  
                    if accumulate_num >= args.accumulate_size:
                        # print("Checkpoint2")       
                        accumulate_loss = accumulate_loss / accumulate_num
                        accumulate_loss.backward()
                    
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                    
                        loss_float = accumulate_loss.detach().cpu().numpy().item()
                        with open(args.out_path+"training_loss.txt","a") as file:
                            file.write('{:.3f}\n'.format(loss_float))
                    
                    
                        print("Current loss:  ", loss_float)
                        print("Predicted:     ", predicted)
                        print("Ground truth:  ", ground_truth)
                        print("Score:         ", [f'{num:.3f}' for num in score])
                        print()
                    
                        predicted = []
                        ground_truth = []
                        score = []
                        
                        accumulate_num = 0
                        accumulate_loss = 0
                    batch_num = 0
                    batch_input = []
                    batch_output = []
                    batch_image_features = []
                
                    # del cur_loss
                    # torch.cuda.empty_cache()
        save = False
        # if (batch_idx + 1) > 0.5 * length or epoch > 0:
        #     if (batch_idx + 1) % (args.save // 2) == 0:
        #         save = True
        # elif (batch_idx + 1) % args.save == 0:
        #     save = True
        
        if (batch_idx + 1) % args.save == 0:
            save = True
        
        if save:
            model.module.save(args.weights_path + str(epoch)+"_"+str(batch_idx+1)+"_"+task)
            torch.save({
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'batch': batch_idx+1
            }, args.weights_path + str(epoch)+"_"+str(batch_idx+1)+"_"+task+"/checkpoint.pth")

            model.eval()
            with torch.no_grad():
                for val_task, val_split, _dataloader in val_envs:
                    eval_epoch(model, _dataloader, loss_fn, val_task, val_split, epoch, batch_idx, args)
            model.train()    
            # torch.cuda.empty_cache()
            
    model.module.save(args.weights_path + str(epoch+1)+"_"+str(0)+"_"+task)
    torch.save({
        'epoch': epoch+1,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'batch': 0
    }, args.weights_path + str(epoch+1)+"_"+str(0)+"_"+task+"/checkpoint.pth")
    model.eval()
    with torch.no_grad():
        for val_task, val_split, _dataloader in val_envs:
            eval_epoch(model, _dataloader, loss_fn, val_task, val_split, epoch+1, 0, args)
    model.train()    

   
def eval_epoch(model, dataloader, loss_fn, task, split, epoch, batch, args):
    print("EVAL: ", task, split)
    batch_num = 0
    batch_input = []
    batch_output = []
    batch_image_features = []
    
    accumulate_num = 0
    accumulate_loss = 0
    
    start_time = time.time()
    
    length = len(dataloader)

    for batch_idx, batch_sample in enumerate(tqdm(dataloader)):
        if batch_idx ==len(dataloader):
            break
        if batch_idx!=0:
            cur_run_time = int(time.time()-start_time)
            est_run_time = int(cur_run_time*(length-batch_idx)/(batch_idx))
            print(f"Running: {cur_run_time//3600} h {(cur_run_time%3600)//60} min {cur_run_time%60} sec, Estimating: {est_run_time//3600} h {(est_run_time%3600)//60} min {est_run_time%60}")

        
        for k in range(len(batch_sample)):
            sample = batch_sample[k]
            prompt, images, instr, task_type = sample 
            print(task_type)
            print(instr)
            
            instr_ids = torch.cat([torch.tensor([model.module.tokenizer(instr, add_special_tokens=False).input_ids], device="cuda"), torch.tensor([[STOP_TOKEN_INDEX]], device = 'cuda')], dim = 1)
            # print(instr_ids.shape[1])
            if instr_ids.shape[1] > 250:
                print(instr_ids.shape[1], " : Too long, Pass.")
                continue
            
            prompt = model.module.format_prompt(prompt)
            input_ids = model.module.tokenize(prompt)
            
            with torch.no_grad():
                image_features = model.module.encode_images(images)

            input_ids_list, instr_ids_list = generate_inputs_instrs(input_ids, instr_ids)
            
            for input_ids, instr_ids in zip(input_ids_list, instr_ids_list):
                batch_input.append(input_ids)
                batch_output.append(instr_ids)
                batch_image_features.append(image_features)
                batch_num+=1
            
                if batch_num >= args.batch_size:
                    batch_input , attention_mask = pad(batch_input)
                    batch_output = torch.cat(batch_output)

                    batch_image_features = torch.cat(batch_image_features, dim = 0)

                    output, attention_mask = model(input_ids = batch_input, image_features_pre_proj = batch_image_features, attention_mask = attention_mask)
                    logits = []
                    for m in range(output.shape[0]):
                        idx = torch.nonzero(attention_mask[m]).squeeze()[-1].item()
                        logits.append(output[m][idx].unsqueeze(0))
                    logits = torch.cat(logits)
                

                    cur_loss = loss_fn(logits,batch_output)
                    accumulate_loss += cur_loss.detach()
                    accumulate_num += 1
                
                   
    loss = accumulate_loss / accumulate_num
    loss_str = task + " " + split + " " + str(epoch) + "_" + str(batch) + " : " + str(loss)
    print(loss_str)
    with open(args.out_path + "eval_loss.txt", "a") as file:
        file.write('{:.3f}\n'.format(loss_str))
    

def train(args):
    print("Resume: ", args.resume)
    model = MM_Model(args)
    model = torch.nn.DataParallel(model)

    model.train()
    model.to(args.device)
    # print(args.lr)
    if args.arch == 'avg_pool':
        optimizer = torch.optim.AdamW([
            {"params": model.module.language_model.parameters(), 'lr': args.lr[0]},
            {"params": model.module.connector.parameters(), 'lr': args.lr[1]}
            ])
    elif args.arch == 'linear':
        optimizer = torch.optim.AdamW([
            {"params": model.module.language_model.parameters(), 'lr': args.lr[0]},
            {"params": model.module.linear.parameters(), 'lr': args.lr[1]}
            ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=args.decay)

    max_epochs = args.epoch
    if args.resume == "None":
        epoch = 0
        start_batch_id = 0
    else:
        # print(args.load_opt)
        if args.load_opt:
            checkpoint = torch.load(args.weights_path + args.resume + r"/checkpoint.pth")
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_batch_id = checkpoint['batch']
            epoch = checkpoint['epoch']
            # Delete the checkpoint from CUDA memory
            del checkpoint
            torch.cuda.empty_cache()  # Optional: Free up any unused memory
        else:
            epoch = 0
            start_batch_id = 0
            checkpoint = torch.load(args.weights_path + args.resume + r"/checkpoint.pth")
            start_batch_id = checkpoint['batch']
            epoch = checkpoint['epoch']
            
    tasks = args.dataset

    task = ""
    for t in tasks:
        task += t + "_"
    print(task)

    loss_fn = torch.nn.CrossEntropyLoss()
    
    val_envs = []
    # for val_task in args.val_dataset:
    #     for val_split in args.val_split:
    #         _dataset = VLN_Dataset(args, [val_split], [val_task], sample = args.val_dataset_sample)
    #         _dataloader = DataLoader(_dataset, batch_size=1, shuffle=False, collate_fn=my_collate)
    #         val_envs.append((val_task, val_split, _dataloader))

    while epoch < max_epochs:
        print(f"Epoch {epoch + 1}/{max_epochs}")
        
        dataset = VLN_Dataset(args, args.split, tasks, sample = args.dataset_sample)
        dataset.data = dataset.data[start_batch_id:]
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=my_collate)
        
        train_epoch(model, dataloader, optimizer, scheduler, loss_fn, epoch, start_batch_id, task, val_envs, args)
        epoch = epoch + 1
        start_batch_id = 0



if __name__ == "__main__":
    args = parse_args()
    if args.seed != -1:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    train(args)
