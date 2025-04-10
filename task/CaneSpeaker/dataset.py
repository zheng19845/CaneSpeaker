import json
from torch.utils.data import Dataset
from env import R2R_Planner
from config import  DEFAULT_IMAGE_TOKEN
import random
import cv2
import re
import numpy
from PIL import Image
from args_parser import parse_args

import heapq

task_prompt = {
        'R2R':      ["Describe this route, note a couple landmarks.",
                    "Give accurate directions to the endpoint.",
                    "Guide the path to the destination.",
                    "Outline the path to the target place.",
                    "Summarize the way to the final location."],
        'Rx2R':      ["Describe the route in detail, focus on objects and landmarks.",
                    "Give directions to the endpoint by pointing out conspicuous things.",
                    "Identify notable items along the way.",
                    "Summarize the path to the target place by pointing out some obvious objects.",
                    "Provide a detailed path."],
        'RxR':      ["Guide me with very detailed steps.",
                    "Describe the route to me with specific landmarks.",
                    "Provide me a fine-grained path to the goal, including objects and landmarks.",
                    "Explain to me the detailed way to the destination by pointing out objects you see.",
                    "Give me precise directions, noting landmarks."],
        'REVERIE':  ["Navigate to the room and identify the object.",
                    "Provide directions and point out the target.",
                    "Guide me to the room and specify the object.",
                    "Explain how to find the room and the target object.",
                    "Describe the path and identify the object in the room."],
        'SOON':     ["Describe the target and the environment of the destination.",
                    "Give detailed description of the target and the room.",
                    "Tell me about the goal object and its surroundings.",
                    "Point out the target and give region descriptions.",
                    "Provide descriptions for the destination and the object in it."],
        'caption':  ["Given the image below, generate a descriptive caption.",
                    "Describe what you see in the following image.",
                    "Write a detailed caption for this image.",
                    "What is happening in the image shown? Provide a caption.",
                    "Create a natural language description for the scene depicted in the image.",
                    "Generate a caption that summarizes the content of this image.",
                    "Based on the image provided, what would be an appropriate caption?",
                    "Describe the main elements and activities in the image with a concise caption.",
                    "Provide a caption that captures the essence of the image.",
                    "Generate a sentence that best describes the scene in this image."]
    
    }

num_sent_template = ["in {} sentences. ",
                     "in {} phrases. ",
                     "in about {} sentences. ",
                     "in about {} phrases. ",
                     "in at least {} sentences. ",
                     "in at least {} phrases. ",
                     "in no more than {} sentences. ",
                     "in no more than {} phrases. ",
                     "using {} sentences. ",
                     "using {} phrases. ",
                     "using about {} sentences. ",
                     "using about {} phrases. ",
                     "using at least {} sentences. ",
                     "using at least {} phrases. ",
                     "using no more than {} sentences. ",
                     "using no more than {} phrases. ",
                     "Use {} sentences. ",
                     "Use {} phrases. ",
                     "Use about {} sentences. ",
                     "Use about {} phrases. ",
                     "Use at least {} sentences. ",
                     "Use at least {} phrases. ",
                     "Use no more than {} sentences. ",
                     "Use no more than {} phrases. ",
                     ]

def merge_lists(lists):
    # Priority queue (min-heap)
    priority_queue = []

    # Initialize the priority queue with the first element of each list
    for i, lst in enumerate(lists):
        if lst:  # Check if the list is not empty
            heapq.heappush(priority_queue, (i * 1e-7, lst[0], i, 0))

    merged_list = []

    while priority_queue:
        ratio, value, list_index, element_index = heapq.heappop(priority_queue)
        merged_list.append(value)
        # print(value)
        # If the list has more elements, push the next element into the priority queue
        if element_index + 1 < len(lists[list_index]):
            next_value = lists[list_index][element_index + 1]
            next_ratio = (element_index + 1) / len(lists[list_index]) + list_index*1e-7
            heapq.heappush(priority_queue, (next_ratio, next_value, list_index, element_index + 1))

    return merged_list


def load_datasets(splits, root_dir = r"./tasks/data/", tasks= ["R2R"], sample = False, max_path = 16):
    ori_data = []
    ori_tasks = tasks
    tasks = []
    for task in ori_tasks:
        if task == "RxR":
            for split in splits:
                if split in ['train', 'val_seen', 'val_unseen', 'test']:
                    with open(root_dir + task+ '/'+ task +'_%s.jsonl' % split) as f:
                        cur_data = []
                        for line in f:
                            tmp = json.loads(line)
                            tmp['instructions'] = [tmp['instruction']]
                            if "en" in tmp['language']:
                            # if "en" in tmp['language']:
                                # print(tmp['language'])
                                cur_data.append(tmp)
                        # print(len(cur_data))
                    ori_data.append(cur_data)
                    tasks.append(task)
                    # print(len(cur_data))
                else:
                    with open(root_dir + task+ '/'+ task +'_%s.json' % split) as f:
                        cur_data = json.load(f)
                    ori_data.append(cur_data)
                    tasks.append(task)
        elif "SOON" in task:                  
            for split in splits:
                if split in ['train', 'val_seen', 'val_unseen', 'test']:  
                    with open(root_dir + task+ '/'+ task +'_%s.json' % split) as f:
                        cur_data = json.load(f)
                    new_data = []
                    for item in cur_data:
                        paths = random.sample(item['path'], k = 1)
                        for path in paths:
                            new_item = {}
                            new_item['instructions'] = [item['instructions'][0][4]]
                            new_item['scan'] = item['bboxes'][0]['scan']
                            new_item['path'] = path[random.randint(-6, -3):]
                            new_item['heading'] = random.randint(0,35) / 18.0 * numpy.pi
                            new_item['path_id'] = new_item['scan'] + new_item['path'][-1]
                            new_item['object'] = item['bboxes'][0]['obj_name']
                            new_data.append(new_item)
                    ori_data.append(new_data)
                    tasks.append(task)
                else:
                    with open(root_dir + task+ '/'+ task +'_%s.json' % split) as f:
                        cur_data = json.load(f)
                    ori_data.append(cur_data)
                    tasks.append(task)
        else:
            for split in splits:
                # assert split in ['train', 'val_seen', 'val_unseen', 'test']
                with open(root_dir + task+ '/'+ task +'_%s.json' % split) as f:
                    cur_data = json.load(f)
                    # print(len(cur_data))
                ori_data.append(cur_data)
                tasks.append(task)
    data = [[] for i in range(len(ori_data))]
    scan = []
    min_size = 99999999999
    for i in range(len(ori_data)):
        dataset = ori_data[i]
        for item in dataset:
            for instr in item['instructions']:
                if 'REVERIE' in tasks[i]:
                    if 'object' in item:
                        data[i].append({
                        'scan' : item['scan'],
                        'path' : item['path'],
                        'heading' : item['heading'],
                        'instruction' : instr + ". ",
                        'id': item['id'],
                        'object': item['object'],
                        # "distance" : item["distance"],
                        'type' : 'REVERIE'
                    })
                    else:
                        data[i].append({
                        'scan' : item['scan'],
                        'path' : item['path'],
                        'heading' : item['heading'],
                        'instruction' : instr + ". ",
                        'id': item['path_id'],
                        # "distance" : item["distance"],
                        'type' : 'REVERIE'
                    })
                elif tasks[i] == 'Speaker':
                    data[i].append({
                        'scan' : item['scan'],
                        'path' : item['path'],
                        'heading' : item['heading'],
                        'instruction' : instr,
                        'id': item['path_id'],
                        # "distance" : item["distance"],
                        'type' : 'Rx2R'
                    })
                elif tasks[i] == 'Marky':
                    data[i].append({
                        'scan' : item['scan'],
                        'path' : item['path'],
                        'heading' : item['heading'],
                        'instruction' : instr,
                        'id': item['path_id'],
                        # "distance" : item["distance"],
                        'type' : 'RxR'
                    })
                elif tasks[i] == 'Rx2R':
                    if len(item['path']) > 8:
                        continue
                    data[i].append({
                        'scan' : item['scan'],
                        'path' : item['path'],
                        'heading' : item['heading'],
                        'instruction' : instr,
                        'id': item['path_id'],
                        # "distance" : item["distance"],
                        'type' : 'Rx2R'
                    })
                elif 'SOON' in tasks[i]:
                    if 'object' in item:
                        if item['object'] == None:
                            continue
                        data[i].append({
                            'scan' : item['scan'],
                            'path' : item['path'],
                            'heading' : item['heading'],
                            'instruction' : instr,
                            'id': item['path_id'] + '_' + item['object'],
                            'object': item['object'],
                            # "distance" : item["distance"],
                            'type' : "SOON"
                        })
                    else:
                        data[i].append({
                            'scan' : item['scan'],
                            'path' : item['path'],
                            'heading' : item['heading'],
                            'instruction' : instr,
                            'id': item['path_id'],
                            # "distance" : item["distance"],
                            'type' : "SOON"
                        })
                else: 
                    if len(item['path']) > max_path:
                        continue
                    data[i].append({
                        'scan' : item['scan'],
                        'path' : item['path'],
                        'heading' : item['heading'],
                        'instruction' : instr,
                        'id': item['path_id'],
                        # "distance" : item["distance"],
                        'type' : tasks[i]
                    })
            scan.append(item['scan'])
        
        if tasks[i] == 'Rx2R':
            # data[i] = random.sample(data[i], k = 6000)
            pass
        if tasks[i] == 'RxR':
            # data[i] = random.sample(data[i], k = 4000)
            pass
        min_size = min(len(data[i]),min_size)
    
    result = []
    for dataset in data:
        if type(sample) == bool:
            if sample and len(dataset)>min_size:
                sampled = random.sample(dataset,min_size)
            else:
                sampled = dataset
        else:
            sampled = random.sample(dataset, int(len(dataset) * sample))
        # print(len(sampled))
        # result.append(sampled)
        result += sampled
    # result = merge_lists(result)


    return result, set(scan)

def load_dataset_path(path, task = "R2R"):
    data = []
    scan = []
    with open(path) as f:
        ori_data = json.load(f)
    for item in ori_data:
        for instr in item['instructions']:
            data.append({
                        'scan' : item['scan'],
                        'path' : item['path'],
                        'heading' : item['heading'],
                        'instruction' : instr,
                        'id': item['path_id'],
                        'type' : task
                        })
            scan.append(item['scan'])
    

    return data, set(scan)

def load_dataset_caption(root_dir = r"./tasks/data/", splits = ['train']):
    from pycocotools.coco import COCO
    import os
    split = splits[0]
    ann_file = root_dir + "Caption/annotations/captions_" + split + "2014.json"
    coco = COCO(ann_file)
    
    ids = list(coco.imgToAnns.keys())
    data = []
    for idx in range(len(ids)):
        img_id = ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        captions = [tmp['caption'] for tmp in random.sample(anns, k=1)]
        for caption in captions:
            # Load image
            img_info = coco.loadImgs(img_id)[0]
            img_path = os.path.join(root_dir + r"Caption/" + split + "2014/", img_info['file_name'])
            if not (caption.endswith('.') or caption.endswith('. ')):
                if caption.endswith(' '):
                    caption = caption[:-1]
                caption = caption + '. '
            if caption.endswith('.'):
                caption = caption + ' '
            data.append({
                'id':img_id,
                'type':'caption',
                'img_path':img_path,
                'caption': caption
            })
    data = random.sample(data, k = 20000)
    
    

    return data

class VLN_Dataset(Dataset):
    def __init__(self, args, splits, tasks = ["R2R"], sample=False):
        if isinstance(splits, str):
            self.data, self.scan = load_dataset_path(splits, tasks[0])
        else:
            if "Caption" in tasks:
                tasks.remove("Caption")
                self.data = []
                self.scan = []
                self.data, self.scan = load_datasets(splits, root_dir=args.dataset_path ,tasks = tasks, sample=sample,max_path=args.max_path)
                self.data += load_dataset_caption(root_dir = args.dataset_path, splits = splits)
                # print(len(self.data))

                tasks.append("Caption")
            else:
                self.data, self.scan = load_datasets(splits, root_dir=args.dataset_path ,tasks = tasks, sample=sample)
        if args.seed != -1: 
            random.shuffle(self.data)
        print("Loaded dataset: ", tasks, ", splits: ", splits, ", size: ", len(self.data))
        self.planner = R2R_Planner(self.scan, args.root_path)


    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index, visualize = False, prompt_idx = None, pano = 0.7, obj = 0.8):
        item = self.data[index]
        
        if item['type'] == 'caption':
            prompt = ""
            prompt += DEFAULT_IMAGE_TOKEN + "\n"
            if prompt_idx == -1:
                pass
            else:
                if prompt_idx is not None:
                    prompt += task_prompt[item['type']][prompt_idx]
                else:
                    prompt += random.choice(task_prompt[item['type']])
            image = Image.open(item['img_path']).convert('RGB')
            return prompt, [image], item['caption'], item['type']
        
        
        # print(item)
        seq, mask = self.planner.generate_actions(item, False)
        prompt = ""
        images = []
        # print(seq)
        if ('REVERIE' in item['type'] or 'SOON' in item['type']) and (random.random() >=  pano):
            for i in range(len(mask)):
                if mask[i] == 0: # image
                    prompt += DEFAULT_IMAGE_TOKEN + "\n"
                    images.append(seq[i])
                    if visualize:
                        img = cv2.cvtColor(numpy.asarray(seq[i]),cv2.COLOR_RGB2BGR)
                        cv2.imshow("RGB", img)
                        cv2.waitKey(0)
                elif mask[i] == 1:
                    prompt += seq[i] + "\n"
                    if seq[i] == 'stop':
                        break
        else:            
            for i in range(len(mask)):
                if mask[i] == 0: # image
                    prompt += DEFAULT_IMAGE_TOKEN + "\n"
                    images.append(seq[i])
                    if visualize:
                        img = cv2.cvtColor(numpy.asarray(seq[i]),cv2.COLOR_RGB2BGR)
                        cv2.imshow("RGB", img)
                        cv2.waitKey(0)
                elif mask[i] == 1:
                    prompt += seq[i] + "\n"
        
        if isinstance(prompt_idx, str):
            prompt += prompt_idx
        else:
            if prompt_idx == -1:
                pass
            else:
                if prompt_idx is not None:
                    prompt += task_prompt[item['type']][prompt_idx]
                else:
                    prompt += random.choice(task_prompt[item['type']])
        
        # if isinstance(num_sent, str):
        #     if num_sent == "":
        #         pass
        #     elif num_sent[0].isupper():
        #         prompt = prompt + " " + num_sent
        #     else:
        #         prompt = prompt[:-1] + " " + num_sent
        # elif num_sent != False:
        #     if (num_sent == True) and random.random() < 0.1:
        #         pass
        #     else:
        #         template = random.choice(num_sent_template)
        #         instr = item['instruction']
                
        #         if num_sent != 1:
        #             num = num_sent
        #         else:
        #             if "sentence" in template:
        #                 num = instr.count('.') + instr.count(';') + instr.count('?') + instr.count('!') - 2 * instr.count('...')
        #             else:
        #                 num = instr.count('.') + instr.count(';') + instr.count('?') + instr.count('!') + instr.count(',') - 2 * instr.count('...')

        #         if "least" in template:
        #             num = max(1, num - random.randint(0, 4))
        #         elif "about" in template:
        #             num = max(2, num + random.randint(-1, 1))
        #         elif "no more" in template:
        #             num = num + random.randint(0, 3)
                    
        #         if random.randint(0,1):
        #             template = template.format(num)
        #         else:
        #             ran = random.choice([[-1,0],[-1,1],[0,1]])
        #             template = template.format(str(ran[0]+num)+random.choice([" to ", "-"]) + str(ran[1]+num))
                
        #         if template[0].isupper():
        #             prompt = prompt + " " + template
        #         else:
        #             prompt = prompt[:-1] + " " + template
            
        item['instruction'] = item['instruction'].replace("...", '.') 
        if item['instruction'].isupper():
            item['instruction'] = item['instruction'].capitalize()
        if item['instruction'][0] == '\"':
            item['instruction'] = item['instruction'].replace('\"', '') 
        if random.random() <= obj:  
            if not isinstance(prompt_idx, str):          
                if 'object' in item:
                    if 'object' in prompt:
                        prompt = prompt.replace('object', item['object'])
                    elif 'target' in prompt:
                        prompt = prompt.replace('target', item['object'])
            
        
        # if item['type'] == "REVERIE":
        #     print(item['instruction'])
        #     pattern = r'\son level (one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s*'
        #     item['instruction'] = re.sub(pattern, ' ', item['instruction'])
        #     item['instruction'] = ' '.join(item['instruction'].split())
        
        
        # for i in range(len(seq)):
        #     if isinstance(seq[i],str):
        #         seq[i] = r"<action>" + seq[i] + r"</action>"
        # # print(seq)
        # seq.insert(0, "<instruction>" + item['instruction'] + "</instruction>\n" )
        # mask.insert(0, 0)
        
        return prompt, images, item['instruction'], item['type']



if __name__ == "__main__":
    import numpy as np
    import os

    args = parse_args()
    args.max_path = 99
    tasks = ["R2R"]
    splits = ["val_seen"]
    args.seed = -1
    
    if args.seed != -1:
        random.seed(args.seed)
    dataset = VLN_Dataset(args, splits= splits, tasks=tasks)
    
    test_id =  5177
    for idx in range(len(dataset.data)):
        if dataset.data[idx]['id'] == test_id:
            test_id = idx
            print(idx)
            break
    dataset.data[test_id]['type'] = 'R2R'
    prompt, images, instr, task_type = dataset.__getitem__(idx, prompt_idx = 0, pano = 0, obj = 0, visualize=True)
