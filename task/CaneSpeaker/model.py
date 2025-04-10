from transformers import AutoTokenizer, AutoModelForCausalLM
import dataclasses
import requests
from PIL import Image
from io import BytesIO
import torch
import numpy as np
from typing import List, Optional, Tuple, Union
from peft import LoraConfig, get_peft_model, PeftModel
from enum import auto, Enum
import os
from config import IMAGE_TOKEN_INDEX, IGNORE_INDEX, STOP_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN


def image_parser(image_files):
    out = image_files.split(', ')
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()
    TINY_LLAMA = auto()
    QWEN_2 = auto()

@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].replace("<image>", "").strip()
            if 'mmtag' in self.version:
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            else:
                messages[0] = (init_role, "<image>\n" + init_msg)

        if self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version)

conv_phi_v0 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="phi",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="<|endoftext|>",
)



class MM_Model(torch.nn.Module):
    def __init__(self, args):
        super(MM_Model, self).__init__()
        self.device = args.device
        base_model = AutoModelForCausalLM.from_pretrained(args.base_model_path, trust_remote_code=True, local_files_only = True)
        base_model.to(self.device)
        
        # print(model)
        self.vision_tower = base_model.vision_tower
        self.image_processor = self.vision_tower._image_processor
        self.connector = base_model.connector
        
        self.arch = args.arch
        
        if self.arch == 'linear':
            self.linear = torch.nn.Linear(728, 14, bias = True, device=self.device)
            self.act = torch.nn.ReLU()
        
        if args.resume == "None":
            peft_config = LoraConfig(
                lora_alpha=16,
                lora_dropout=0.1,
                r=64,
                bias="none",
                target_modules = ['qkv_proj', 'out_proj', 'proj_1', 'proj_2'],
                task_type="CAUSAL_LM"
            )
            self.language_model = get_peft_model(base_model.language_model, peft_config)
            self.language_model.print_trainable_parameters()
            if self.arch == 'linear':
                input_dim = 728
                output_dim = 14
                avg_group_size = input_dim // output_dim
                with torch.no_grad():
                    self.linear.weight.fill_(0)
                    self.linear.bias.fill_(0)
                    for i in range(output_dim):
                        start_idx = i * avg_group_size
                        end_idx = start_idx + avg_group_size
                        self.linear.weight[i, start_idx:end_idx] = 1.0 / avg_group_size
                        # Add small random noise
                        self.linear.weight[i, start_idx:end_idx] += torch.randn(avg_group_size,device=self.device) * 0.01
            
        else: # path
            path = args.weights_path + args.resume
            self.language_model = PeftModel.from_pretrained(base_model.language_model, path + "/peft", is_trainable = True)
            self.language_model.print_trainable_parameters()
            if self.arch == 'avg_pool':
                checkpoint = torch.load(path + "/connector.pth")
                self.connector.load_state_dict(checkpoint['connector'])
            elif self.arch == 'linear':
                checkpoint = torch.load(path + "/linear.pth")
                self.linear.load_state_dict(checkpoint['linear'])              
            
        self.config = base_model.config
        self.tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=False, model_max_length = self.config.tokenizer_model_max_length, padding_side = self.config.tokenizer_padding_side, local_files_only = True)
        self.dropout = torch.nn.Dropout(p = args.dropout)


        
    def prepare_inputs_labels_for_multimodal(self, input_ids, image_features = None, position_ids = None, attention_mask = None, past_key_values = None, labels = None):
        if self.vision_tower is None or image_features is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels


        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.language_model.get_input_embeddings()(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.language_model.get_input_embeddings()(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            # for x in cur_new_input_embeds:
            #     print(x.shape)

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
    
    def encode_images(self, images):
        images_tensor = np.array(self.image_processor(images)['pixel_values'])
        images_tensor = torch.tensor(images_tensor).to(self.device) # [batch, 3, 384, 384]
        # all at once
        # image_features = self.vision_tower(images_tensor)
        
        # one at a time
        # image_features = []
        # for i in range(images_tensor.shape[0]):
        #     feat = self.vision_tower(images_tensor[i].unsqueeze(0))
        #     # print(images_tensor[i].unsqueeze(0).shape)
        #     # print(feat.shape)
        #     image_features.append(feat)
        # image_features = torch.cat(image_features)
        
        # n at a time
        image_features = []
        batch_size = 2
        for i in range(0, images_tensor.shape[0], batch_size):
            batch_end = min(i + batch_size, images_tensor.shape[0])
            batch = images_tensor[i:batch_end]
            feat = self.vision_tower(batch)
            image_features.append(feat)
        image_features = torch.cat(image_features)
        
        if self.arch == 'avg_pool':
            ori_len = image_features.shape[1]
            new_len = 14
            image_features = torch.nn.functional.avg_pool1d(image_features.permute(0,2,1), kernel_size = ori_len//new_len, stride = ori_len//new_len).permute(0,2,1)
        elif self.arch == 'linear':
                image_features = self.connector(image_features)
        return image_features
    
    def format_prompt(self, prompt):
        conv = conv_phi_v0.copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        return prompt
    
    def tokenize(self, prompt):
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        return input_ids
    
    def forward(self, input_ids, image_features_pre_proj, attention_mask = None):
        if self.arch == 'avg_pool':
            image_features = self.connector(image_features_pre_proj)
        elif self.arch == 'linear':
            # with torch.no_grad():
            #     image_features = self.connector(image_features_pre_proj)
            image_features = image_features_pre_proj
            image_features_shape = image_features.shape
            image_features = image_features.permute(0, 2, 1)  # (batchsize, embeddim, 728)
            image_features = image_features.contiguous().view(-1, 728)
            image_features = self.linear(image_features)
            # image_features = self.act(image_features)
            image_features = image_features.view(image_features_shape[0], image_features_shape[2], 14)
            image_features = image_features.permute(0, 2, 1)
            
        image_features = self.dropout(image_features)
        # print("Image features: ", image_features.shape)
        (
            _,
            _,
            attention_mask,
            _,
            inputs_embeds,
            _
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids = input_ids,
            image_features = image_features,
            attention_mask = attention_mask,
        )
        
        # print("Input embeds: ", inputs_embeds.shape)
        
        return self.language_model.forward(inputs_embeds = inputs_embeds, attention_mask = attention_mask)['logits'], attention_mask
    
    def generate(self, input_ids, image_features_pre_proj, attention_mask = None, skip_special_tokens = True, **kwargs):
        if self.arch == 'avg_pool':
            image_features = self.connector(image_features_pre_proj)
        elif self.arch == 'linear':
            # with torch.no_grad():
            #     image_features = self.connector(image_features_pre_proj)
            image_features = image_features_pre_proj
            image_features_shape = image_features.shape
            image_features = image_features.permute(0, 2, 1)  # (batchsize, embeddim, 728)
            image_features = image_features.contiguous().view(-1, 728)
            image_features = self.linear(image_features)
            # image_features = self.act(image_features)
            image_features = image_features.view(image_features_shape[0], image_features_shape[2], 14)
            image_features = image_features.permute(0, 2, 1)
            
        image_features = self.dropout(image_features)
        
        (
            _,
            _,
            attention_mask,
            _,
            inputs_embeds,
            _
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids = input_ids,
            image_features = image_features,
            attention_mask = attention_mask,
        )
        

        output_ids = self.language_model.generate(inputs_embeds = inputs_embeds, attention_mask = attention_mask, **kwargs)
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=skip_special_tokens)
    
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.language_model.save_pretrained(path + "/peft", save_embedding_layers=False)
        if self.arch == 'avg_pool':
            torch.save({
                        'connector': self.connector.state_dict(),                 
                    }, path +"/connector.pth")
        elif self.arch == 'linear':
            torch.save({
                        'linear': self.linear.state_dict(),                 
                    }, path +"/linear.pth")
    
    