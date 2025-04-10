import torch
import transformers
import json
import time

prompt = "Please rephrase the above route description into a much shorter and concise command. "
prompt += "Eliminate direct address to the user (avoid using 'you') and do not start with \"From ...\" or \"Facing...\". "
prompt += "Generate instructions that sounds natural. Use smooth phrases. "
prompt += "You can leave out redundant information, but the focus should be on clarity, precision and natural expression. "
prompt += "For example, instead of \"Turn right, move forward, stop. \", you should say \"Turn right and move to something/someplace. Stop in front of something/someplace. \""
prompt += "You only need to reply the rephrased description. "
prompt += '\n\n'
prompt += "Here are some examples: \n"
prompt += "Walk out of the bathroom and wait on the stairs that are on the right. "
prompt += '\n'
prompt += "Go into the archway to the left of the room with the dining table into the room with the circle table in the middle, make a right towards the front door, take two steps up the stairs onto the level and stop. "
prompt += '\n'
prompt += "Move to the mirror and take a left before you reach it. Walk straight and out onto the patio. Stop once you are on the patio. "
prompt += '\n'
prompt += "Walk past the laundry room into the office at the end of the hall. Wait in the office between the love seat and chair. "
prompt += '\n'


if __name__ == "__main__":
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    splits = ['val_seen', 'val_unseen']
    path_length = [0, 20]


    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    
    for split in splits:
        with open(f'./task/data/RxR/RxR_{split}.jsonl') as f:
            for count, line in enumerate(f):
                pass
            length  = count

        cur_data = []
        with open(f'./task/data/RxR/RxR_{split}.jsonl') as f:
            start_time = time.time()
            # length = len(f)
            for count, line in enumerate(f):
                tmp = json.loads(line)
                if len(tmp['path']) < path_length[0] or len(tmp['path']) > path_length[1]:
                        continue
                
                if "en" in tmp['language']:
                    print("Current: ", count, " / ", length)
                    cur_run_time = int(time.time()-start_time)
                    est_run_time = int(cur_run_time*(length-count - 1)/(count + 1))
                    print(f"Running: {cur_run_time//3600} h {(cur_run_time%3600)//60} min {cur_run_time%60} sec, Estimating: {est_run_time//3600} h {(est_run_time%3600)//60} min {est_run_time%60}")

                    del tmp['timed_instruction']
                    del tmp['edit_distance']
                    del tmp['annotator_id']
                    # print(tmp)
                    messages = [
                        {"role": "system", "content": "You are a chatbot who give helpful responses to the user!"},
                        {"role": "user", "content": tmp['instruction'] + '\n' + prompt},
                    ]
                    # print(messages[1]['content'])

                    outputs = pipeline(
                        messages,
                        max_new_tokens=512,
                    )

                    instr = outputs[0]["generated_text"][-1]['content']
                    instr = instr.replace('\n\n', ' ')
                    instr = instr.replace('\n', ' ')
                    instr = instr.replace('\"', ' ')
                    instr = instr.replace('  ', ' ')

                    print(instr)
                
                    tmp['instructions'] = [instr]
                    del tmp['instruction']
                    
                    cur_data.append(tmp)



        with open(f'/./task/data/Rx2R/Rx2R_{split}.json', 'a') as file:
            json.dump(cur_data, file, indent=4)

