import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
from tqdm import tqdm
import string
import json


device = "cuda"

tokenizer = AutoTokenizer.from_pretrained("/home/zhuyao/Sunpeng/MLLMs/4-10B/GLM4V_9B", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    "/home/zhuyao/Sunpeng/MLLMs/4-10B/GLM4V_9B",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map = "auto",
    trust_remote_code=True
).eval()

gen_kwargs = {"max_length": 2500, "do_sample": False}

with open("/home/zhuyao/Sunpeng/wxx_data_eval/data/11.5/coco_11.5_A100.json",'r') as f:
    data = json.load(f)

coco_results = []
# process
for dp in tqdm(data, desc="Processing"):

    query = dp['conversations'][0]['value'].strip('\n<image>').strip()
    image = Image.open(dp['image']).convert('RGB')
    inputs = tokenizer.apply_chat_template([{"role": "user", "image": image, "content": query}],
                                        add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                        return_dict=True)  # chat mode

    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs,         
                                return_dict_in_generate=True,
                                output_scores=True,**gen_kwargs)
        re = outputs['sequences'][:, inputs['input_ids'].shape[1]:][0]
        answer = tokenizer.decode(re,skip_special_tokens=True)
        
        logitsf1 = outputs['scores'][0][0]
        logits_yn = [logitsf1[9450],logitsf1[2753],logitsf1[9689],logitsf1[2152]]

        output = F.softmax(torch.tensor(logitsf1), dim=0)
        logits_yn_softmax = [output[9450].item(),output[2753].item(),output[9689].item(),output[2152].item()]

        prob_yn_softmax = F.softmax(torch.tensor(logits_yn), dim=0).tolist()

        #Yes [9450]  yes [9689] No [2753] no [2152] 
        print(logits_yn_softmax)
        print(prob_yn_softmax)
        print(answer)
        re = {"id":dp['id']}
        re['logits_yn_softmax'] = logits_yn_softmax
        re['prob_yn_softmax'] = prob_yn_softmax
        re['answer'] = answer
        re['expected_answer'] = dp['conversations'][1]['value'].strip().lower()
        if re['answer'].split()[0].strip(string.punctuation).lower() == re['expected_answer'].lower():
            re['ac'] = 1
        else:
            re['ac'] = 0
        coco_results.append(re)

with open("/home/zhuyao/Sunpeng/wxx_data_eval/data/11.5/results/GLM4V_9B.json",'w') as f :
    json.dump(coco_results,f,indent=4)