# test.py
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from tqdm import tqdm
import string
import json


model = AutoModel.from_pretrained('/home/zhuyao/Sunpeng/MLLMs/0-4B/MiniCPM_V_2', trust_remote_code=True, torch_dtype=torch.bfloat16, local_files_only=True)
# For Nvidia GPUs support BF16 (like A100, H100, RTX3090)
model = model.to(device='cuda', dtype=torch.bfloat16)
# For Nvidia GPUs do NOT support BF16 (like V100, T4, RTX2080)
#model = model.to(device='cuda', dtype=torch.float16)
# For Mac with MPS (Apple silicon or AMD GPUs).
# Run with `PYTORCH_ENABLE_MPS_FALLBACK=1 python test.py`
#model = model.to(device='mps', dtype=torch.float16)
#No 6460 no 3768  Yes 24346 yes 14081
tokenizer = AutoTokenizer.from_pretrained('/home/zhuyao/Sunpeng/MLLMs/0-4B/MiniCPM_V_2', trust_remote_code=True, local_files_only=True)
model.eval()


with open("/home/zhuyao/Sunpeng/wxx_data_eval/data/11.5/coco_11.5_A100.json",'r') as f:
    data = json.load(f)

coco_results = []
# process
for dp in tqdm(data, desc="Processing"):
        
    image = Image.open(dp['image']).convert('RGB')
    question = dp['conversations'][0]['value'].strip('\n<image>').strip()
    msgs = [{'role': 'user', 'content': question}]

    res, context, _,logitsf1 = model.chat(
        image=image,
        msgs=msgs,
        context=None,
        tokenizer=tokenizer,
        sampling=False
    )

    print(res)
    print(context)

    logits_yn = [logitsf1[24346],logitsf1[6460],logitsf1[14081],logitsf1[3768]]

    output = F.softmax(torch.tensor(logitsf1), dim=0)
    logits_yn_softmax = [output[24346].item(),output[6460].item(),output[14081].item(),output[3768].item()]

    prob_yn_softmax = F.softmax(torch.tensor(logits_yn), dim=0).tolist()
    # Yes No yes no 24346 6460 14081 3768
    # print(f'User: {question}\nAssistant: {response}')
    # print(logits_yn)
    print(logits_yn_softmax)
    print(prob_yn_softmax)
    re = {"id":dp['id']}
    re['logits_yn_softmax'] = logits_yn_softmax
    re['prob_yn_softmax'] = prob_yn_softmax
    re['answer'] = res
    re['expected_answer'] = dp['conversations'][1]['value'].strip().lower()
    if re['answer'].split()[0].strip(string.punctuation).lower() == re['expected_answer'].lower():
        re['ac'] = 1
    else:
        re['ac'] = 0
    coco_results.append(re)

with open("/home/zhuyao/Sunpeng/wxx_data_eval/data/11.5/results/MiniCPM_V_2.json",'w') as f :
    json.dump(coco_results,f,indent=4)
    
#缺：MINICPM_V_2
#缺：MINICPM_8B
#缺：cogvlm2_llama3_chat-19B 2卡
#缺：internvl2_26B 3卡
#缺：qwen2vl_72B 6卡