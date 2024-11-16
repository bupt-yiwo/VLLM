import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from tqdm import tqdm
import string
import json
import torch.nn.functional as F

model_id = "/home/zhuyao123/sunpeng/llama3.2"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

with open("/home/zhuyao123/sunpeng/wxx/11.5/coco_11.5.json",'r') as f:
    data = json.load(f)

coco_results = []

# process
for dp in tqdm(data, desc="Processing"):
    with torch.no_grad():
        image_path = dp['image']
            
    image = Image.open(image_path)

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": dp['conversations'][0]['value'].strip('\n<image>').strip()}
        ]}
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=30,do_sample=False,return_dict_in_generate=True,output_scores=True,)
        

        logits =output.scores[0][0]
        
        answer = processor.decode(output.sequences[0])
        
        logitsf1 = logits
        logits_yn = [logitsf1[9642],logitsf1[2822],logitsf1[9891],logitsf1[2201]]

        output = F.softmax(torch.tensor(logitsf1), dim=0)
        logits_yn_softmax = [output[9642].item(),output[2822].item(),output[9891].item(),output[2201].item()]

        prob_yn_softmax = F.softmax(torch.tensor(logits_yn), dim=0).tolist()

        #Yes [9642]  yes [9891] No [2822] no [2201] 
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
    
    
with open("/home/zhuyao123/sunpeng/wxx/11.5/llama3.2_11B_11.5.json",'w') as f :
    json.dump(coco_results,f,indent=4)
    
    


