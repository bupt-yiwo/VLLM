import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

from tqdm import tqdm
import string
import json


MODEL_PATH = "/home/zhuyao/Sunpeng/MLLMs/10-20B/cogvlm2_llama3_chat-19B"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=TORCH_TYPE,
    trust_remote_code=True,
    device_map ="auto"
).eval()

text_only_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"



with open("/home/zhuyao/Sunpeng/wxx_data_eval/data/11.5/coco_11.5_A100_select.json",'r') as f:
    data = json.load(f)

coco_results = []

# process
for dp in tqdm(data, desc="Processing"):
    with torch.no_grad():
        image_path = dp['image']
        if image_path == '':
            print('You did not enter image path, the following will be a plain text conversation.')
            image = None
            text_only_first_query = True
        else:
            image = Image.open(image_path).convert('RGB')

        history = []


        query = dp['conversations'][0]['value'].strip('\n<image>').strip()

        if image is None:
            if text_only_first_query:
                query = text_only_template.format(query)
                text_only_first_query = False
            else:
                old_prompt = ''
                for _, (old_query, response) in enumerate(history):
                    old_prompt += old_query + " " + response + "\n"
                query = old_prompt + "USER: {} ASSISTANT:".format(query)
        if image is None:
            input_by_model = model.build_conversation_input_ids(
                tokenizer,
                query=query,
                history=history,
                template_version='chat'
            )
        else:
            input_by_model = model.build_conversation_input_ids(
                tokenizer,
                query=query,
                history=history,
                images=[image],
                template_version='chat'
            )
        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
            'images': [[input_by_model['images'][0].to(DEVICE).to(TORCH_TYPE)]] if image is not None else None,
        }
        gen_kwargs = {
            "max_new_tokens": 2048,
            "pad_token_id": 128002,  
        }
        outputs = model.generate(**inputs, **gen_kwargs,
                    return_dict_in_generate=True,
                    output_scores=True,)
        re = outputs.sequences[:, inputs['input_ids'].shape[1]:][0]
        answer = tokenizer.decode(re,skip_special_tokens=True)
        
        logitsf1 = outputs.scores[0][0]
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
    
with open("/home/zhuyao/Sunpeng/wxx_data_eval/data/11.5/results/cogvlm2_llama3_chat-19B.json",'w') as f :
    json.dump(coco_results,f,indent=4)
    
    
    