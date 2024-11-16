from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch.nn.functional as F
import json 
from tqdm import tqdm
import torch
import string



model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/home/zhuyao/Sunpeng/MLLMs/4-10B/qwen2vl_7b", torch_dtype="auto", device_map="cuda"
)


processor = AutoProcessor.from_pretrained("/home/zhuyao/Sunpeng/MLLMs/4-10B/qwen2vl_7b")

with open("/home/zhuyao/Sunpeng/wxx_data_eval/data/11.5/coco_11.5_A100.json",'r') as f:
    data = json.load(f)

coco_results = []

# process
for dp in tqdm(data, desc="Processing"):
    with torch.no_grad():
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": dp['image'],
                    },
                    {"type": "text", "text": dp['conversations'][0]['value'].strip('\n<image>').strip()},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample = False, return_dict_in_generate=True,output_scores=True,)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids.sequences)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text)
        
        #########logits

        last_logits = generated_ids.scores[0][0]

        output = F.softmax(last_logits, dim=0)

        logits_yn = [last_logits[9454].item(),last_logits[2753].item(),last_logits[9693].item(),last_logits[2152].item()]
        logits_yn_softmax = [output[9454].item(),output[2753].item(),output[9693].item(),output[2152].item()]
        prob_yn_softmax = F.softmax(torch.tensor(logits_yn), dim=0).tolist()

        print(logits_yn)
        print(logits_yn_softmax)
        print(prob_yn_softmax)
        
        re = {"id":dp['id']}
        re['logits_yn_softmax'] = logits_yn_softmax
        re['prob_yn_softmax'] = prob_yn_softmax
        re['answer'] = output_text[0]
        re['expected_answer'] = dp['conversations'][1]['value'].strip().lower()
        if re['answer'].split()[0].strip(string.punctuation).lower() == re['expected_answer'].lower():
            re['ac'] = 1
        else:
            re['ac'] = 0
        coco_results.append(re)
        #Yes 9454 No 2753 yes 9693 no 2152
        #Yes 9454 No 2753 yes 9693 no 2152
        
with open("/home/zhuyao/Sunpeng/wxx_data_eval/data/11.5/results/qwen2vl_7B.json",'w') as f :
    json.dump(coco_results,f,indent=4)
    
