import torch
from transformers import AutoModelForCausalLM

from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images
import torch.nn.functional as F
import json 
from tqdm import tqdm


# specify the path to the model
model_path = "/home/zhuyao/Sunpeng/MLLMs/0-4B/deepseek_vl_1.3b_chat"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

# data
with open("/home/zhuyao/Sunpeng/wxx_data_eval/data/11.5/coco_11.5_A100.json",'r') as f:
    data = json.load(f)

coco_results = []

# process
for dp in tqdm(data, desc="Processing"):
    with torch.no_grad():
        conversation = [
            {
                "role": "User",
                "images": [dp['image']],
                "content": "<image_placeholder>" + dp['conversations'][0]['value'].strip('\n<image>').strip()
            },
            {
                "role": "Assistant",
                "content": ""
            }
        ]
        print(conversation[0]['content'])
        # load images and prepare for inputs
        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        ).to(vl_gpt.device)

        # run image encoder to get the image embeddings
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)


        ######prob

        # Yes:5661 No:3233

        ######result
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=10,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )
        answer = tokenizer.decode(outputs.sequences[0].cpu().tolist(), skip_special_tokens=True)
        print(answer)
        

        last_logits = outputs.scores[0][0]

        scores = F.softmax(last_logits, dim=0)
        # Yes:5661 No:3233
        # 5661  3233  8711  2470
        logits_yn = [last_logits[5661].item(),last_logits[3233].item(),last_logits[8711].item(),last_logits[2470].item()]
        logits_yn_softmax = [scores[5661].item(),scores[3233].item(),scores[8711].item(),scores[2470].item()]
        prob_yn_softmax = F.softmax(torch.tensor(logits_yn), dim=0).tolist()

        print(logits_yn)
        print(logits_yn_softmax)
        print(prob_yn_softmax)
        
        re = {"id":dp['id']}
        re['logits_yn_softmax'] = logits_yn_softmax
        re['prob_yn_softmax'] = prob_yn_softmax
        
        re['answer'] = answer
        re['expected_answer'] = dp['conversations'][1]['value'].strip().lower()
        if re['answer'].lower() == re['expected_answer'].lower():
            re['ac'] = 1
        else:
            re['ac'] = 0
        coco_results.append(re)
        
with open("/home/zhuyao/Sunpeng/wxx_data_eval/data/11.5/results/deepseek1.3B.json",'w') as f :
    json.dump(coco_results,f,indent=4)
    
    
