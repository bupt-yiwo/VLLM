# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
import sys 
sys.path.append('/home/zhuyao/Sunpeng/LLaVA-NeXT-main')
sys.path.append('/home/zhuyao/Sunpeng/LLaVA-NeXT-main/llava')

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch
import warnings
import torch.nn.functional as F
from tqdm import tqdm
import string
import json

#Yes [9693]  yes [9693] No [2152] no [2152]

warnings.filterwarnings("ignore")
pretrained = "/home/zhuyao/Sunpeng/MLLMs/0-4B/LLaVA_OneVision_0.5B"
model_name = "llava_qwen"
device = "cuda"
device_map = "cuda"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation="eager")  # Add any other thing you want to pass in llava_model_args

model.eval()


with open("/home/zhuyao/Sunpeng/wxx_data_eval/data/11.5/coco_11.5_A100.json",'r') as f:
    data = json.load(f)

coco_results = []
# process
for dp in tqdm(data, desc="Processing"):
    image = Image.open(dp['image'])
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    question = DEFAULT_IMAGE_TOKEN + "\n" + dp['conversations'][0]['value'].strip('\n<image>').strip()
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [image.size]


    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
        return_dict_in_generate=True,
        output_scores=True,
    )
    text_outputs = tokenizer.batch_decode(cont['sequences'], skip_special_tokens=True)[0]
    print(text_outputs)
    logitsf1 = cont['scores'][0][0]
    logits_yn = [logitsf1[9454],logitsf1[2753],logitsf1[9693],logitsf1[2152]]

    output = F.softmax(torch.tensor(logitsf1), dim=0)
    logits_yn_softmax = [output[9454].item(),output[2753].item(),output[9693].item(),output[2152].item()]

    prob_yn_softmax = F.softmax(torch.tensor(logits_yn), dim=0).tolist()

    #Yes [9454]  yes [9693] No [2753] no [2152] 
    print(logits_yn_softmax)
    print(prob_yn_softmax)
    re = {"id":dp['id']}
    re['logits_yn_softmax'] = logits_yn_softmax
    re['prob_yn_softmax'] = prob_yn_softmax
    re['answer'] = text_outputs
    re['expected_answer'] = dp['conversations'][1]['value'].strip().lower()
    if re['answer'].split()[0].strip(string.punctuation).lower() == re['expected_answer'].lower():
        re['ac'] = 1
    else:
        re['ac'] = 0
    coco_results.append(re)

with open("/home/zhuyao/Sunpeng/wxx_data_eval/data/11.5/results/llava_onversion_0.5B.json",'w') as f :
    json.dump(coco_results,f,indent=4)