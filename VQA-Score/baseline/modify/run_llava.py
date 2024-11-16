import argparse
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image
import torch.nn.functional as F
import string
import requests
from PIL import Image
from io import BytesIO
import re

from tqdm import tqdm
import json

def image_parser(args):
    out = args.image_file.split(args.sep)
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


def eval_model(args,models,type):
    # Model
    disable_torch_init()
    with open("/home/zhuyao/Sunpeng/wxx_data_eval/data/11.5/coco_11.5_A100.json",'r') as f:
        data = json.load(f)
    

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name,device_map='auto'
    )
    coco_results = []
    for dp in tqdm(data, desc="Processing"):
        with torch.no_grad():
            qs = dp['conversations'][0]['value'].strip('\n<image>').strip()
            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            if IMAGE_PLACEHOLDER in qs:
                if model.config.mm_use_im_start_end:
                    qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
                else:
                    qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
            else:
                if model.config.mm_use_im_start_end:
                    qs = image_token_se + "\n" + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            if "llama-2" in model_name.lower():
                conv_mode = "llava_llama_2"
            elif "mistral" in model_name.lower():
                conv_mode = "mistral_instruct"
            elif "v1.6-34b" in model_name.lower():
                conv_mode = "chatml_direct"
            elif "v1" in model_name.lower():
                conv_mode = "llava_v1"
            elif "mpt" in model_name.lower():
                conv_mode = "mpt"
            else:
                conv_mode = "llava_v0"

            if args.conv_mode is not None and conv_mode != args.conv_mode:
                print(
                    "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                        conv_mode, args.conv_mode, args.conv_mode
                    )
                )
            else:
                args.conv_mode = conv_mode

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            images = load_images([dp['image']])
            image_sizes = [x.size for x in images]
            images_tensor = process_images(
                images,
                image_processor,
                model.config
            ).to(model.device, dtype=torch.float16)

            input_ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            
            outputs = tokenizer.batch_decode(output_ids['sequences'], skip_special_tokens=True)[0].strip()
            logitsf1 = output_ids.scores[0][0]
            #  llava-v1.5-7b   3869 1939  4874 694 Yes No yes no
            # logits_yn = [logitsf1[3869],logitsf1[1939],logitsf1[4874],logitsf1[694]]
            # output = F.softmax(torch.tensor(logitsf1), dim=0)
            # logits_yn_softmax = [output[3869].item(),output[1939].item(),output[4874].item(),output[694].item()]       
                 
            # llava-v1.6-34b 6599 Yes 2409 No 4130 yes 1010 no 
            if type == 1:
                logits_yn = [logitsf1[6599],logitsf1[2409],logitsf1[4130],logitsf1[1010]]
                output = F.softmax(torch.tensor(logitsf1), dim=0)
                logits_yn_softmax = [output[6599].item(),output[2409].item(),output[4130].item(),output[1010].item()]
                prob_yn_softmax = F.softmax(torch.tensor(logits_yn), dim=0).tolist()
            else:
                logits_yn = [logitsf1[3869],logitsf1[1939],logitsf1[4874],logitsf1[694]]
                output = F.softmax(torch.tensor(logitsf1), dim=0)
                logits_yn_softmax = [output[3869].item(),output[1939].item(),output[4874].item(),output[694].item()] 
                prob_yn_softmax = F.softmax(torch.tensor(logits_yn), dim=0).tolist()
            print(logits_yn_softmax)    
            print(prob_yn_softmax)
            re = {"id":dp['id']}
            re['logits_yn_softmax'] = logits_yn_softmax
            re['prob_yn_softmax'] = prob_yn_softmax
            re['answer'] = outputs
            re['expected_answer'] = dp['conversations'][1]['value'].strip().lower()
            if re['answer'].split()[0].strip(string.punctuation).lower() == re['expected_answer'].lower():
                re['ac'] = 1
            else:
                re['ac'] = 0
            coco_results.append(re)
    with open(f"/home/zhuyao/Sunpeng/wxx_data_eval/data/11.5/results/{models}.json",'w') as f :
        json.dump(coco_results,f,indent=4)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)
