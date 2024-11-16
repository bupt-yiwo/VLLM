# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
import sys 
sys.path.append('/home/zhuyao/Sunpeng/LLaVA-main')
sys.path.append('/home/zhuyao/Sunpeng/LLaVA-main/llava')
import torch
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
import torch.nn.functional as F

from llava.eval.run_llava import eval_model

model_path = "/home/zhuyao/Sunpeng/MLLMs/4-10B/llava-v1.5-7b"
prompt = "Who is in the picture?"
image_file = "/home/zhuyao/Sunpeng/A8EB_0151_0492_1_11.png"

with torch.no_grad():
    args = type('Args', (), {
        "model_path": "/home/zhuyao/Sunpeng/MLLMs/4-10B/llava-v1.5-7b",
        "model_base": None,
        "model_name": "liuhaotian/llava-v1.5-7b",
        "query": prompt,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()

    eval_model(args,"llava-v1.5-7b",0)
