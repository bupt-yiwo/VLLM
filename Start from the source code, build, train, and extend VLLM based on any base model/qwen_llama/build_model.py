from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from peft import PeftModel, PeftConfig


model_qwen = Qwen2VLForConditionalGeneration.from_pretrained(
    "/home/zhuyao/Sunpeng/models/qwen_2B_instruct", torch_dtype="auto", device_map="cpu"
)


import torch.nn as nn
new_mlp = nn.Sequential(
    nn.Linear(in_features=5120, out_features=5120, bias=True),
    nn.GELU(approximate='none'),
    nn.Linear(in_features=5120, out_features=2048, bias=True) 
)
model_qwen.visual.merger.mlp = new_mlp


import torch
from transformers import pipeline
from transformers import AutoTokenizer,AutoModelForCausalLM

model_id = "/home/zhuyao/Sunpeng/models/llama3.2_1B"
model_llama = AutoModelForCausalLM.from_pretrained(
                                            model_id,
                                            torch_dtype=torch.bfloat16,
                                            device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained("/home/zhuyao/Sunpeng/llava_qwen/storage_model", use_fast=True)

tokenizer.eos_token = tokenizer.eos_token if tokenizer.eos_token else "<|endoftext|>"
tokenizer.pad_token = tokenizer.pad_token if tokenizer.pad_token else tokenizer.eos_token



model_llama.resize_token_embeddings(len(tokenizer))



from model.configuration_qwen_llama import Qwen2VLVisionConfig, LlamaConfig
import json

with open("/home/zhuyao/Sunpeng/llava_qwen/model/init_config.json", "r") as f:
    model_config_file = json.load(f)
model_config = LlamaConfig(**model_config_file)



from model.modeling_qwen_llama import LlamaForCausalLM
model = LlamaForCausalLM(model_config)


model.model = model_llama
model.visual = model_qwen.visual


model.to(device = "cpu")
from model.modeling_qwen_llama import LlamaForCausalLM
model.save_pretrained("/home/zhuyao/Sunpeng/llava_qwen/tes")



from safetensors.torch import safe_open, save_file


input_file = "/home/zhuyao/Sunpeng/llava_qwen/tes/model.safetensors"
output_file = "/home/zhuyao/Sunpeng/llava_qwen/tes/model.safetensors"
# .replace('visual.model.', 'visual.').replace('visual.model.', 'visual.').replace('visual.visual.', 'visual.')
# 加载文件/home/zhuyao/Sunpeng/llava_qwen/tes
data = {}
metadata = None
with safe_open(input_file, framework="pt", device="cpu") as f:
    metadata = f.metadata() 
    for key in f.keys():
        print(key)
        modified_key = key.replace('model.model.', 'model.').replace('visual.model.', 'visual.').replace('visual.model.', 'visual.').replace('visual.visual.', 'visual.')
        print(modified_key)
        data[modified_key] = f.get_tensor(key)
    data['lm_head.weight'] = data['model.embed_tokens.weight'].clone()
save_file(data, output_file, metadata=metadata)
