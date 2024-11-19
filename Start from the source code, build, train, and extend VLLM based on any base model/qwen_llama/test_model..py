from model.configuration_qwen_llama import Qwen2VLVisionConfig, LlamaConfig
import json
import os 
from model.modeling_qwen_llama import LlamaForCausalLM

from transformers import logging
# logging.set_verbosity_error()

model = LlamaForCausalLM.from_pretrained("/home/zhuyao/Sunpeng/llava_qwen/tes")

from model_processor import LQ_Tokenizer


min_image_tokens = 2
max_image_tokens = 10
processor_path = "/home/zhuyao/Sunpeng/models/qwen_2B_instruct"
model_path = "/home/zhuyao/Sunpeng/llava_qwen/tes"

lq_tokenizer = LQ_Tokenizer(model_path,processor_path,min_image_tokens,max_image_tokens)

messages = [[
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "/home/zhuyao/Sunpeng/5c975639-2028-4347-a4bf-5072cf39b283.png"},
            {"type": "text", "text": "What is this?"},
        ],
    }
]]

inputs = lq_tokenizer(messages)

import torch
labels = torch.ones_like(inputs["attention_mask"]) 
# inputs["labels"] = labels
# with torch.no_grad():
# c = model(**inputs)
# c["loss"].backward()

# print(c["loss"])
generated_sequence = model.generate(**inputs, max_new_tokens=20, pad_token_id =  lq_tokenizer.tokenizer.pad_token_id)

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_sequence)
]

output_text = lq_tokenizer.tokenizer.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print("output:")
print(output_text[0])

# model.save_pretrained("/home/zhuyao/Sunpeng/llava_qwen/tes")