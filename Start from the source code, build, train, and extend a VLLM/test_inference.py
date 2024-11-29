
import json
import os 
from model.modeling_qwen_llama import LlamaForCausalLM
import torch
from tes_RAG import vllm_rage
from model.model_processing import LQ_Tokenizer



model_path = "/home/zhuyao/Sunpeng/llava_qwen/check_point/instruct_525k"
# model_path = "/home/zhuyao/Sunpeng/llava_qwen/check_point/DPO_5.73k/step-150"

model = LlamaForCausalLM.from_pretrained(model_path,device_map = "cuda:7",torch_dtype=torch.float16,attn_implementation="flash_attention_2",)

# from peft import PeftModel
# lora
# lora_model = PeftModel.from_pretrained(model, "/home/zhuyao/Sunpeng/llava_qwen/check_point/downstream_tasks/emo") 



min_image_tokens = 4
max_image_tokens = 336
processor_path = "/home/zhuyao/Sunpeng/models/qwen_2B_instruct"
tokenizer_path = "/home/zhuyao/Sunpeng/llava_qwen/tes"
lq_tokenizer = LQ_Tokenizer(tokenizer_path,processor_path,min_image_tokens,max_image_tokens)
# lq_tokenizer.tokenizer.chat_template = lq_tokenizer.tokenizer.chat_template.replace("You are a helpful assistant.","You are a helpful assistant.")

messages = [[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "/home/zhuyao/Sunpeng/llava_qwen/data/rag/emo/000317dc-9047-4d68-bb55-e40c09ed0f9a.jpg"
                    },
                    {
                        "type": "text",
                        "text": "这个表情包表达了什么？"
                    }
                ]
            }
            ]]
# messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "image",
#                         "image": "/home/zhuyao/Sunpeng/274e3ef7c2bc430877b5e15d2838e1c1.jpg"
#                     },
#                     {
#                         "type": "text",
#                         "text": "Why is the girl in the image beautiful?"
#                     }
#                 ]
#             }
#             ]

inputs = lq_tokenizer(messages)

# test loss
# labels = torch.ones_like(inputs["attention_mask"]) 
# inputs["labels"] = labels
# with torch.no_grad():
# c = model(**inputs)
# c["loss"].backward()
# print(c["loss"])

inputs = {k:v.to(model.device) for k,v in inputs.items()}
generetion_config = {
    "bos_token_id": lq_tokenizer.tokenizer.bos_token_id,
    "do_sample": True,
    "eos_token_id": [128009],
    "repetition_penalty": 1.15,
    "temperature": 1.0,
    "top_p": 0.001,
    "top_k": 5
  }
  
generated_sequence = model.generate(**inputs, max_new_tokens=200, pad_token_id =  lq_tokenizer.tokenizer.pad_token_id,**generetion_config)

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_sequence)
]

output_text = lq_tokenizer.tokenizer.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print("output:")
print(output_text[0])

