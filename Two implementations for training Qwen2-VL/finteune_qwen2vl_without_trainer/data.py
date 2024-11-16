from datasets import load_dataset
from PIL import Image
import json
from tqdm import tqdm
import random
"""
export HF_ENDPOINT="https://hf-mirror.com"
echo 'export HF_ENDPOINT="https://hf-mirror.com"' >> ~/.bashrc
export TRANSFORMERS_CACHE="/home/zhuyao/Sunpeng/finetune_qwen2vl/data"
"""
dataset = load_dataset("xjs521/pokemen_small", cache_dir="/home/zhuyao/Sunpeng/finetune_qwen2vl/data")
messages =  []

for turn in tqdm(range(len(dataset["train"])),desc= "datas"):
    image_path = f"/home/zhuyao/Sunpeng/finetune_qwen2vl/data/pokemen/{turn}.png"
    dataset["train"][turn]["image"].save(image_path)
    message = { "id": turn,
                "messages":[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": image_path,
                                },
                                {"type": "text", "text": "What does this image depict?" },
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": dataset["train"][turn]["en_text"]},
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Who is his owner?" },
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "Pokémon Master"},
                            ],
                        }
                    ]
              }
    messages.append(message)
    
train_dataset = random.sample(messages,int(0.8*len(messages)))
eval_dataset = [dp for dp in messages if dp not in train_dataset]

with open("/home/zhuyao/Sunpeng/finetune_qwen2vl/data/pokemen_train.json", 'w') as f:
    json.dump(train_dataset,f,indent=4)
    
with open("/home/zhuyao/Sunpeng/finetune_qwen2vl/data/pokemen_eval.json", 'w') as f:
    json.dump(eval_dataset,f,indent=4)
