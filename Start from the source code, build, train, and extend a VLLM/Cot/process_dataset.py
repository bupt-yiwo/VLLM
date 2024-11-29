from datasets import Dataset,load_dataset
from PIL import Image
import os
import json
from tqdm import tqdm

class LazyImageDataset(Dataset):
    def __init__(self, dataset_path, image_dir, split='train'):
        self.ds = load_dataset(dataset_path, split=split)
        self.image_dir = image_dir
        self.dataset_path = dataset_path

    def __getitem__(self, idx):
        dp = self.ds[idx]

        image_path = os.path.join(self.image_dir, f"{idx}.png")
        dp["image"].save(image_path)
        
        message = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": dp["question"]}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": dp["output"]}
                    ]
                }
            ],
            "ground_truth": dp["ground_truth"]
        }

        return message

    def __len__(self):
        return len(self.ds)

image_dir = "/home/zhuyao/Sunpeng/llava_qwen/data/cot/images/train"
dataset_path = "/home/zhuyao/Sunpeng/llava_qwen/data/cot"
split = 'train'

lazy_ds = LazyImageDataset(dataset_path, image_dir, split=split)

train_messages = []
for dp in tqdm(range(len(lazy_ds))):
    message = lazy_ds[dp]
    train_messages.append(message)

with open("/home/zhuyao/Sunpeng/llava_qwen/data/cot/train.json", "w") as f:
    json.dump(train_messages, f, indent=4)
