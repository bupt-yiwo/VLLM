from model_processor import LQ_Tokenizer






min_image_tokens = 2
max_image_tokens = 10
processor_path = "/home/zhuyao/Sunpeng/models/qwen_2B_instruct"
model_path = "/home/zhuyao/Sunpeng/llava_qwen/storage_model"

lq_tokenizer = LQ_Tokenizer(model_path,processor_path,min_image_tokens,max_image_tokens)

messages = [[
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "/home/zhuyao/Sunpeng/28.png"},
            {"type": "image", "image": "/home/zhuyao/Sunpeng/A8EB_0151_0492_1_11.png"},
            {"type": "text", "text": "What are the common elements in these pictures?"},
        ],
    }
]
            ,
[
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "/home/zhuyao/Sunpeng/5c975639-2028-4347-a4bf-5072cf39b283.png"},
            {"type": "text", "text": "What aa?"},
        ],
    }
]
]
print(lq_tokenizer(messages))
messages = [[
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "/home/zhuyao/Sunpeng/5c975639-2028-4347-a4bf-5072cf39b283.png"},
            {"type": "text", "text": "What aa?"},
        ],
    }
]]

print(lq_tokenizer(messages))

# tokenizer.eos_token = tokenizer.eos_token if tokenizer.eos_token else "<|endoftext|>"


# special_tokens_dict = {'pad_token': '<|pad|>',
#                        'additional_special_tokens':["<image>","<image_start>","<image_end>"]}

# tokenizer.add_special_tokens(special_tokens_dict)

# tokenizer.save_pretrained("/home/zhuyao/Sunpeng/llava_qwen/storage_model")