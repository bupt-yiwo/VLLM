from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from peft import PeftModel, PeftConfig
import json

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/home/zhuyao/Sunpeng/models/qwen_2B_instruct", torch_dtype="auto", device_map="cuda"
)


# peft_config = PeftConfig.from_pretrained("/home/zhuyao/Sunpeng/models/qwen_2B_instruct_3")
# model = PeftModel.from_pretrained(model, "/home/zhuyao/Sunpeng/models/qwen_2B_instruct_3")

# model = model.merge_and_unload()
# model.save_pretrained("/home/zhuyao/Sunpeng/A100_QWEN2VL/result/downstream/tes", safe_serialization=True)

min_pixels = 336*28*28
max_pixels = 336*28*28
processor = AutoProcessor.from_pretrained("/home/zhuyao/Sunpeng/models/qwen_2B_instruct", min_pixels=min_pixels, max_pixels=max_pixels)
# processor.chat_template = processor.chat_template.replace("You are a helpful assistant.","You are a cervical cancer specialist, focusing on interpreting MRI images and answering related medical questions.")

with open("/home/zhuyao/Sunpeng/finetune_qwen2vl/data/pokemen_eval.json", 'r') as f:
    eval_data = json.load(f)
    
all_answers = []
eval_batch_size = 5

eval_steps = len(eval_data) // eval_batch_size
for i in range(eval_steps):
    batch = eval_data[eval_batch_size*i:eval_batch_size*i+eval_batch_size]
    messages = [[dp["messages"][0]] for dp in batch]
    # Preparation for inference 
    texts = [
    processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    for msg in messages
    ]

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=500)
        generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text_0 = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    for turn in range(eval_batch_size):
        messages[turn].append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": output_text_0
                    }
                ]
            })
        messages[turn].append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Who is his owner?"
                    }
                ]
            })
    texts = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text= texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=500)
        generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text_1 = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    for turn in range(eval_batch_size):
        all_answers.append({
            "label_0": batch[turn]["messages"][1]["content"][-1]["text"],
            "answer_0": output_text_0[turn],
            "label_1": batch[turn]["messages"][3]["content"][-1]["text"],
            "answer_1": output_text_1[turn]
        })

with open("/home/zhuyao/Sunpeng/finetune_qwen2vl/result/eval_result.json", 'w') as f:
    json.dump(all_answers,f,indent=4)

