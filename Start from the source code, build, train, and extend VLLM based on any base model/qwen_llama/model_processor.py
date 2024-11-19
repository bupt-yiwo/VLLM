import torch.nn as nn
from qwen_vl_utils import process_vision_info
from transformers import AutoTokenizer, AutoTokenizer, AutoProcessor
def get_image_input(messages,processor):
    text = ""
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    if "pixel_values" in inputs and inputs["pixel_values"] is not None:
        return {"pixel_values":inputs["pixel_values"], 
                "grid_thw":inputs["image_grid_thw"], 
                "image_token_num": ((inputs["image_grid_thw"][:,-2]*inputs["image_grid_thw"][:,-1]) // 4).tolist()}
    else:
        return {}

def expand_images_in_multiple_messages(messages, multipliers):
    expanded_messages = []
    index = 0
    for message in messages:
        if "<image>" not in message:
            expanded_messages.append(message)  
            continue

        result = []
        segments = message.split("<image>")
        for i, segment in enumerate(segments):
            if i < len(segments) - 1:  
                expanded_images = "<image_start>" + ("<image>" * multipliers[index]) + "<image_end>"
                index+=1
                result.append(segment + expanded_images)
            else:
                result.append(segment)  
        expanded_messages.append(''.join(result))
    return expanded_messages

class LQ_Tokenizer(nn.Module):
    def __init__(
        self,
        model_path,
        processor_path,
        min_image_tokens,
        max_image_tokens
    ):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(processor_path, min_pixels=min_image_tokens*28*28, max_pixels=max_image_tokens*28*28)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    
    def forward(
        self,
        messages
    ):
        image_inputs = get_image_input(messages, self.processor)
        for message in messages:
            for turn in message:
                turn["content"] = "".join(
                    dp["text"] if dp["type"] == "text" else "<image>\n"
                    for dp in turn["content"]
                )
        prompts_raw = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
        if "image_token_num" in image_inputs and image_inputs["image_token_num"] is not None:
            prompts = expand_images_in_multiple_messages(prompts_raw, image_inputs["image_token_num"])
            image_inputs.pop("image_token_num")
        else:
            prompts = prompts_raw
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        return {**image_inputs,
                **inputs}




