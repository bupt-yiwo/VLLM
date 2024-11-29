import torch.nn as nn
from qwen_vl_utils import process_vision_info
from transformers import AutoTokenizer, AutoTokenizer, AutoProcessor
import copy


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
    if isinstance(messages,list):
        for message in messages:
            if "<image>" not in message:
                expanded_messages.append(message)  
                continue

            result = []
            segments = message.split("<image>")
            for i, segment in enumerate(segments):
                if i < len(segments) - 1:  
                    expanded_images = "<image>" * multipliers[index]
                    index+=1
                    result.append(segment + expanded_images)
                else:
                    result.append(segment)  
            expanded_messages.append(''.join(result))
    else:
        if "<image>" not in messages:
            expanded_messages.append(messages)  
        result = []
        segments = messages.split("<image>")
        for i, segment in enumerate(segments):
            if i < len(segments) - 1:  
                expanded_images = "<image>" * multipliers[index]
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
        max_image_tokens,
        padding_side:str ="right",
        **kwargs
    ):

        super().__init__()
        self.processor = AutoProcessor.from_pretrained(processor_path, min_pixels=min_image_tokens*28*28, max_pixels=max_image_tokens*28*28)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True,padding_side = padding_side)

    
    def forward(
        self,
        messages
    ):
        # print(messages)
        image_inputs = get_image_input(messages, self.processor)
    
        messages_copy = copy.deepcopy(messages) # to train
        if isinstance(messages_copy[0], list):
            for message in messages_copy:
                for turn in message:
                    if isinstance(turn["content"], list):
                        turn["content"] = "".join(
                            dp["text"] if dp["type"] == "text" else "<image_start><image><image_end>\n"
                            for dp in turn["content"]
                        )
                    else:
                        pass #system_message
        else:
            for turn in messages_copy:
                if isinstance(turn["content"], list):
                    turn["content"] = "".join(
                        dp["text"] if dp["type"] == "text" else "<image_start><image><image_end>\n"
                        for dp in turn["content"]
                    )
                else:
                        pass #system_message
        prompts_raw = self.tokenizer.apply_chat_template(
                        messages_copy,
                        tokenize=False,
                        add_generation_prompt=True
                    )
        if "image_token_num" in image_inputs and image_inputs["image_token_num"] is not None:
            prompts = expand_images_in_multiple_messages(prompts_raw, image_inputs["image_token_num"])
            image_inputs.pop("image_token_num")
        else:
            prompts = prompts_raw
        inputs = self.tokenizer(prompts, return_tensors="pt", padding="longest", max_length=2048, truncation=True )
        return {**image_inputs,
                **inputs}




