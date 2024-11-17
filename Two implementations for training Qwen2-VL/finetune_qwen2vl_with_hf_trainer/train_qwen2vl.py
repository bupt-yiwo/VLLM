import torch
import json
import datetime
import os
import logging
from peft import LoraConfig, get_peft_model
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from torch.utils.data import Dataset
from util.vision_util import process_vision_info
from tqdm import tqdm
from contextlib import redirect_stdout
import sys
import argparse
import torch.distributed as dist


parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', type=str, default="/home/zhuyao123/SunPeng/models/qwen_2B_instruct",required=False, help='Pretrained model directory')
parser.add_argument('--train_data_path', type=str, default="/home/zhuyao123/SunPeng/train_pa_data/qwen_33W.json",required=False, help='Training data file path')
parser.add_argument('--output_dir', type=str, default="/home/zhuyao123/SunPeng/train_pa_data/results", required=False, help='Output directory')
parser.add_argument('--deepspeed_config', type=str, default="/home/zhuyao123/SunPeng/train_pa_data/zero_stage2.json",required=False, help='DeepSpeed config path')
parser.add_argument('--finetune_choice', type=str, choices=['use_lora','full_finetune_visual','full_finetune_merger','full_finetune_llm','all'],required=True, help='how to finetune')
parser.add_argument('--lora_modules', type=str, nargs='+', default=["qkv", "fc1", "fc2", "mlp.0", "mlp.2"], required=False, help='The modules you want to lora finetune')
parser.add_argument('--lora_rank', type=int, default=128,required=False, help='lora_rank')
parser.add_argument('--lora_alpha', type=int, default=256,required=False, help='double_lora_rank')
parser.add_argument('--max_tokens', type=int, default=256,required=False, help='max_tokens(of a photo)')
parser.add_argument('--min_tokens', type=int, default=512,required=False, help='min_tokens(of a photo)')
parser.add_argument('--per_device_train_batch_size', type=int, default=1,required=False, help='per_device_train_batch_size')
parser.add_argument('--num_train_epochs', type=int, default=3,required=False, help='per_device_train_batch_size')
parser.add_argument('--logging_steps', type=int, default=20,required=False, help='logging_steps')
parser.add_argument('--save_steps', type=int, default=200,required=False, help='save_steps')
parser.add_argument('--save_total_limit', type=int, default=100,required=False, help='save_total_limit')
parser.add_argument('--gradient_accumulation_steps', type=int, default=16,required=False, help='gradient_accumulation_steps')
parser.add_argument('--learning_rate', type=float, default=5e-4,required=False, help='learning_rate')
parser.add_argument('--warmup_steps', type=int, default=500,required=False, help='warmup_steps')
parser.add_argument('--lr_scheduler_type', type=str, default="cosine",required=False, help='lr_scheduler_type')
parser.add_argument('--system_message', type=str, default="You are an expert in the medical field, focused on discussions regarding medical knowledge, diagnostics, treatment plans, and related health issues. You should provide advice and information based on the latest scientific research, clinical practice, and authoritative medical guidelines. When answering questions, ensure that the medical information provided is accurate, reliable, and evidence-based. Do not offer unverified treatment advice. Based on patient history, symptoms, and medical data, make reasonable assumptions and recommendations, and when necessary, suggest further tests or that the patient consults a professional doctor. Your goal is to help users understand medical knowledge and support them in making informed health decisions.", required=False, help='system_message')
parser.add_argument('--eval_data_path', type=str, default="/home/zhuyao123/SunPeng/train_pa_data/qwen_33W.json",required=False, help='Eval data file path')
parser.add_argument('--per_device_eval_batch_size', type=int, default=1,required=False, help='per_device_eval_batch_size')
parser.add_argument('--evaluation_strategy', type=str, default="step",required=False, help='evaluation_strategy')
parser.add_argument('--eval_steps', type=int, default=50,required=False, help='eval_steps')

args = parser.parse_args()


# output directory
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)


# logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
f_handler = logging.FileHandler(os.path.join(output_dir, 'training.log'), mode='a')
f_handler.setFormatter(formatter)


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

c_handler = TqdmLoggingHandler()
c_handler.setFormatter(formatter)

logger.addHandler(f_handler)
logger.addHandler(c_handler)

transformers_logger = logging.getLogger('transformers')
transformers_logger.setLevel(logging.INFO)
transformers_logger.handlers = logger.handlers


# Dataset
class QwenDataSet(Dataset):  
    def __init__(self, data_path):
        super().__init__()
        with open(data_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def find_assistant_content_sublist_indexes(l):
    start_token_ids = [151644, 77091]
    end_token_id = 151645

    start_indexes = []
    end_indexes = []

    i = 0
    while i < len(l) - 1:
        if l[i:i+2] == start_token_ids:
            start_indexes.append(i+1)
            j = i + 2
            while j < len(l):
                if l[j] == end_token_id:
                    end_indexes.append(j)
                    break
                j += 1
            i = j
        else:
            i += 1

    return list(zip(start_indexes, end_indexes))


class DataCollatorForQwen2VL:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        messages = [m['messages'] for m in batch]
        texts = [
            self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=False
            )
            for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        input_ids_lists = inputs['input_ids'].tolist()
        assert len(messages) == len(input_ids_lists)

        labels_list = []
        for ids_list in input_ids_lists:
            label_ids = [-100] * len(ids_list)
            for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
                # Exclude the special tokens for assistant start
                label_ids[
                    begin_end_indexs[0] + 2 : begin_end_indexs[1] + 1
                ] = ids_list[begin_end_indexs[0] + 2 : begin_end_indexs[1] + 1]
            labels_list.append(label_ids)

        labels_ids = torch.tensor(labels_list, dtype=torch.int64)
        inputs['labels'] = labels_ids
        return inputs

# Save chat_template
def write_chat_template(processor, output_dir):
    output_chat_template_file = os.path.join(output_dir, "chat_template.json")
    chat_template_json_string = (
        json.dumps({"chat_template": processor.chat_template}, indent=2, sort_keys=True)
        + "\n"
    )
    with open(output_chat_template_file, "w", encoding="utf-8") as writer:
        writer.write(chat_template_json_string)
        logger.info(f"Chat template saved in {output_chat_template_file}")

# Train
def train():

    # Set Model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
    )
    for name, param in model.named_parameters():
        param.requires_grad = True  
    if args.finetune_choice == "use_lora":
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_modules,
            lora_dropout=0.05,
            bias="none",
        )
        print(args.lora_rank,args.lora_alpha,args.lora_modules)
        model = get_peft_model(model, lora_config)
        for name, param in model.model.named_parameters():
            if 'lora' not in name:
                param.requires_grad = False
                
    elif(args.finetune_choice == "full_finetune_visual"):
        for name, param in model.model.named_parameters():
            param.requires_grad = False
        for name, param in model.visual.merger.named_parameters():
            param.requires_grad = False
        
    elif(args.finetune_choice == "full_finetune_merger"):
        for name, param in model.visual.named_parameters():
            param.requires_grad = False
        for name, param in model.model.named_parameters():
            param.requires_grad = False
        for name, param in model.visual.merger.named_parameters():
            param.requires_grad = True

    elif(args.finetune_choice == "full_finetune_llm"):        
        for name, param in model.visual.named_parameters():
            param.requires_grad = False      

    elif(args.finetune_choice == "all"):
        print("you are finetuning the full model")
        with open("output.txt", "w") as file:
            file.write("full_finetune_all")
    else:
        raise ValueError("you are finetuning the full model,but the args(finetune_choice) you give is wrong!")

    processor = AutoProcessor.from_pretrained(
        args.model_dir,
        min_pixels=args.min_tokens * 28 * 28,
        max_pixels=args.max_tokens * 28 * 28,
    )
    processor.chat_template = processor.chat_template.replace("You are a helpful assistant.",args.system_message)
    
    # Set Data
    train_dataset = QwenDataSet(args.train_data_path)
    eval_dataset = QwenDataSet(args.eval_data_path)
    data_collator = DataCollatorForQwen2VL(processor)
    
    # Set training_args
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        logging_dir=output_dir,
        logging_steps=args.logging_steps,  
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=0.02,
        lr_scheduler_type=args.lr_scheduler_type,
        bf16=True,
        fp16=False,
        deepspeed= args.deepspeed_config,
        ddp_find_unused_parameters=False,
        log_level='info',   
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps, 
        per_device_eval_batch_size=args.per_device_eval_batch_size, 
    )
    # LogLossCallback
    class LogLossCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.is_world_process_zero:
                if logs is not None:
                    if 'loss' in logs:
                        logger.info(f"Step {state.global_step}: loss = {logs['loss']}")
                    if 'eval_loss' in logs:
                        logger.info(f"Step {state.global_step}: eval_loss = {logs['eval_loss']}")

    # Set Trainer
    class CustomTrainer(Trainer):
        def _save_checkpoint(self, model, trial, metrics=None):
            if self.is_world_process_zero():
                model_save_path = os.path.join(
                    self.args.output_dir, f"step-{self.state.global_step}"
                )
                model.save_pretrained(model_save_path)
                
                optimizer_save_path = os.path.join(model_save_path, "optimizer.pt")
                scheduler_save_path = os.path.join(model_save_path, "scheduler.pt")
                torch.save(self.optimizer.state_dict(), optimizer_save_path)
                if self.lr_scheduler is not None:
                    torch.save(self.lr_scheduler.state_dict(), scheduler_save_path)
                    
                self.state.save_to_json(os.path.join(self.args.output_dir, "trainer_state.json"))
           
             
        if args.finetune_choice == "use_lora":
            def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
                model.eval()  
                eval_dataloader = self.get_eval_dataloader(eval_dataset)

                total_loss = 0
                for batch in tqdm(eval_dataloader, desc="Evaluating", disable=not self.is_world_process_zero()):
                    with torch.no_grad():
                        outputs = self.model(**batch)
                        loss = outputs.loss
                        total_loss += loss.item()
                
                avg_loss = total_loss / len(eval_dataloader)

                avg_loss_tensor = torch.tensor(avg_loss, device=self.args.device)
                dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
                avg_loss_tensor /= dist.get_world_size()

                if self.is_world_process_zero():
                    avg_loss = avg_loss_tensor.item()
                    logger.info(f"Evaluation Loss: {avg_loss}")
                
                return {"eval_loss": avg_loss_tensor.item()}


    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[LogLossCallback()],
    )

    with redirect_stdout(sys.stderr):
        trainer.train()

    if trainer.is_world_process_zero():
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        write_chat_template(processor, output_dir)

if __name__ == "__main__":
    train()
