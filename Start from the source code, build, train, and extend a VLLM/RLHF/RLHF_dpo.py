from datasets import features
from datasets import load_dataset
from model.model_processing import LQ_Tokenizer
from trl import DPOConfig, DPOTrainer
from model.modeling_qwen_llama import LlamaForCausalLM
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default="/home/zhuyao/Sunpeng/models/qwen_2B_instruct",required=False, help='Pretrained model directory')
parser.add_argument('--tokenizer_path', type=str, default="/home/zhuyao/Sunpeng/models/qwen_2B_instruct",required=False, help='Pretrained model directory')
parser.add_argument('--processor_path', type=str, default="/home/zhuyao/Sunpeng/models/qwen_2B_instruct",required=False, help='Pretrained model directory')
parser.add_argument('--output_dir', type=str, default="/home/zhuyao/Sunpeng/finteune_qwen2vl_no_trainer/result",required=False, help='Pretrained model directory')
parser.add_argument('--min_image_tokens', type=int, default=336,required=False, help='min_image_tokens(of a photo)')
parser.add_argument('--max_image_tokens', type=int, default=336,required=False, help='max_image_tokens(of a photo)')
parser.add_argument('--train_data_path', type=str, default="/home/zhuyao/Sunpeng/finetune_qwen2vl/data/pokemen_train.json",required=False, help='Training data file path')
parser.add_argument('--num_train_epochs', type=int, default=3,required=False, help='num_train_epochs')
parser.add_argument('--learning_rate', type=float, default=1e-5,required=False, help='learning_rate')
parser.add_argument('--per_device_train_batch_size', type=int, default=2,required=False, help='per_device_eval_batch_size')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,required=False, help='gradient_accumulation_steps')
parser.add_argument('--log_steps', type=int, default=10,required=False, help='log_steps')
parser.add_argument('--save_steps', type=int, default=10,required=False, help='save_steps')

args = parser.parse_args()

min_image_tokens = args.min_image_tokens
max_image_tokens = args.max_image_tokens
processor_path = args.processor_path
tokenizer_path = args.tokenizer_path
model_path = args.model_path

model = LlamaForCausalLM.from_pretrained(model_path,torch_dtype="auto")
ref_model = LlamaForCausalLM.from_pretrained(model_path,torch_dtype="auto")
processor = LQ_Tokenizer(tokenizer_path,processor_path,min_image_tokens,max_image_tokens)

dataset = load_dataset(args.train_data_path, split="train") # [:5%]
def format(example):
    prompt = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": example["question"]}],
        },
    ]
    chosen = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example["chosen"]}],
        },
    ]
    rejected = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example["rejected"]}],
        },
    ]
    # Apply the chat template
    prompt = processor.processor.apply_chat_template(prompt, tokenize=False).replace("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n","")
    chosen = processor.processor.apply_chat_template(chosen, tokenize=False).replace("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n","")
    rejected = processor.processor.apply_chat_template(rejected, tokenize=False).replace("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n","")

    return {"images": [example["image"]], "prompt": prompt, "chosen": chosen, "rejected": rejected}


dataset = dataset.map(lambda dp: {"question": json.loads(dp["text"])["question"]})
dataset = dataset.map(lambda dp: {"chosen": json.loads(dp["text"])["chosen"]})
dataset = dataset.map(lambda dp: {"rejected": json.loads(dp["text"])["rejected"]})
dataset = dataset.map(format, remove_columns=dataset.column_names)
f = dataset.features
f["images"] = features.Sequence(features.Image(decode=True)) 
dataset = dataset.cast(f)


training_args = DPOConfig(
    output_dir=args.output_dir,
    bf16=True,
    gradient_checkpointing=False,
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    num_train_epochs=args.num_train_epochs,
    dataset_num_proc=56, 
    dataloader_num_workers=56,
    logging_steps=args.log_steps,
    save_steps=args.save_steps,
    max_grad_norm=1.0
)


trainer = DPOTrainer(
    model,
    ref_model=ref_model, 
    args=training_args,
    train_dataset=dataset,
    tokenizer=processor
)

trainer.train()