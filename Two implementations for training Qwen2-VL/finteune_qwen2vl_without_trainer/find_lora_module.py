from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from peft import PeftModel, PeftConfig


model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/home/zhuyao/Sunpeng/models/qwen_2B_instruct", torch_dtype="auto", device_map="cpu"
)

lora_modules = []
for name, module in model.visual.named_modules():
    if isinstance(module, torch.nn.Linear):
        lora_modules.append(name)

print(f"适用于 LoRA 的模块有:{lora_modules}")

# lora_modules = ['layers.0.self_attn.q_proj', 'layers.0.self_attn.k_proj', 'layers.0.self_attn.v_proj', 
#                 'layers.0.self_attn.o_proj', 'layers.0.mlp.gate_proj', 'layers.0.mlp.up_proj', 
#                 'layers.0.mlp.down_proj', 'layers.1.self_attn.q_proj', 'layers.1.self_attn.k_proj', 
#                 'layers.1.self_attn.v_proj', 'layers.1.self_attn.o_proj', 'layers.1.mlp.gate_proj', 
#                 'layers.1.mlp.up_proj', 'layers.1.mlp.down_proj'] 
# # 查看指定模块的形状
# for module_name in lora_modules:
#     module = dict(model.model.named_modules())[module_name]  # 获取模块对象
#     if hasattr(module, 'weight'):
#         print(f"{module_name} 的权重矩阵形状: {module.weight.shape}")
#     else:
#         print(f"{module_name} 无权重参数")

# 适用于 LoRA 的模块有:['layers.0.self_attn.q_proj', 'layers.0.self_attn.k_proj', 'layers.0.self_attn.v_proj', 'layers.0.self_attn.o_proj', 'layers.0.mlp.gate_proj', 'layers.0.mlp.up_proj', 'layers.0.mlp.down_proj']