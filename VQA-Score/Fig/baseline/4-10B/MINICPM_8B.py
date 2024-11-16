import json
import string
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
from mantis.models.mllava import chat_mllava, MLlavaProcessor, LlavaForConditionalGeneration

# 加载模型和processor
processor = MLlavaProcessor.from_pretrained("/home/zhuyao/Sunpeng/MLLMs/4-10B/MINICPM_V_2.6_8B")
model = LlavaForConditionalGeneration.from_pretrained(
    "/home/zhuyao/Sunpeng/MLLMs/4-10B/MINICPM_V_2.6_8B",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation=None  # 或者使用 "flash_attention_2"
)

generation_kwargs = {
    "max_new_tokens": 5,
    "num_beams": 1,
    "do_sample": False
}

# 加载数据
with open("/home/zhuyao/Sunpeng/wxx_data_eval/data/11.5/coco_11.5_A100.json", 'r') as f:
    data = json.load(f)

coco_results = []
logits_indices = [9642, 2822, 9828, 2305]  # Yes 和 No 对应的 logits 索引

# 处理数据
for dp in tqdm(data, desc="Processing"):
    torch.cuda.empty_cache()
    text = "<image>\n" + dp['conversations'][0]['value'].strip('\n<image>').strip()
    image = Image.open(dp["image"]).convert("RGB")  # 转换为 RGB 模式
    images = [image]

    # 无需计算梯度，节省显存
    with torch.no_grad():
        response, history, logitsf1 = chat_mllava(text, images, model, processor, **generation_kwargs)
        logits_yn = [logitsf1[idx] for idx in logits_indices]

        # Softmax 计算
        logits_yn_softmax = F.softmax(torch.tensor(logitsf1), dim=0)[logits_indices].tolist()
        prob_yn_softmax = F.softmax(torch.tensor(logits_yn), dim=0).tolist()

        # 打印调试信息
        print("logits_yn:", logits_yn)
        print("logits_yn_softmax:", logits_yn_softmax)
        print("prob_yn_softmax:", prob_yn_softmax)
        print("USER:", text)
        print("ASSISTANT:", response)

        # 结果保存
        re = {
            "id": dp['id'],
            "logits_yn_softmax": logits_yn_softmax,
            "prob_yn_softmax": prob_yn_softmax,
            "answer": response,
            "expected_answer": dp['conversations'][1]['value'].strip().lower(),
            "ac": int(response.split()[0].strip(string.punctuation).lower() == dp['conversations'][1]['value'].strip().lower())
        }
        coco_results.append(re)

# 保存结果
with open("/home/zhuyao/Sunpeng/wxx_data_eval/data/11.5/results/MINICPM_8B.json", 'w') as f:
    json.dump(coco_results, f, indent=4)
