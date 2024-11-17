When using, all paths need to be modified to your local paths.

## Download Data

```python
python data.py
```

## Train

When training using the trainer, you can select various fine-tuning methods. If you are using LoRA, you need to first find the LoRA fine-tuning module:

```python
python find_lora_module.py
```

For non-trainer-based training, I have only provided the code for full fine-tuning. However, to change the training module or use LoRA, you can refer to the files that use the trainer—simply copy and paste as needed.  If you want to learn more about LoRA, you can refer to the PEFT section in the project **Code implementation of various Hugging Face tools**.

Then, start the training process:

```python
sh run.sh
```

If you find errors, rewrite the code this way：

```python
#/home/xx/anaconda3/envs/qwen2vl/lib/python3.10/site-packages/transformers/trainer_utils.py
    def __call__(self, features: List[dict]):
        # features = [self._remove_columns(feature) for feature in features]
        return self.data_collator(features)

```

To visualize training progress and curve changes:

```python
visual_train.ipynb
```

## Evaluate

```python
python eval.py
```

This script runs batch inference on the eval set. If you want to evaluate the quality of open-ended text generation, you can refer to the project **Quality Evaluation Metrics for Text Generation**. To evaluate closed-ended answers (e.g., yes/no or multiple-choice questions), you can refer to the **VQA Score** project.

## WebUI

```python
python web_demo_mm.py
```

