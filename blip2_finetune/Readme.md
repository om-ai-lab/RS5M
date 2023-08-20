# Inference with Tuned BLIP2

## Data

1. Download from: https://huggingface.co/datasets/Zilun/blip2_lora_dataset

2. Change name from xxx_test to xxx/test

   i.e. /home/zilun/RS5M_v4/blip2_ft/data/RSITMD_test ---> /home/zilun/RS5M_v4/blip2_ft/data/RSITMD/test

## How to Use

```
# Try BLIP2 without LoRA
python blip2_peft_inference.py --inference_dataset_name "RSITMD" --inference_dataset_dir /home/zilun/RS5M_v4/blip2_ft/data/RSITMD

# Try BLIP2 with LoRA
python blip2_peft_inference.py --inference_dataset_name "RSITMD" --inference_dataset_dir /home/zilun/RS5M_v4/blip2_ft/data/RSITMD --use_lora_weight --blip2_lora_weight_dir ./blip2_lora_ckpt/BLIP2-RSITMD-Lora-4-20_5e-05_0.01_30-16
```
