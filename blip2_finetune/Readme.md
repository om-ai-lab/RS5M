# Inference Dataset

# How to Use

python blip2_peft_inference.py --inference_dataset_name "RSITMD" --inference_dataset_dir /home/zilun/RS5M_v4/blip2_ft/data/RSITMD

python blip2_peft_inference.py --inference_dataset_name "RSITMD" --inference_dataset_dir /home/zilun/RS5M_v4/blip2_ft/data/RSITMD --use_lora_weight --blip2_lora_weight_dir ./blip2_lora_ckpt/BLIP2-RSITMD-Lora-4-20_5e-05_0.01_30-16
