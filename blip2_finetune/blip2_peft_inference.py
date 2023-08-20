import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoProcessor, AutoModelForVision2Seq
import nltk
from nltk.translate.meteor_score import single_meteor_score
import pickle
import numpy as np
from tqdm import tqdm
import argparse
import os
import pdb
import pytorch_lightning as pl


def get_meteor_score(infer, gt):
    gt = gt.lower().strip().split(" ")
    infer = infer.lower().strip().split(" ")

    meteor_scores = single_meteor_score(
        gt, infer
    )
    return meteor_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_dataset_name", type=str, default="RSITMD", help="name of inference dataset")
    parser.add_argument("--inference_dataset_dir", type=str, default="/home/zilun/RS5M_v4/blip2_ft/data/RSITMD", help="save dir for test set")
    parser.add_argument("--use_lora_weight", action="store_true", help="use lora weight or not")
    parser.add_argument("--blip2_lora_weight_dir", type=str, default="./blip2_lora_ckpt/BLIP2-RSITMD-Lora-15-12_5e-05_1e-06_50-64",
                        help="save dir for BLIP2 lora weight (OPT-6.7B)")

    parser.add_argument("--result_dir", type=str, default="./eval_result", help="evaluation result save dir")
    parser.add_argument("--blip2_model_name", type=str, default="Salesforce/blip2-opt-6.7b", help="which blip2 model to use")

    args = parser.parse_args()

    pl.seed_everything(2023)

    print("---------load dataset---------")
    dataset = load_dataset("imagefolder", data_dir=args.inference_dataset_dir, split="test")
    print("---------load model------")
    model = AutoModelForVision2Seq.from_pretrained(
        args.blip2_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(
        args.blip2_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    if args.use_lora_weight:
        print("use lora weights for blip2")
        assert args.blip2_lora_weight_dir is not None
        model = PeftModel.from_pretrained(model, args.blip2_lora_weight_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("---------start test--------------")
    meteor_scores = []
    for index, exmaple in enumerate(dataset):
        image = exmaple["image"]
        inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
        pixel_values = inputs.pixel_values
        # inference
        generated_output = model.generate(
            pixel_values=pixel_values,
            max_length=50,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
        )
        generated_caption = processor.batch_decode(generated_output, skip_special_tokens=True)[0].replace("\n", "")
        # get gt cap
        gt_caption = exmaple["text"].replace("\n", "")
        # calculate score
        if gt_caption is None:
            print("None caption in gt: {}".format(index))

        print(index, generated_caption, "|||", gt_caption)
        meteor_score = get_meteor_score(generated_caption, gt_caption)
        meteor_scores.append(meteor_score)

    meteor_scores = np.array(meteor_scores)
    os.makedirs(args.result_dir, exist_ok=True)

    if args.use_lora_weight:
        print("{} lora mode meteor score:{}".format(np.mean(meteor_scores), args.inference_dataset_name))

        with open("{}/lora_meteor_{}.pkl".format(args.result_dir, args.inference_dataset_name), "wb") as f:
            pickle.dump(meteor_scores, f)
    else:
        print("{} vanilla model meteor score:{}".format(np.mean(meteor_scores), args.inference_dataset_name))

        with open("{}/origin_meteor_{}.pkl".format(args.result_dir, args.inference_dataset_name), "wb") as f:
            pickle.dump(meteor_scores, f)


if __name__ == "__main__":
    main()