import open_clip
import torch
import os
import random
import numpy as np
import argparse
from inference_tool import (zeroshot_evaluation,
                            retrieval_evaluation,
                            semantic_localization_evaluation,
                            get_preprocess
                            )


def random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def build_model(model_name, ckpt_path, device):
    if model_name == "ViT-B-32":
        model, _, _ = open_clip.create_model_and_transforms("ViT-B/32", pretrained="openai")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        msg = model.load_state_dict(checkpoint, strict=False)

    elif model_name == "ViT-H-14":
        model, _, _ = open_clip.create_model_and_transforms("ViT-H/14", pretrained="laion2b_s32b_b79k")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        msg = model.load_state_dict(checkpoint, strict=False)

    print(msg)
    model = model.to(device)
    print("loaded RSCLIP")

    preprocess_val = get_preprocess(
        image_resolution=224,
    )

    return model, preprocess_val


def evaluate(model, preprocess, args):
    print("making val dataset with transformation: ")
    print(preprocess)
    zeroshot_datasets = [
        'EuroSAT',
        'RESISC45',
        'AID'
    ]
    selo_datasets = [
        'AIR-SLT'
    ]

    model.eval()
    all_metrics = {}

    # zeroshot classification
    metrics = {}
    for zeroshot_dataset in zeroshot_datasets:
        zeroshot_metrics = zeroshot_evaluation(model, zeroshot_dataset, preprocess, args)
        metrics.update(zeroshot_metrics)
        all_metrics.update(zeroshot_metrics)
    print(all_metrics)

    # RSITMD
    metrics = {}
    retrieval_metrics_rsitmd = retrieval_evaluation(model, preprocess, args, recall_k_list=[1, 5, 10],
                                                    dataset_name="rsitmd")
    metrics.update(retrieval_metrics_rsitmd)
    all_metrics.update(retrieval_metrics_rsitmd)
    print(all_metrics)

    # RSICD
    metrics = {}
    retrieval_metrics_rsicd = retrieval_evaluation(model, preprocess, args, recall_k_list=[1, 5, 10],
                                                   dataset_name="rsicd")
    metrics.update(retrieval_metrics_rsicd)
    all_metrics.update(retrieval_metrics_rsicd)
    print(all_metrics)

    # selo_datasets
    # Semantic Localization
    metrics = {}
    for selo_dataset in selo_datasets:
        selo_metrics = semantic_localization_evaluation(model, selo_dataset, preprocess, args)
        metrics.update(selo_metrics)
        all_metrics.update(selo_metrics)
    print(all_metrics)

    return all_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name", default="ViT-B-32", type=str,
        help="ViT-B-32 or ViT-H-14",
    )
    parser.add_argument(
        "--ckpt-path", default="/home/zilun/RS5M_v5/ckpt/RS5M_ViT-B-32.pt", type=str,
        help="Path to RS5M_ViT-B-32.pt",
    )
    parser.add_argument(
        "--random-seed", default=3407, type=int,
        help="random seed",
    )
    parser.add_argument(
        "--test-dataset-dir", default="/home/zilun/RS5M_v5/data/rs5m_test_data", type=str,
        help="test dataset dir",
    )
    parser.add_argument(
        "--batch-size", default=500, type=int,
        help="batch size",
    )
    parser.add_argument(
        "--workers", default=8, type=int,
        help="number of workers",
    )
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(args)
    # random_seed(args.random_seed)

    model, img_preprocess = build_model(args.model_name, args.ckpt_path, args.device)

    eval_result = evaluate(model, img_preprocess, args)

    for key, value in eval_result.items():
        print("{}: {}".format(key, value))


if __name__ == "__main__":
    main()
