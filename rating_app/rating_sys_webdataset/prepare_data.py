import yaml
import os
from PIL import Image
import csv
import torch
from torch.utils.data import IterableDataset
from torchvision import transforms
import webdataset as wds
from torch.utils.data import DataLoader, IterableDataset, ChainDataset
import io
import numpy as np
from tqdm import tqdm
import random
from PIL import ImageFilter

Image.MAX_IMAGE_PIXELS = None


def decode_img_text(k, v):
    if "caption" in k:
        value = v.decode('ascii', 'replace')
        return value
    elif "image" in k:
        value = Image.open(io.BytesIO(v)).convert("RGB")
        return value
    elif "name" in k:
        value = v.decode('ascii', 'replace')
        return value


def prepare_data(
        dataset_root, num_shuffle
):
    rs3_val_path = os.path.join(dataset_root, "rs3", "RS5M_rs3_val.tar")
    val_dataset_fpath = rs3_val_path
    raw_dataset = wds.WebDataset(val_dataset_fpath)
    dataset = (
        raw_dataset
            .shuffle(num_shuffle, initial=num_shuffle)
            .decode(decode_img_text)
            .to_tuple("input.name", "input.image", "input.caption")
    )
    return dataset


def select_data(val_dataset, sample_num, image_save_path, csv_path):
    header = ["name", "caption"]
    rows = []
    index = 0
    for i, (name, image, text) in enumerate(val_dataset):
        if index >= sample_num:
            break
        print(i, name)
        dst_path = os.path.join(image_save_path, name)
        image.save(dst_path)
        rows.append((name, text))
        index += 1
    with open(csv_path, 'w', encoding='utf8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


if __name__ == "__main__":
    image_save_path = "./static"
    os.makedirs(image_save_path, exist_ok=True)

    sample_num = 10000
    val_dataset = prepare_data(dataset_root="/media/zilun/fanxiang4t/RS5M_data", num_shuffle=200000)

    csv_path = "./rs3_dump.csv"
    select_data(val_dataset, sample_num, image_save_path, csv_path)

