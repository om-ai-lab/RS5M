from PIL import Image
import clip
import webdataset as wds
from torch.utils.data import DataLoader, Dataset
import io
import random
import argparse
import pandas as pd
from tqdm import tqdm
import os
import time
import torch
from torchvision.utils import save_image
from torchvision.transforms import ToTensor, Compose


Image.MAX_IMAGE_PIXELS = None


def build_ordinary_dataset_dataloader(pub11_intermediate_path, pub11_img_dir, preproc, batch_size, num_worker):
    class RS5MDataset(Dataset):
        def __init__(self, img_dir, intermediate_path, transform):
            self.metainfo = pd.read_csv(intermediate_path)
            self.img_dir = img_dir
            self.transform = transform

        def __len__(self):
            return len(self.metainfo)

        def __getitem__(self, idx):
            img_name = self.metainfo.iloc[idx, 0]
            img_path = os.path.join(self.img_dir, img_name)
            image = Image.open(img_path)
            image = self.transform(image)
            caption = self.metainfo.iloc[idx, 1]
            return img_name, image, caption

    val_dataset = RS5MDataset(pub11_img_dir, pub11_intermediate_path, preproc)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_worker,
        pin_memory=True,
    )

    return val_dataloader


def get_wds_loader(train_dir, val_dir, num_workers, batch_size, num_shuffle, preproc):
    def byte_decode(x):
        return x.decode("utf-8")

    train_url = os.path.join(train_dir, "{pub11,rs3}-train-{0000..0031}.tar")
    val_url = os.path.join(val_dir, "{pub11,rs3}-val-{0000..0031}.tar")

    # train_url = os.path.join(train_dir, "pub11-train-{0000..0031}.tar")
    # val_url = os.path.join(val_dir, "pub11-val-{0000..0031}.tar")

    # train_url = os.path.join(train_dir, "rs3-train-{0000..0031}.tar")
    # val_url = os.path.join(val_dir, "rs3-val-{0000..0031}.tar")

    def my_decoder(key, value):
        if key.endswith(".img_content"):
            assert isinstance(value, bytes)
            value = Image.open(io.BytesIO(value))
            value = preproc(value)
        elif key.endswith(".img_name") or key.endswith(".caption"):
            value = byte_decode(value)
        return value

    train_dataset = wds.WebDataset(train_url).shuffle(num_shuffle).decode(my_decoder)
    train_dataloader = DataLoader(train_dataset, num_workers=num_workers, batch_size=batch_size)

    val_dataset = wds.WebDataset(val_url).shuffle(num_shuffle).decode(my_decoder)
    val_dataloader = DataLoader(val_dataset, num_workers=num_workers, batch_size=batch_size)

    return train_dataloader, val_dataloader


def get_rs3_loader(val_dir, num_workers, batch_size, num_shuffle, preproc):
    def byte_decode(x):
        return x.decode("utf-8")

    val_url = os.path.join(val_dir, "rs3-val-{0000..0031}.tar")

    def my_decoder(key, value):
        if key.endswith(".img_content"):
            assert isinstance(value, bytes)
            value = Image.open(io.BytesIO(value))
            value = preproc(value)
        elif key.endswith(".img_name") or key.endswith(".caption"):
            value = byte_decode(value)
        return value

    val_dataset = wds.WebDataset(val_url).shuffle(num_shuffle).decode(my_decoder)
    val_dataloader = DataLoader(val_dataset, num_workers=num_workers, batch_size=batch_size)

    return val_dataloader


def run_wds_dataloader(train_dataloader, val_dataloader, num_worker, N_stop):
    start = time.perf_counter()
    total_batch_size = 0
    for idx, items in enumerate(train_dataloader):
        img_names, imgs, captions = items["img_name"], items["img_content"], items["caption"]
        batch_size = len(img_names)
        total_batch_size += batch_size

        if total_batch_size > N_stop:
            end = time.perf_counter()
            time_cost = end - start
            print(f"wds dataloader: num_workers={num_worker}, fps={total_batch_size / time_cost:.2f}")
            break


def run_ordinary_dataset(ordinary_dataset_dataloader, num_worker, N_stop):
    start = time.perf_counter()
    total_batch_size = 0
    for index, batch in tqdm(enumerate(ordinary_dataset_dataloader)):
        names, image, caption = batch
        batch_size = len(names)
        total_batch_size += batch_size

        if total_batch_size > N_stop:
            end = time.perf_counter()
            time_cost = end - start
            print(f"ordinary dataloader: num_workers={num_worker}, fps={total_batch_size / time_cost:.2f}")
            break


def dump_from_wds_dataloader(val_dataloader, num_worker, N_stop):
    start = time.perf_counter()
    name_list = []
    caption_list = []
    total_batch_size = 0
    for idx, items in tqdm(enumerate(val_dataloader)):
        img_names, imgs, captions = items["img_name"], items["img_content"], items["caption"]
        batch_size = len(img_names)
        total_batch_size += batch_size
        name_list += img_names
        caption_list += captions
        for (name, img) in zip(img_names, imgs):
            save_path = os.path.join("statics", name)
            save_image(img, save_path)

        if total_batch_size > N_stop:
            end = time.perf_counter()
            time_cost = end - start
            df = pd.DataFrame({
                "name": name_list,
                "caption": caption_list
            })
            df.to_csv("rs3_dump.csv", index=False)
            print(f"wds dataloader: num_workers={num_worker}, fps={total_batch_size / time_cost:.2f}")
            break


def main():
    random.seed(2023)
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str,
                        default="/Volumes/Tipro7000/RS5M_v5/data/train",
                        help='RS5M webdataset train dir')
    parser.add_argument("--val_dir", type=str,
                        default="/Volumes/Tipro7000/RS5M_v5/data/val",
                        help='RS5M webdataset val dir')
    parser.add_argument("--pub11_intermediate_path", type=str,
                        default="/media/zilun/mx500/RS5M/tools/pub11_val_intermediate.csv",
                        help='pub11 intermediate file path')
    parser.add_argument("--pub11_img_dir", type=str,
                        default="/home/zilun/RS5M_processing_v2/pub11_img/img",
                        help='pub11 image dir')
    parser.add_argument("--num_worker", type=int,
                        default=20,
                        help='number of workers')
    parser.add_argument("--batch_size", type=int,
                        default=400,
                        help='batch size')
    parser.add_argument("--num_shuffle", type=int,
                        default=10000,
                        help='number of shuffle (for webdataset)')

    args = parser.parse_args()

    model, preprocess = clip.load("ViT-B/32", device="cpu", jit=False)
    ordinary_dataset_dataloader = build_ordinary_dataset_dataloader(args.pub11_intermediate_path, args.pub11_img_dir,
                                                                    preprocess, args.batch_size, args.num_worker)
    run_ordinary_dataset(ordinary_dataset_dataloader, args.num_worker, N_stop=10000)

    train_dataloader, val_dataloader = get_wds_loader(args.train_dir, args.val_dir, args.num_worker, args.batch_size,
                                                      args.num_shuffle, preprocess)
    run_wds_dataloader(train_dataloader, val_dataloader, args.num_worker, N_stop=10000)

    # # Uncomment if you want to dump data for visualization
    # dump_preprocess = Compose(
    #     [ToTensor()]
    # )
    # rs3_val_dataloader = get_rs3_loader(args.val_dir, 0, 1, args.num_shuffle, dump_preprocess)
    # os.makedirs("statics", exist_ok=True)
    # dump_from_wds_dataloader(rs3_val_dataloader, 0, N_stop=10000)


if __name__ == "__main__":
    main()
