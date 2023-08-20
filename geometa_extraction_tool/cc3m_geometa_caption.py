import pandas as pd
import os
import argparse
from tqdm import tqdm
import json
import pickle as pkl
import numpy as np
import random


def select_tags(tags, probs):
    # Get indices of tags with prob >= 0.9
    high_prob_indices = np.where(probs >= 0.9)[0]

    # If there are more than 3 tags with prob >= 0.9
    if len(high_prob_indices) >= 3:
        top_three_indices = high_prob_indices[:3]
        return tags[top_three_indices].tolist()

    # If there are less than 3 tags with prob >= 0.9
    elif len(high_prob_indices) > 0:
        return tags[high_prob_indices].tolist()

    # If no tags with prob >= 0.9
    else:
        return [tags[np.argmax(probs)]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pub11_label_pkl', type=str,
                        default="/Volumes/Tipro7000/RS5M_v3/pub11/RS5M_pub11_label.pkl",
                        help='path of pub11 pkl')
    parser.add_argument('--cc3m_tsv_path', type=str,
                        default="./cc3m/Image_Labels_Subset_Train_GCC-Labels-training.tsv",
                        help='path of cc3m tsv path')
    parser.add_argument('--geometa_template_path', type=str,
                        default="./geometa_template.json",
                        help='path of geometa_template json')
    args = parser.parse_args()

    geometa_template = json.load(open(args.geometa_template_path, "r"))
    geometa_template_cc3m = geometa_template["cc3m"]
    cc3m_info = pd.read_csv(args.cc3m_tsv_path, sep="\t", names=["caption", "url", "tag", "hash", "prob"])
    pub11_label = pkl.load(open(args.pub11_label_pkl, "rb"))
    pub11_label["meta_caption"] = None
    pub11_label["country"] = None
    pub11_label["month"] = None
    cc3m_subset_df = pub11_label[pub11_label["img_name"].str.contains("cc3m")]
    all_cc3m_rs_list = cc3m_subset_df["url"].tolist()
    cc3m_info_rs = cc3m_info[cc3m_info["url"].isin(all_cc3m_rs_list)]

    for df_row in tqdm(cc3m_subset_df.iterrows()):
        url = df_row[1]["url"]
        cc3m_rs_list = cc3m_info_rs["url"].tolist()
        if "cc3m" in df_row[1]["img_name"]:
            if url in cc3m_rs_list:
                tags = cc3m_info_rs[cc3m_info_rs["url"] == url]
                probs = cc3m_info_rs[cc3m_info_rs["url"] == url]
                try:
                    tags = np.array(tags["tag"].values[0].split(","))
                except:
                    continue
                probs = np.array(probs["prob"].values[0].split(","), dtype=np.float32)
                selected_tags = select_tags(tags, probs)
                random.shuffle(geometa_template_cc3m)
                joint_tags = ", ".join(selected_tags) if len(selected_tags) > 1 else selected_tags[0]
                geometa_cap = geometa_template_cc3m[0].replace("{class_label}", joint_tags)
                df_row[1]["meta_caption"] = geometa_cap
    cc3m_subset_df.to_csv("cc3m_geometa.csv", index=False)


if __name__ == "__main__":
    main()