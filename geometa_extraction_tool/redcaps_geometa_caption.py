import pandas as pd
import os
import argparse
from tqdm import tqdm
import json
import pickle as pkl
import numpy as np
import random
from helper import convert_utc_timestamp


def build_redcaps_url_utc_dict(redcap_dir):
    json_dir = "{}/annotations".format(redcap_dir)
    json_file_list = [filename for filename in os.listdir(json_dir) if filename.endswith(".json")]

    url_utc_dict = dict()
    for json_file in tqdm(json_file_list):
        content = json.load(open(os.path.join(json_dir, json_file), "r"))
        for annotation in content["annotations"]:
            url, created_utc = annotation["url"], annotation["created_utc"]
            if url not in url_utc_dict:
                url_utc_dict[url] = created_utc
    return url_utc_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pub11_label_pkl', type=str,
                        default="/Volumes/Tipro7000/RS5M_v3/pub11/RS5M_pub11_label.pkl",
                        help='path of pub11 pkl')
    parser.add_argument('--redcaps_dir', type=str,
                        default="./redcaps",
                        help='redcaps dir')
    parser.add_argument('--geometa_template_path', type=str,
                        default="./geometa_template.json",
                        help='path of geometa_template json')
    args = parser.parse_args()

    geometa_template = json.load(open(args.geometa_template_path, "r"))
    geometa_template_redcaps = geometa_template["redcaps"]
    pub11_label = pkl.load(open(args.pub11_label_pkl, "rb"))
    pub11_label["meta_caption"] = None
    pub11_label["country"] = None
    pub11_label["month"] = None

    redcaps_subset_df = pub11_label[pub11_label["img_name"].str.contains("redcaps")]
    redcaps_url_utc_dict = build_redcaps_url_utc_dict(args.redcaps_dir)

    for df_row in tqdm(redcaps_subset_df.iterrows()):
        url = df_row[1]["url"]
        utc = redcaps_url_utc_dict[url]
        season, day, month, year = convert_utc_timestamp(utc)
        timestamp = "{} {}, {}".format(month, day, year)
        random.shuffle(geometa_template_redcaps)
        geometa_cap = geometa_template_redcaps[0].replace("{season}", season).replace("{timestamp}", timestamp)
        df_row[1]["meta_caption"] = geometa_cap

    redcaps_subset_df.to_csv("redcaps_geometa.csv", index=False)


if __name__ == "__main__":
    main()