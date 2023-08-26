import pandas as pd
import os
import argparse
from tqdm import tqdm
import json
from helper import countrycode2countryname, fmow_timestamp2symdh, fmow_bbox_location, parse_fmow_df
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rs3_train_csv_path', type=str,
                        default="/media/zilun/wd-16/RS5M_v3/rs3/RS5M_rs3_train.csv",
                        help='path of rs3 train csv')
    parser.add_argument('--rs3_val_csv_path', type=str,
                        default="/media/zilun/wd-16/RS5M_v3/rs3/RS5M_rs3_val.csv",
                        help='path of rs3 val csv')
    parser.add_argument('--fmow_gt_dir', type=str,
                        default="/media/zilun/wd-16/datasets/fmow/groundtruth",
                        help='dir of fmow gt')
    parser.add_argument('--geometa_template_path', type=str,
                        default="./geometa_template.json",
                        help='path of geometa_template json')
    args = parser.parse_args()

    geometa_template = json.load(open(args.geometa_template_path, "r"))
    geometa_template_fmow = geometa_template["fmow_complex"]

    rs3_train_csv = pd.read_csv(args.rs3_train_csv_path)
    fmow_train = rs3_train_csv[rs3_train_csv["subset_name"] == "fmow"]
    img_name_train_list = []
    geometa_cap_train_list = []
    country_train_list = []
    month_train_list = []
    latitude_train_list = []
    longitude_train_list = []
    utm_train_list = []
    for df_row_train in tqdm(fmow_train.iterrows()):
        img_name_train = df_row_train[1]["img_name"]
        gt_json_path_train = os.path.join(args.fmow_gt_dir, img_name_train[:-3] + "json")
        gt_json_train = json.load(open(gt_json_path_train, "r"))
        random.shuffle(geometa_template_fmow)
        geometa_cap_train, country_train, month_train, coord_train, utm_train = parse_fmow_df(gt_json_train, geometa_template_fmow[0])
        latitude_train, longitude_train = coord_train
        latitude_train_list.append(latitude_train)
        longitude_train_list.append(longitude_train)
        utm_train_list.append(utm_train)
        country_train_list.append(country_train)
        month_train_list.append(month_train)
        # print(img_name_train, geometa_cap)
        img_name_train_list.append(img_name_train)
        geometa_cap_train_list.append(geometa_cap_train)
    geometa_df_train = pd.DataFrame({
        "img_name": img_name_train_list,
        "geometa_cap": geometa_cap_train_list,
        "country": country_train_list,
        "month": month_train_list,
        "latitude": latitude_train_list,
        "longitude": longitude_train_list,
        "utm": utm_train_list
    })
    geometa_df_train.to_csv("fmow_geometa_train_coord.csv", index=False)

    rs3_val_csv = pd.read_csv(args.rs3_val_csv_path)
    fmow_val = rs3_val_csv[rs3_val_csv["subset_name"] == "fmow"]
    img_name_val_list = []
    geometa_cap_val_list = []
    country_val_list = []
    month_val_list = []
    latitude_val_list = []
    longitude_val_list = []
    utm_val_list = []
    for df_row_val in tqdm(fmow_val.iterrows()):
        img_name_val = df_row_val[1]["img_name"]
        gt_json_path_val = os.path.join(args.fmow_gt_dir, img_name_val[:-3] + "json")
        gt_json_val = json.load(open(gt_json_path_val, "r"))
        random.shuffle(geometa_template_fmow)
        geometa_cap_val, country_val, month_val, coord_val, utm_val = parse_fmow_df(gt_json_val, geometa_template_fmow[0])
        latitude_val, longitude_val = coord_val
        latitude_val_list.append(latitude_val)
        longitude_val_list.append(longitude_val)
        utm_val_list.append(utm_val)
        country_val_list.append(country_val)
        month_val_list.append(month_val)
        # print(img_name_val, geometa_cap_val)
        img_name_val_list.append(img_name_val)
        geometa_cap_val_list.append(geometa_cap_val)
    geometa_df_val = pd.DataFrame({
        "img_name": img_name_val_list,
        "geometa_cap": geometa_cap_val_list,
        "country": country_val_list,
        "month": month_val_list,
        "latitude": latitude_val_list,
        "longitude": longitude_val_list,
        "utm": utm_val_list
    })
    geometa_df_val.to_csv("fmow_geometa_val_coord.csv", index=False)


if __name__ == "__main__":
    main()