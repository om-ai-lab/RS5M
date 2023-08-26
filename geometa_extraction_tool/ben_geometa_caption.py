import pandas as pd
import os
import argparse
from tqdm import tqdm
import json
from helper import countrycode2countryname, parse_ben_df
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rs3_train_csv_path', type=str,
                        default="/media/zilun/wd-16/RS5M_v3/rs3/RS5M_rs3_train.csv",
                        help='path of rs3 train csv')
    parser.add_argument('--rs3_val_csv_path', type=str,
                        default="/media/zilun/wd-16/RS5M_v3/rs3/RS5M_rs3_val.csv",
                        help='path of rs3 val csv')
    parser.add_argument('--ben_gt_csv', type=str,
                        default="./ben/BENs2_info.csv",
                        help='dir of ben gt')
    parser.add_argument('--geometa_template_path', type=str,
                        default="./geometa_template.json",
                        help='path of geometa_template json')

    args = parser.parse_args()

    geometa_template = json.load(open(args.geometa_template_path, "r"))
    geometa_template_ben = geometa_template["ben"]
    ben_info = pd.read_csv(args.ben_gt_csv, index_col=0)

    rs3_train_csv = pd.read_csv(args.rs3_train_csv_path)
    ben_train = rs3_train_csv[rs3_train_csv["subset_name"] == "ben"]
    img_name_train_list = []
    geometa_cap_train_list = []
    month_train_list = []
    utm_train_list = []
    for df_row_train in tqdm(ben_train.iterrows()):
        img_name_train = int(df_row_train[1]["img_name"].split(".")[0])
        random.shuffle(geometa_template_ben)
        ben_img_info = ben_info[ben_info["img_name"] == img_name_train]
        geometa_cap_train, month_train, utm_train = parse_ben_df(ben_img_info, geometa_template_ben[0])
        utm_train_list.append(utm_train)
        # print(img_name_train, geometa_cap)
        img_name_train_list.append(img_name_train)
        geometa_cap_train_list.append(geometa_cap_train)
        month_train_list.append(month_train)
    geometa_df_train = pd.DataFrame({
        "img_name": img_name_train_list,
        "geometa_cap": geometa_cap_train_list,
        "country": None,
        "month": month_train_list,
        "utm": utm_train_list
    })
    geometa_df_train.to_csv("ben_geometa_train_coord.csv", index=False)

    rs3_val_csv = pd.read_csv(args.rs3_val_csv_path)
    ben_val = rs3_val_csv[rs3_val_csv["subset_name"] == "ben"]
    img_name_val_list = []
    geometa_cap_val_list = []
    month_val_list = []
    utm_val_list = []
    for df_row_val in tqdm(ben_val.iterrows()):
        img_name_val = int(df_row_val[1]["img_name"].split(".")[0])
        random.shuffle(geometa_template_ben)
        ben_img_info = ben_info[ben_info["img_name"] == img_name_val]
        geometa_cap_val, month_val, utm_val = parse_ben_df(ben_img_info, geometa_template_ben[0])
        utm_val_list.append(utm_val)
        # print(img_name_val, geometa_cap_val)
        img_name_val_list.append(df_row_val[1]["img_name"])
        geometa_cap_val_list.append(geometa_cap_val)
        month_val_list.append(month_val)
    geometa_df_val = pd.DataFrame({
        "img_name": img_name_val_list,
        "geometa_cap": geometa_cap_val_list,
        "country": None,
        "month": month_val_list,
        "utm": utm_val_list
    })
    geometa_df_val.to_csv("ben_geometa_val_coord.csv", index=False)


if __name__ == "__main__":
    main()