import pandas as pd
import os
import argparse
from tqdm import tqdm
import json
import pickle as pkl
import numpy as np
import random
from helper import latlon_to_city_country, convert_yfcc_timestamp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pub11_label_pkl', type=str,
                        default="/media/zilun/wd-16/RS5M_v3/pub11/RS5M_pub11_label.pkl",
                        help='path of pub11 pkl')
    parser.add_argument('--geometa_template_path', type=str,
                        default="./geometa_template.json",
                        help='path of geometa_template json')
    parser.add_argument('--yfcc_info_path', type=str,
                        default="./yfcc_geometa_selected_rows.pkl",
                        help='path of yfcc info pkl')
    args = parser.parse_args()

    geometa_template = json.load(open(args.geometa_template_path, "r"))
    pub11_label = pkl.load(open(args.pub11_label_pkl, "rb"))
    pub11_label["meta_caption"] = None
    pub11_label["country"] = None
    pub11_label["month"] = None
    pub11_label["longitude"] = None
    pub11_label["latitude"] = None

    yfcc_subset_df = pub11_label[pub11_label["img_name"].str.contains("yfcc")]
    yfcc_info_list = pkl.load(open(args.yfcc_info_path, "rb"))

    count = 0
    none_country = 0
    has_coord = 0
    is_country = 0
    for df_row in tqdm(yfcc_subset_df.iterrows()):
        url = df_row[1]["url"]
        for line in yfcc_info_list:
            photo_id, user_id, user_name, date_taken, date_upload, device, title, description, user_tag, machine_tag, longitude, latitude, _, page_url, download_url, _, _, _, _, _, _, ext, photo_marker = line
            country, timestamp, season, cls_labels = None, None, None, None
            if url == download_url:
                if latitude and longitude is not None:
                    latitude = float(latitude)
                    longitude = float(longitude)
                    country = latlon_to_city_country(latitude, longitude)
                    df_row[1]["longitude"] = longitude
                    df_row[1]["latitude"] = latitude
                    has_coord += 1

                if date_taken is not None:
                    season, day, month, year, hour = convert_yfcc_timestamp(date_taken)
                    timestamp = "{} o'clock, {} {}, {}".format(hour, month, day, year)

                if cls_labels is not None:
                    pass

                if country is not None and timestamp is None:
                    geometa_template_yfcc = geometa_template["yfcc14m_country"]
                    random.shuffle(geometa_template_yfcc)
                    geometa_cap = geometa_template_yfcc[0].replace("{country}", country)

                elif country is None and timestamp is not None:
                    geometa_template_yfcc = geometa_template["yfcc14m_timestamp"]
                    random.shuffle(geometa_template_yfcc)
                    geometa_cap = geometa_template_yfcc[0].replace("{season}", season).replace("{timestamp}", timestamp)

                elif country is not None and timestamp is not None:
                    geometa_template_yfcc = geometa_template["yfcc14m_country_timestamp"]
                    random.shuffle(geometa_template_yfcc)
                    geometa_cap = geometa_template_yfcc[0].replace("{country}", country).replace("{season}", season).replace("{timestamp}", timestamp)

                else:
                    count += 1

                df_row[1]["meta_caption"] = geometa_cap

                if country is not None:
                    df_row[1]["country"] = country.split(", ")[-1]
                    is_country += 1
                else:
                    none_country += 1

                if month is not None:
                    df_row[1]["month"] = month.split(", ")[-1]

    print("There are {} non-country, {}, country, {} coord".format(none_country, is_country, has_coord))
    yfcc_subset_df.to_csv("yfcc14m_geometa_coord.csv", index=False)


if __name__ == "__main__":
    main()