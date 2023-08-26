import os
import pandas as pd
import numpy as np
import argparse
import utm
import geopandas as gpd
import matplotlib.pyplot as plt
from collections import Counter


def draw_final_corrected_map(zone_data, lat_dict, cm_func, zones_numbers, zones_letters):
    # Convert zone to lat/lon with adjustment for the exceptional "X" zone

    def zone_to_lat_lon(zone):
        num = int(zone[:-1])
        letter = zone[-1]
        lon = -180 + (num - 1) * 6
        lat = lat_dict[letter]
        return lon, lat

    # Plot World Map and UTM Data
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    fig, ax = plt.subplots(figsize=(15, 10))
    world.plot(ax=ax, color='lightgrey')

    # Plot heatmap rectangles for each UTM zone with transparency
    for zone, count in zone_data.items():
        if zone[-1] != "X":  # Exclude zones above X
            lon, lat = zone_to_lat_lon(zone)
            rect = plt.Rectangle((lon, lat), 6, 8,
                                 color=cm_func(count),
                                 edgecolor='k', linewidth=0.5, alpha=0.5)
            ax.add_patch(rect)
        else:
            lon, lat = zone_to_lat_lon(zone)
            rect = plt.Rectangle((lon, lat), 6, 12,
                                 color=cm_func(count),
                                 edgecolor='k', linewidth=0.5, alpha=0.5)
            ax.add_patch(rect)

    # Draw UTM Gridlines
    for lon in range(-180, 181, 6):
        ax.axvline(lon, color='grey', linestyle='--', linewidth=0.5)

    # Adjusting the last gridline to reach 84 for the exceptional "X" zone
    for lat in list(range(-80, 80, 8)) + [84]:
        ax.axhline(lat, color='grey', linestyle='--', linewidth=0.5)

    # Set the y-ticks to be the letters and position them correctly
    letter_ticks = [lat + 4 for lat in range(-80, 80, 8)]  # Excludes the 84°N location
    letter_ticks[-1] = 78
    ax.set_yticks(letter_ticks)
    ax.set_yticklabels(zones_letters)

    # Set the x-ticks to be the UTM zone numbers and adjust the font size
    number_ticks = [-180 + (int(num) - 0.5) * 6 for num in zones_numbers]
    ax.set_xticks(number_ticks)
    ax.set_xticklabels(zones_numbers, fontsize=8)  # Adjusting font size

    ax.set_xlim([-180, 180])
    ax.set_ylim([-90, 90])
    ax.set_title('Heatmap of UTM Zones')
    fig.colorbar(
        plt.cm.ScalarMappable(
            cmap=cm_func,
            norm=plt.Normalize(
                0,
                int(np.median(list(zone_data.values()))),
                # 400,
                clip=True
            )
        ),
        ax=ax,
        label='Count (Long-tailed Distribution, Threshold by Median, {})'.format(int(np.median(list(zone_data.values()))))
    )

    plt.tight_layout()
    plt.show()


def draw_count_dist(zone_dict):
    plt.hist(zone_dict.values(), bins=100)
    plt.xlabel("# Images in UTM Zone")
    plt.ylabel("# UTM Zone")
    plt.title("UTM Zone Image Count")
    plt.tight_layout()
    plt.show()


def latlon_to_utm_zone(lat, lon):
    _, _, zone_number, zone_letter = utm.from_latlon(lat, lon)
    return f"{zone_number}{zone_letter}"


# Create "utm" column based on "longitude" and "latitude"


def extract_dataset_geoinfo(dataset_dict, dfs):
    geo_df = []
    for dataset_name, content in dataset_dict.items():
        print(dataset_name, content)
        df = dfs[dataset_name]
        for geo_type in content:
            if geo_type == "coordinate":
                df['utm'] = df.apply(lambda row: latlon_to_utm_zone(row['latitude'], row['longitude']), axis=1)
                sub_df = df[["img_name", "utm"]]
                geo_df.append(sub_df)
            elif geo_type == "utm":
                sub_df = df[["img_name", "utm"]]
                geo_df.append(sub_df)
    geo_info_df = pd.concat(geo_df)
    print(geo_info_df)
    return geo_info_df


def count_elements(lst):
    # Using a dictionary to store the count of each element
    count_dict = {}
    for item in lst:
        if item in count_dict:
            count_dict[item] += 1
        else:
            count_dict[item] = 1
    return count_dict


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--fmow_geometa_train_csv_path', type=str,
                        default="./fmow_geometa_train_coord.csv",
                        help='path of fmow geometa train csv')
    parser.add_argument('--fmow_geometa_val_csv_path', type=str,
                        default="./fmow_geometa_val_coord.csv",
                        help='path of fmow geometa val csv')
    parser.add_argument('--ben_geometa_train_csv_path', type=str,
                        default="./ben_geometa_train_coord.csv",
                        help='path of ben geometa train csv')
    parser.add_argument('--ben_geometa_val_csv_path', type=str,
                        default="./ben_geometa_val_coord.csv",
                        help='path of ben geometa val csv')
    parser.add_argument('--yfcc14m_geometa_csv_path', type=str,
                        default="./yfcc14m_geometa_coord.csv",
                        help='path of cc3m geometa csv')
    parser.add_argument('--save_info_path', type=str,
                        default="./geo_info.csv",
                        help='path of geo info statistics csv')

    args = parser.parse_args()

    dataset_dict = {
        "ben": ["utm"],
        "fmow": ["utm"],
        "yfcc14m": ["coordinate"]
    }

    fmow_df_train = pd.read_csv(args.fmow_geometa_train_csv_path)
    fmow_df_val = pd.read_csv(args.fmow_geometa_val_csv_path)
    fmow_df = pd.concat([
        fmow_df_train,
        fmow_df_val
    ])

    ben_geometa_train_df = pd.read_csv(args.ben_geometa_train_csv_path)
    ben_geometa_val_df = pd.read_csv(args.ben_geometa_val_csv_path)
    ben_geometa_train_df["img_name"] = ben_geometa_train_df["img_name"].apply(lambda x: str(x) + '.jpg')
    ben_df = pd.concat([
        ben_geometa_train_df,
        ben_geometa_val_df
    ])
    ben_country_list = ["Austria", "Belgium", "Finland", "Ireland", "Kosovo", "Lithuania", "Luxembourg", "Portugal", "Serbia", "Switzerland"]

    yfcc14m_df = pd.read_csv(args.yfcc14m_geometa_csv_path)
    yfcc14m_df["img_name"] = yfcc14m_df["img_name"].apply(lambda x: str(x) + '.jpg')
    yfcc14m_df.dropna(subset=['longitude', 'latitude'], inplace=True)

    dfs = {
        "fmow": fmow_df,
        "ben": ben_df,
        "yfcc14m": yfcc14m_df
    }

    geo_info_df = extract_dataset_geoinfo(dataset_dict, dfs)
    all_utm = geo_info_df["utm"].to_list()

    utm_count = Counter(all_utm)
    print([(element, count) for element, count in utm_count.most_common()])
    utm_count_dict = dict(utm_count)
    print(utm_count_dict)

    zone_dict = {k: v for k, v in sorted(utm_count_dict.items(), key=lambda item: item[1])}
    print(zone_dict)
    lat_dict = {
        "C": -80,
        "D": -72,
        "E": -64,
        "F": -56,
        "G": -48,
        "H": -40,
        "J": -32,
        "K": -24,
        "L": -16,
        "M": -8,
        "N": 0,
        "P": 8,
        "Q": 16,
        "R": 24,
        "S": 32,
        "T": 40,
        "U": 48,
        "V": 56,
        "W": 64,
        "X": 72,
    }

    zones_numbers = [str(i).zfill(2) for i in range(1, 61)]  # Representing as double-digit strings
    zones_letters = [chr(i) for i in range(67, 89) if i not in [73, 79]]  # C to X excluding I and O
    cm_func = plt.cm.Spectral
    # cm_func = plt.cm.plasma
    # Drawing the corrected heatmap using the provided data
    draw_count_dist(zone_dict)
    draw_final_corrected_map(zone_dict, lat_dict, cm_func, zones_numbers, zones_letters)
    print(sum(zone_dict.values()))


if __name__ == "__main__":
    main()
