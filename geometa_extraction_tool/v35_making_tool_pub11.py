import os
import pickle as pkl
import argparse
import pandas as pd


def split_df(merged_df, train_df_old, val_df_old):
    train_names = train_df_old["img_name"].to_list()
    val_names = val_df_old["img_name"].to_list()

    train_df = merged_df[merged_df["img_name"].isin(train_names)]
    val_df = merged_df[merged_df["img_name"].isin(val_names)]

    train_df_combined = pd.merge(train_df, train_df_old, on="img_name", how="outer")
    val_df_combined = pd.merge(val_df, val_df_old, on="img_name", how="outer")

    return train_df_combined, val_df_combined


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--label_pkl', type=str,
                        default="/Users/zilun/Desktop/RS5M_v4_dataset/RS5M_v3/pub11/RS5M_pub11_label.pkl",
                        help='path of pub11 pkl')
    parser.add_argument('--old_train_csv_path', type=str,
                        default="/Users/zilun/Desktop/RS5M_v4_dataset/RS5M_v3/pub11/RS5M_pub11_train.csv",
                        help='path of pub11 val csv')
    parser.add_argument('--old_val_csv_path', type=str,
                        default="/Users/zilun/Desktop/RS5M_v4_dataset/RS5M_v3/pub11/RS5M_pub11_val.csv",
                        help='path of pub11 val csv')
    parser.add_argument('--cc3m_geometa_csv_path', type=str,
                        default="/Users/zilun/Desktop/nips_rebuttal/geometa/cc3m_geometa.csv",
                        help='path of cc3m geometa csv')
    parser.add_argument('--yfcc14m_geometa_csv_path', type=str,
                        default="/Users/zilun/Desktop/nips_rebuttal/geometa/yfcc14m_geometa.csv",
                        help='path of cc3m geometa csv')
    parser.add_argument('--redcaps_geometa_csv_path', type=str,
                        default="/Users/zilun/Desktop/nips_rebuttal/geometa/redcaps_geometa.csv",
                        help='path of cc3m geometa csv')

    parser.add_argument('--new_dataset_dir', type=str,
                        default="/Users/zilun/Desktop/RS5M_v4_dataset/RS5M_v4/pub11",
                        help='path of pub11/rs3 pkl')
    parser.add_argument('--new_label_pkl', type=str,
                        default="/Users/zilun/Desktop/RS5M_v4_dataset/RS5M_v4/pub11/RS5M_pub11_label.pkl",
                        help='path of pub11/rs3 pkl')
    parser.add_argument('--new_train_csv_path', type=str,
                        default="/Users/zilun/Desktop/RS5M_v4_dataset/RS5M_v4/pub11/RS5M_pub11_train.csv",
                        help='path of pub11 val csv')
    parser.add_argument('--new_val_csv_path', type=str,
                        default="/Users/zilun/Desktop/RS5M_v4_dataset/RS5M_v4/pub11/RS5M_pub11_val.csv",
                        help='path of pub11 val csv')

    args = parser.parse_args()
    label_df = pkl.load(open(args.label_pkl, "rb"))
    cc3m_geometa_df = pd.read_csv(args.cc3m_geometa_csv_path)
    yfcc14m_geometa_df = pd.read_csv(args.yfcc14m_geometa_csv_path)
    redcaps_geometa_df = pd.read_csv(args.redcaps_geometa_csv_path)
    add_df = pd.concat([
        cc3m_geometa_df,
        yfcc14m_geometa_df,
        redcaps_geometa_df
    ])
    del add_df["text"]
    del add_df["url"]
    del add_df["download_status"]
    merged_df = pd.merge(label_df, add_df, on="img_name", how="outer")

    train_df_old = pd.read_csv(args.old_train_csv_path)
    val_df_old = pd.read_csv(args.old_val_csv_path)
    train_df_old_filtered = train_df_old[~train_df_old["img_name"].isin(val_df_old["img_name"])]

    print("Count for meta cap from cc3m, yfcc14m, redcaps, and summary: {}, {}, {}, {}, {}".format(
        cc3m_geometa_df["meta_caption"].count(),
        yfcc14m_geometa_df["meta_caption"].count(),
        redcaps_geometa_df["meta_caption"].count(),
        add_df["meta_caption"].count(),
        merged_df["meta_caption"].count()
    ))

    train_df_combined, val_df_combined = split_df(merged_df, train_df_old_filtered, val_df_old)
    os.makedirs(args.new_dataset_dir, exist_ok=True)
    train_df_combined.to_csv(args.new_train_csv_path, index=False)
    val_df_combined.to_csv(args.new_val_csv_path, index=False)
    pkl.dump(merged_df, open(args.new_label_pkl, "wb"))


if __name__ == "__main__":
    main()
