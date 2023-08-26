import os
import pickle as pkl
import argparse
import pandas as pd


def split_df(merged_df, train_df_old, val_df_old):
    train_names = train_df_old["img_name"].to_list()
    val_names = val_df_old["img_name"].to_list()

    train_df = merged_df[merged_df["img_name"].isin(train_names)]
    val_df = merged_df[merged_df["img_name"].isin(val_names)]

    return train_df, val_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--old_label_pkl', type=str,
                        default="/Users/zilun/Desktop/RS5M_v4_dataset/RS5M_v3/rs3/RS5M_rs3_label.pkl",
                        help='path of old rs3 pkl')
    parser.add_argument('--old_train_csv_path', type=str,
                        default="/Users/zilun/Desktop/RS5M_v4_dataset/RS5M_v3/rs3/RS5M_rs3_train.csv",
                        help='path of old rs3 val csv')
    parser.add_argument('--old_val_csv_path', type=str,
                        default="/Users/zilun/Desktop/RS5M_v4_dataset/RS5M_v3/rs3/RS5M_rs3_val.csv",
                        help='path of old rs3 val csv')
    parser.add_argument('--fmow_geometa_train_csv_path', type=str,
                        default="/Users/zilun/Desktop/nips_rebuttal/geometa/fmow_geometa_train.csv",
                        help='path of fmow geometa train csv')
    parser.add_argument('--fmow_geometa_val_csv_path', type=str,
                        default="/Users/zilun/Desktop/nips_rebuttal/geometa/fmow_geometa_val.csv",
                        help='path of fmow geometa val csv')
    parser.add_argument('--ben_geometa_train_csv_path', type=str,
                        default="/Users/zilun/Desktop/nips_rebuttal/geometa/ben_geometa_train.csv",
                        help='path of ben geometa train csv')
    parser.add_argument('--ben_geometa_val_csv_path', type=str,
                        default="/Users/zilun/Desktop/nips_rebuttal/geometa/ben_geometa_val.csv",
                        help='path of ben geometa val csv')

    parser.add_argument('--new_dataset_dir', type=str,
                        default="/Users/zilun/Desktop/RS5M_v4_dataset/RS5M_v4/rs3",
                        help='path of new rs3 pkl')
    parser.add_argument('--new_label_pkl', type=str,
                        default="/Users/zilun/Desktop/RS5M_v4_dataset/RS5M_v4/rs3/RS5M_rs3_label.pkl",
                        help='path of new rs3 pkl')
    parser.add_argument('--new_train_csv_path', type=str,
                        default="/Users/zilun/Desktop/RS5M_v4_dataset/RS5M_v4/rs3/RS5M_rs3_train.csv",
                        help='path of new rs3 train csv')
    parser.add_argument('--new_val_csv_path', type=str,
                        default="/Users/zilun/Desktop/RS5M_v4_dataset/RS5M_v4/rs3/RS5M_rs3_val.csv",
                        help='path of new rs3 val csv')

    args = parser.parse_args()
    label_df = pkl.load(open(args.old_label_pkl, "rb"))
    fmow_geometa_train_df = pd.read_csv(args.fmow_geometa_train_csv_path)
    fmow_geometa_val_df = pd.read_csv(args.fmow_geometa_val_csv_path)
    ben_geometa_train_df = pd.read_csv(args.ben_geometa_train_csv_path)
    ben_geometa_val_df = pd.read_csv(args.ben_geometa_val_csv_path)
    ben_geometa_train_df["img_name"] = ben_geometa_train_df["img_name"].apply(lambda x: str(x) + '.jpg')

    add_fmow_df = pd.concat([
        fmow_geometa_train_df,
        fmow_geometa_val_df
    ])

    add_ben_df = pd.concat([
        ben_geometa_train_df,
        ben_geometa_val_df
    ])

    add_df = pd.concat([
        add_fmow_df,
        add_ben_df
    ])

    add_df["meta_caption"] = add_df["geometa_cap"]
    del add_df["geometa_cap"]
    merged_df = pd.merge(label_df, add_df, on="img_name", how="outer")

    train_df_old = pd.read_csv(args.old_train_csv_path)
    val_df_old = pd.read_csv(args.old_val_csv_path)

    print("Count for meta cap from fmow, ben, and summary: {}, {}, {}, {}".format(
        add_fmow_df["geometa_cap"].count(),
        add_ben_df["geometa_cap"].count(),
        add_df["meta_caption"].count(),
        merged_df["meta_caption"].count()
    ))

    train_df_combined, val_df_combined = split_df(merged_df, train_df_old, val_df_old)
    os.makedirs(args.new_dataset_dir, exist_ok=True)
    train_df_combined.to_csv(args.new_train_csv_path, index=False)
    val_df_combined.to_csv(args.new_val_csv_path, index=False)
    pkl.dump(merged_df, open(args.new_label_pkl, "wb"))


if __name__ == "__main__":
    main()
