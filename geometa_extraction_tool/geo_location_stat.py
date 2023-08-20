import pandas as pd
from collections import Counter


pub11_train_path = "/home/zilun/RS5M_v4/RS5M_metadata/pub11/RS5M_pub11_train.csv"
pub11_val_path = "/home/zilun/RS5M_v4/RS5M_metadata/pub11/RS5M_pub11_validation.csv"
rs3_train_path = "/home/zilun/RS5M_v4/RS5M_metadata/rs3/RS5M_rs3_train.csv"
rs3_val_path = "/home/zilun/RS5M_v4/RS5M_metadata/rs3/RS5M_rs3_validation.csv"


pub11_train = pd.read_csv(pub11_train_path)
pub11_val = pd.read_csv(pub11_val_path)
rs3_train = pd.read_csv(rs3_train_path)
rs3_val = pd.read_csv(rs3_val_path)

pub11 = pd.concat([pub11_train, pub11_val])
rs3 = pd.concat([rs3_train, rs3_val])

pub11_country = pub11["country"].dropna().to_list()
pub11_month = pub11["month"].dropna().to_list()

rs3_country = rs3["country"].dropna().to_list()
rs3_month = rs3["month"].dropna().to_list()

pub11_country = [country.lower() for country in pub11_country]
rs3_country = [country.lower() for country in rs3_country]
all_country = pub11_country + rs3_country
all_month = pub11_month + rs3_month

print("all country: {}, all month: {}".format(len(all_country), len(all_month)))

country_count = Counter(all_country)
print([(element, count) for element, count in country_count.most_common()])

month_count = Counter(all_month)
print([(element, count) for element, count in month_count.most_common()])

non_repeat_country = list(set(all_country))
print(len(non_repeat_country))
print(len(all_country) / len(non_repeat_country))



