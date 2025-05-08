import json
import pandas as pd
from sklearn.model_selection import train_test_split


def convert_to_json_format(df):
    return [
        {
            "instruction": row['content'],
            "input": "",
            "output": str(row['toxic'])
        }
        for _, row in df.iterrows()
    ]


df = pd.read_csv('./datasets/Full_Perturbation/emoji_full.tsv', on_bad_lines='skip', quoting=0, sep='\t')

train_df, res_df = train_test_split(df, test_size=0.2, random_state=114514, shuffle=True)
val_df, test_df = train_test_split(res_df, test_size=0.5, random_state=1919810, shuffle=True)

train_data = convert_to_json_format(train_df)
val_data = convert_to_json_format(val_df)
test_data = convert_to_json_format(test_df)


with open('train.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

with open('val.json', 'w', encoding='utf-8') as f:
    json.dump(val_data, f, ensure_ascii=False, indent=4)

with open('test.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)

print("saved")
