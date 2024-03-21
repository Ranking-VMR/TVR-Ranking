from utils.basic_utils import load_jsonl, save_jsonl, save_json
import random
random.seed(2024)
import os
import pandas as pd


def collect_dataset(raw_data_path, out_data_path, top_k):
    data = load_jsonl(raw_data_path)
    df = pd.DataFrame(data)
    top_per_qid = df.groupby('query_id').apply(lambda x: x.sort_values('similarity', ascending=False).head(top_k)).reset_index(drop=True)
    top_data = top_per_qid.to_dict('records')

    floder = os.path.split(out_data_path)[0]
    os.makedirs(floder, exist_ok=True)
    save_jsonl(top_data, out_data_path)


top = 1
raw_data_path = "data/TVR_Ranking/train_top40.jsonl"
out_data_path =  f"data/TVR_Ranking/train_top{top:02d}.jsonl"
collect_dataset(raw_data_path, out_data_path, top)
print(raw_data_path, "->", out_data_path)

# raw_data_path = "data/TVR_Ranking_raw/test_raw.jsonl"
# out_data_path =  f"data/TVR_Ranking_{top}/test.jsonl"
# collect_dataset(raw_data_path, out_data_path, "test")

# raw_data_path = "data/TVR_Ranking_raw/val_raw.jsonl"
# out_data_path =  f"data/TVR_Ranking_{top}/val.jsonl"
# collect_dataset(raw_data_path, out_data_path, "val")



