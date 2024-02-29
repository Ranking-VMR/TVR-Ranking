from utils.basic_utils import load_jsonl, save_jsonl, save_json
import random
random.seed(2024)
import os
import pandas as pd



def collect_dataset(raw_data_path, out_data_path, data_type):
    data = load_jsonl(raw_data_path)
    df = pd.DataFrame(data)
    if data_type == "train":
        top_per_qid = df.groupby('qid').apply(lambda x: x.sort_values('similarity', ascending=False).head(top)).reset_index(drop=True)
    else:
        top_per_qid = df.groupby('qid').apply(lambda x: x.sort_values('final_match_score', ascending=False).head(top)).reset_index(drop=True)
    top_data = top_per_qid.to_dict('records')

    new_data = []
    for row in top_data:
        tmp = {"vid_name": row["vname"],
                "duration": row["duration"],
                "desc": row["query"],
                'type': 'v',
                "desc_id":row["qid"],
                "caption": row["caption"],
                }
        if data_type == "train":
            tmp["similarity"] =  row["similarity"]
            tmp["ts"] =  eval(row["timestamp"])
        else:
            tmp["ts"] =  row["timestamp"]
            tmp["match_score"] =  row["final_match_score"]
        new_data.append(tmp)

    floder = os.path.split(out_data_path)[0]
    os.makedirs(floder, exist_ok=True)
    save_jsonl(new_data, out_data_path)


top = 5
raw_data_path = "data/TVR_Ranking_raw/train_raw.jsonl"
out_data_path =  f"data/TVR_Ranking_{top}/train.jsonl"
collect_dataset(raw_data_path, out_data_path, "train")

# raw_data_path = "data/TVR_Ranking_raw/test_raw.jsonl"
# out_data_path =  f"data/TVR_Ranking_{top}/test.jsonl"
# collect_dataset(raw_data_path, out_data_path, "test")

# raw_data_path = "data/TVR_Ranking_raw/val_raw.jsonl"
# out_data_path =  f"data/TVR_Ranking_{top}/val.jsonl"
# collect_dataset(raw_data_path, out_data_path, "val")




## test record
# {"pid": 20, "qid": 10821, "query": "While a person is holding a man's right arm up, the man lifts his left arm and another person moves closer, the man looks up at the other person and then down.", "vname": "house_s08e11_seg02_clip_10", "timestamp": [41.4, 46.29], "duration": 89.02, "caption": "While a person is holding a man's right arm up, the man lifts his left arm and another person moves closer, the man looks up at the other person and then down.", "raw_match_scores": [5, 5], "similarity": 1.0000001192092896, "workers": ["03", "04"], "consensus": "Pass", "final_match_score": 5, "trimmed_variance": 0.0}

## traing record
# {'pid': 346640, 'qid': 72922, 'vname': 'castle_s04e10_seg02_clip_13', 'timestamp': '[46.31, 49.46]', 'duration': 89.93, 'query': 'A person moves the fingerprints around on the monitor.', 'caption': 'A person moves the fingerprints around on the monitor.', 'similarity': 1.0}
# {'pid': 346641, 'qid': 72922, 'vname': 'castle_s04e10_seg02_clip_14', 'timestamp': '[46.31, 49.46]', 'duration': 109.93, 'query': 'A person moves the fingerprints around on the monitor.', 'caption': 'ingerprints around on the monitor.', 'similarity': 0.9}

# the train data like above
# For each "qid",  select top 10 ordered by similarity 


## target record
# {"vid_name": "friends_s01e03_seg02_clip_19", "duration": 61.46, "ts": [16.48, 33.87], "desc": "Phoebe puts one of her ponytails in her mouth.", "type": "v", "desc_id": 90200}
