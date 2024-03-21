from utils.basic_utils import load_jsonl, save_jsonl, load_json


data = load_jsonl("./data/TVR_Ranking_raw/train_raw.jsonl")


new_data = []
for i in data:
    tmp = {"pair_id": i["pid"],
           "query_id": i["qid"],
           "query": i["query"],
           "video_name": i["vname"],
           "timestamp": i["timestamp"],
           "duration": i["duration"],
           "caption": i["caption"],
           "similarity": 1.0}

save_jsonl(data, "./data/TVR_Ranking/train_top40.jsonl")
