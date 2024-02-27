from utils.basic_utils import load_jsonl, save_jsonl, save_json
import random
random.seed(2024)

data = load_jsonl("data/TVR_Ranking_raw/train_raw.jsonl")
single_data = []

for row in data:
    qid = row["qid"]
    tmp = {"vid_name": row["vname"],
            "duration": row["duration"],
            "ts": eval(row["timestamp"]),
            # "ts": row["timestamp"],
            "desc": row["query"],
            'type': 'v',
            "desc_id":row["qid"],
            "caption": row["caption"],
            "similarity": row["similarity"]
            }
    single_data.append(tmp)
    
save_jsonl(single_data, "data/TVR_Ranking_raw/train_tvrr.jsonl")

d
## test record
# {"pid": 20, "qid": 10821, "query": "While a person is holding a man's right arm up, the man lifts his left arm and another person moves closer, the man looks up at the other person and then down.", "vname": "house_s08e11_seg02_clip_10", "timestamp": [41.4, 46.29], "duration": 89.02, "caption": "While a person is holding a man's right arm up, the man lifts his left arm and another person moves closer, the man looks up at the other person and then down.", "raw_match_scores": [5, 5], "similarity": 1.0000001192092896, "workers": ["03", "04"], "consensus": "Pass", "final_match_score": 5, "trimmed_variance": 0.0}

## traing record
# {'pid': 346640, 'qid': 72922, 'vname': 'castle_s04e10_seg02_clip_13', 'timestamp': '[46.31, 49.46]', 'duration': 89.93, 'query': 'A person moves the fingerprints around on the monitor.', 'caption': 'A person moves the fingerprints around on the monitor.', 'similarity': 1.0}

## target record
# {"vid_name": "friends_s01e03_seg02_clip_19", "duration": 61.46, "ts": [16.48, 33.87], "desc": "Phoebe puts one of her ponytails in her mouth.", "type": "v", "desc_id": 90200}
