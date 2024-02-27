from utils.basic_utils import load_jsonl, save_jsonl
import random
random.seed(2024)

raw_data_path = "./data/TVR_Ranking/raw_anno_tmp.jsonl"
raw_data = load_jsonl(raw_data_path)

qid_set = [i["qid"] for i in raw_data]
val_qid = random.sample(qid_set, 500)
test_qid = list(set(qid_set) - set(val_qid))

test_data = [i for i in raw_data if i["qid"] in test_qid]
val_data = [i for i in raw_data if i["qid"] in val_qid]


save_jsonl(test_data, "./data/TVR_Ranking/test_tvrr.jsonl")
save_jsonl(val_data, "./data/TVR_Ranking/val_tvrr.jsonl")