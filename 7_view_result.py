from utils.basic_utils import load_jsonl, save_jsonl, load_json


result_path = "results/single_train/tvr-video_sub_tef-demo-2024_02_26_18_11_17/best_tvr_val_predictions_VCMR.json"

data = load_json(result_path)
print(type(data))
for k, v in data.items():
    print(k, len(v))

print(data["VCMR"][0])
# video2idx 19614
# VCMR 10895