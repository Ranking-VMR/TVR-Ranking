from utils.basic_utils import load_jsonl, save_jsonl, save_json



raw_data_path = "./data/tvr_all_release.jsonl"
raw_data = load_jsonl(raw_data_path)

# {"vid_name": "friends_s09e23-24_seg02_clip_33", "duration": 61.03,
# "ts": [0, 2.75], "desc": "Phoebe picks up a drink from the bar.", "type": "v", "desc_id": 28592}

video_set = set()
for i in raw_data:
    video_set.add(i['vid_name']) 

# video_list = [[v, k] for v, k in video_dict.items()]

# save_jsonl(video_list, "./data/TVR_Ranking/video_corpus.jsonl")
# print(len(video_list))

raw_data = load_jsonl("./data/tvr_video2dur_idx.json")[0]
new_data = []

video_dict = dict()
for k, dt in raw_data.items():
    for k, v in dt.items():
        if k in video_set:
            video_dict[k] = v
print(len(video_dict))
save_json(video_dict, "./data/TVR_Ranking/video_corpus.json")
