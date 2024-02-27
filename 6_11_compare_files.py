import h5py
from utils.basic_utils import load_jsonl, save_jsonl, save_json

import h5py

def compare_h5_structure(file1, file2):
    with h5py.File(file1, 'r') as f1, h5py.File(file2, 'r') as f2:
        f1_keys = set(f1.keys())
        f2_keys = set(f2.keys())
        
        print("Keys in file1 not in file2:", f1_keys - f2_keys)
        print("Keys in file2 not in file1:", f2_keys - f1_keys)
        
        common_keys = f1_keys & f2_keys
        for key in common_keys:
            print(f"Comparing dataset '{key}':")
            print("Shape in file1:", f1[key].shape)
            print("Shape in file2:", f2[key].shape)
            print("Data type in file1:", f1[key].dtype)
            print("Data type in file2:", f2[key].dtype)

# You would call it like this:
# compare_h5_structure("/storage_fast/rjliang/tvr/feat/bert_feature/query_only/tvr_query_pretrained_w_query.h5", "./data/TVR_Ranking/query_bert2.h5")





def compare_json_structure(file1, file2):
    data1 = load_jsonl(file1)
    data2 = load_jsonl(file2)
    
    for i, j in zip(data1, data2):
        print(i)
        print(j)
        input()
# For train data
compare_json_structure("data/TVR/tvr_train_release.jsonl", "data/TVR_Ranking_Single/train_single.jsonl")

# For test data
compare_json_structure("data/TVR/tvr_val_release.jsonl", "data/TVR_Ranking_Single/test_single.jsonl")


# compare the struction for this three pairs