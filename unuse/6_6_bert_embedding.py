from utils.basic_utils import load_jsonl, save_jsonl, save_json
import h5py

import torch
import h5py
import numpy as np
from tqdm import tqdm

train_data = load_jsonl("data/TVR_Ranking_Single/train_single.jsonl")
val_data = load_jsonl("data/TVR_Ranking_Single/val_single.jsonl")
test_data = load_jsonl("data/TVR_Ranking_Single/test_single.jsonl")
all_data = train_data + val_data + test_data

from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained('bert-base-cased')
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
print(device)

# # Prepare the HDF5 file
k_set = set()
name = "./data/TVR_Ranking/query_bert2.h5"
with h5py.File(name, "w") as h5file:
    for row in tqdm(all_data):
        query = row["desc"]
        qid = str(row["query_id"])
        # Tokenize and encode the queries for BERT
        inputs = tokenizer(query, return_tensors="pt", padding=False, truncation=True, max_length=512).to(device)

        with torch.no_grad():
            output = model(**inputs)
        embedding = output.last_hidden_state.squeeze(0).cpu().numpy().astype(np.float32)

        if qid not in k_set:
            k_set.add(qid)
            h5file.create_dataset(qid, data=embedding)
 
# old_query_embedding =  h5py.File("/storage_fast/rjliang/tvr/feat/bert_feature/query_only/tvr_query_pretrained_w_query.h5")
# for k, v in old_query_embedding.items():
#     print(k, v.shape)
#     break
# print(len(old_query_embedding))

new_query_embedding =  h5py.File(name)
for k, v in new_query_embedding.items():
    print(k, v.shape)
    break
print(len(new_query_embedding))
print(len(all_data))

if "24146" in new_query_embedding.keys():
    print("yes")
else:
    print("no")