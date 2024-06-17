from utils.basic_utils import load_json, save_jsonl
import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from modules.ndcg_iou import calculate_ndcg_iou

KS = [10, 20, 40]
TS = [0.3, 0.5, 0.7]


def validated_metric(pred_path, gt_path):
    video2idx = load_json("./data/TVR_Ranking_old/video_name_duration_id.json")
    video2idx = {k: v[1] for k, v in video2idx.items()}
    idx2video = {v: k for k, v in video2idx.items()}

    pred_data= load_json(pred_path)
    gt_data = load_json(gt_path)
    gt_data = {data["query_id"]: data["relevant_moment"] for data in gt_data}


    all_pred = []
    for pred_one_query in pred_data:
        pred_result = []
        query_id = pred_one_query["query_id"]
        for i in pred_one_query["predictions"]:
            vid, s, e, score = i
            video_name = idx2video[vid]
            pred_result.append({
            "video_name": video_name,
            "timestamp": [s, e],
            "model_scores": score,
            "query_id": query_id,
        })
        all_pred.append(pred_result)
        
    all_gt = []
    for pred_one_query in pred_data:
        pred_result = []
        query_id = pred_one_query["query_id"]
        all_gt.append(gt_data[query_id])

    average_ndcg = calculate_ndcg_iou(all_gt, all_pred, TS, KS)
    for K, vs in average_ndcg.items():
        for T, v in vs.items():
            print(f"{data_type} NDCG@{K}, IoU={T}: {v:.4f}")

for data_type in ["val", "test"]:
    gt_path = f"data/TVR_Ranking/{data_type}.json"
    
    # for pred_path in [ f"./results/author/CONQUER_top01_20240330_175853/best_{data_type}_predictions.json",
    #                    f"./results/author/CONQUER_top20_20240330_175925/best_{data_type}_predictions.json",
    #                    f"./results/author/CONQUER_top40_20240330_175929/best_{data_type}_predictions.json"]:

    # for pred_path in [ f"./results/author/XML_top01_20240328_132029/best_{data_type}_predictions.json",
    #                    f"./results/author/XML_top20_20240327_152318/best_{data_type}_predictions.json",
    #                    f"./results/author/XML_top40_20240327_152350/best_{data_type}_predictions.json"]:
    
    for pred_path in [ f"./results/author/ReLoCLNet_top_01_20240326_115033/best_{data_type}_predictions.json",
                       f"./results/author/ReLoCLNet_top_20_20240326_162832/best_{data_type}_predictions.json",
                       f"./results/author/ReLoCLNet_top_40_20240326_163257/best_{data_type}_predictions.json"]:


        print(pred_path)
        validated_metric(pred_path, gt_path)
        print("")