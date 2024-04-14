from utils.basic_utils import load_jsonl, save_jsonl, load_json
import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import defaultdict

def calculate_iou(pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> float:
    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    intersection = max(0, intersection_end - intersection_start)
    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
    return intersection / union if union > 0 else 0


# Function to calculate DCG
def calculate_dcg(scores):
    return sum((2**score - 1) / np.log2(idx + 2) for idx, score in enumerate(scores))

# Function to calculate NDCG
def calculate_ndcg(pred_scores, true_scores):
    dcg = calculate_dcg(pred_scores)
    idcg = calculate_dcg(sorted(true_scores, reverse=True))
    return dcg / idcg if idcg > 0 else 0



def recall_iou_ndcg(gt_data, pred_data, video2idx, idx2video, video2duration, TS, KS):
    performance = defaultdict(lambda: defaultdict(list))
    performance_avg = defaultdict(lambda: defaultdict(float))
    
    ###### ------------------ #######

    gt_data_tmp = pd.DataFrame(gt_data)
    gt_data_tmp["start_time"] = gt_data_tmp["timestamp"].apply(lambda x: x[0])
    gt_data_tmp["end_time"] = gt_data_tmp["timestamp"].apply(lambda x: x[1])
    gt_data_tmp["vid"] = gt_data_tmp["video_name"].map(video2idx) 
    gt_data = gt_data_tmp[["query_id", "query", "video_name", "vid", "start_time", "end_time", "relevance", "duration"]]
    
    for i in tqdm(range(len(pred_data))):
        one_query_preds = pred_data[i]
        qid = one_query_preds["query_id"]
        one_query_preds = one_query_preds["predictions"]
        one_query_preds_df = pd.DataFrame(one_query_preds, columns=["vid", "start_time", "end_time", "model_scores"])
        one_query_preds_df["video_name"] = one_query_preds_df["vid"].map(idx2video) 
        one_query_preds_df["duration"] = one_query_preds_df["video_name"].map(video2duration) 

        one_query_gts = gt_data[gt_data["query_id"]==qid].sort_values(by="relevance", ascending=False).reset_index(drop=True)

        for T in TS:
            one_query_gts_drop = one_query_gts.copy()
            predictions_with_scores = []
            for index, pred in one_query_preds_df.iterrows():
                pred_vid, pred_st, pred_ed = pred["vid"], pred["start_time"], pred["end_time"]
                matched_rows = one_query_gts_drop[one_query_gts_drop["vid"] == pred_vid].reset_index(drop=True)
                
                if matched_rows.empty:
                    pred["pred_relevance"] = 0
                else:
                    matched_rows["iou"] = matched_rows.apply(lambda row: calculate_iou(pred_st, pred_ed, row["start_time"], row["end_time"]), axis=1)
                    max_iou_idx = matched_rows["iou"].idxmax()
                    max_iou_row = matched_rows.iloc[max_iou_idx]
                    
                    if max_iou_row["iou"] > T:
                        pred["pred_relevance"] = max_iou_row["relevance"]
                        # Remove the matched ground truth row
                        one_query_gts_drop = one_query_gts_drop.drop(index=matched_rows.index[max_iou_idx]).reset_index(drop=True)
                    else:
                        pred["pred_relevance"] = 0
                
                predictions_with_scores.append(pred)
                predictions_with_scores = predictions_with_scores
            one_query_preds_df = pd.DataFrame(predictions_with_scores)
            for K in KS:
                true_scores = one_query_gts["relevance"].tolist()[:K]
                pred_scores = one_query_preds_df["pred_relevance"].tolist()[:K]
                
                ndcg_score = calculate_ndcg(pred_scores, true_scores)
                performance[K][T].append(ndcg_score)
                if  ndcg_score  > 0.7:
                    print(qid, "NDCG", ndcg_score)
                    print("gt")
                    print(one_query_gts["relevance"].tolist()[:K])
                    print(one_query_gts["video_name"].tolist()[:K])
                    duration_series = one_query_gts.loc[:, "duration"]
                    normalized_times = one_query_gts.loc[:, ["start_time", "end_time"]].div(duration_series, axis=0)
                    print(normalized_times.head(K).to_dict("split")["data"])

                    print("prediction")
                    print(pred_scores)
                    print(one_query_preds_df["video_name"].tolist()[:K])
                    duration_series = one_query_preds_df.loc[:, "duration"]
                    normalized_times = one_query_preds_df.loc[:, ["start_time", "end_time"]].div(duration_series, axis=0)
                    print(normalized_times.head(K).to_dict("split")["data"])

                    print()
                    input()
    for K, vs in performance.items():
        for T, v in vs.items():
            performance_avg[K][T] = np.mean(v)
    return performance_avg


# KS = [10, 20, 40]
# TS = [0.3, 0.5, 0.7]

KS = [10]
TS = [0.3]

video2idx = load_json("./data/TVR_Ranking/video_corpus.json")
video2duration = {k: v[0] for k, v in video2idx.items()}
video2idx = {k: v[1] for k, v in video2idx.items()}
idx2video = {v: k for k, v in video2idx.items()}


gt_path = "data/TVR_Ranking/test.jsonl"
prediction_path = "./results/tvr_ranking/ReLoCLNet_top_40_20240326_163257/best_test_predictions.json"
pred_data= load_json(prediction_path)
gt_data = load_jsonl(gt_path)

average_ndcg = recall_iou_ndcg(gt_data, pred_data, video2idx, idx2video, video2duration, TS, KS)
print(prediction_path)
for K, vs in average_ndcg.items():
    for T, v in vs.items():
        print(f"VAL Top {K}, IoU={T}, NDCG: {v:.6f}")

    # test_gt_path = "data/TVR_Ranking_raw/test_raw.jsonl"
    # # test_result_path = "results/mul_train/TVR-Ranking-video_sub_tef-top_40-2024_03_05_20_17_23/test_best_TVR-Ranking_val_predictions_VCMR.json"
    # test_result_path = "results/mul_train/top1/best_test.json"
    # # test_result_path = "results/mul_train/tvr-video_sub_tef-top_10-2024_02_28_02_09_16/best_test.json"
    # average_ndcg = recall_iou_ndcg(test_result_path, test_gt_path, T, K)
    # print(test_result_path)
    # print(f"TEST Top {K}, IoU={T}, NDCG: {average_ndcg}")


