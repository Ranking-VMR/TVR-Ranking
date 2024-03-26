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



def recall_iou_ndcg(pred_data, gt_data, video2idx, TS, KS):
    performance = defaultdict(lambda: defaultdict(list))
    performance_avg = defaultdict(lambda: defaultdict(float))
    gt_data_tmp = pd.DataFrame(gt_data)
    gt_data_tmp["start_time"] = gt_data_tmp["timestamp"].apply(lambda x: x[0])
    gt_data_tmp["end_time"] = gt_data_tmp["timestamp"].apply(lambda x: x[1])
    gt_data_tmp["vid"] = gt_data_tmp["video_name"].map(video2idx) 
    gt_data = gt_data_tmp[["query_id", "query", "vid", "start_time", "end_time", "relevance"]]
    
    for i in tqdm(range(len(pred_data))):
        one_query_preds = pred_data[i]
        qid = one_query_preds["desc_id"]
        one_query_preds = one_query_preds["predictions"]
        one_query_preds_df = pd.DataFrame(one_query_preds, columns=["vid", "start_time", "end_time", "model_scores"])

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
    
    for K, vs in performance.items():
        for T, v in vs.items():
            performance_avg[K][T] = np.mean(v)
    return performance_avg


if __name__ == "__main__":
    KS = [10, 20, 40]
    TS = [0.3, 0.5, 0.7]
    val_gt_path = "data/TVR_Ranking/val.jsonl"
    val_result_path = "./results/tmp/best_val_predictions.json"
    result_data= load_json(val_result_path)
    video2idx = result_data["video2idx"]
    pred_data = result_data["VCMR"]
    gt_data = load_jsonl(val_gt_path)

    average_ndcg = recall_iou_ndcg(gt_data, pred_data, video2idx, TS, KS)
    print(val_result_path)
    for K, vs in average_ndcg.items():
        for T, v in vs.items():
            print(f"VAL Top {K}, IoU={T}, NDCG: {v:.6f}")


