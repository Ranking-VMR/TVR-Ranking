from utils.basic_utils import load_jsonl, save_jsonl, load_json
import pandas as pd
from tqdm import tqdm
import numpy as np

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



def recall_iou_ndcg(pred_data, result_path, gt_path, T, K):
    ###### ------------------ #######

    gt_data_tmp = pd.DataFrame(gt_data)
    gt_data_tmp["start_time"] = gt_data_tmp["timestamp"].apply(lambda x: x[0])
    gt_data_tmp["end_time"] = gt_data_tmp["timestamp"].apply(lambda x: x[1])
    gt_data_tmp["vid"] = gt_data_tmp["video_name"].map(video2idx) 
    gt_data = gt_data_tmp[["query_id", "query", "vid", "start_time", "end_time", "relevance"]]
    # gt_data["relevance"] = gt_data["relevance"] - 1
    # gt_data[gt_data["relevance"] <= 0] = 0
    
    ndcg_scores = []
    for i in tqdm(range(len(pred_data))):
        one_query_preds = pred_data[i]
        qid = one_query_preds["desc_id"]
        one_query_preds = one_query_preds["predictions"]
        one_query_preds_df = pd.DataFrame(one_query_preds, columns=["vid", "start_time", "end_time", "model_scores"])

        one_query_gts = gt_data[gt_data["query_id"]==qid].sort_values(by="relevance", ascending=False).reset_index(drop=True)

        one_query_gts_back = one_query_gts.copy()
        
        predictions_with_scores = []

        for index, pred in one_query_preds_df.iterrows():
            pred_vid, pred_st, pred_ed = pred["vid"], pred["start_time"], pred["end_time"]
            matched_rows = one_query_gts[one_query_gts["vid"] == pred_vid].reset_index(drop=True)
            
            if matched_rows.empty:
                pred["pred_relevance"] = 0
            else:
                matched_rows["iou"] = matched_rows.apply(lambda row: calculate_iou(pred_st, pred_ed, row["start_time"], row["end_time"]), axis=1)
                max_iou_idx = matched_rows["iou"].idxmax()
                max_iou_row = matched_rows.iloc[max_iou_idx]
                if max_iou_row["iou"] > T:
                    pred["pred_relevance"] = max_iou_row["relevance"]
                        
                    # Remove the matched ground truth row
                    one_query_gts = one_query_gts.drop(index=matched_rows.index[max_iou_idx]).reset_index(drop=True)
                else:
                    pred["pred_relevance"] = 0
            
            predictions_with_scores.append(pred)
            predictions_with_scores = predictions_with_scores
        one_query_preds_df = pd.DataFrame(predictions_with_scores)
        
        true_scores = one_query_gts_back["relevance"].tolist()[:K]
        pred_scores = one_query_preds_df["pred_relevance"].tolist()[:K]
        
        ndcg_score = calculate_ndcg(pred_scores, true_scores)
        ndcg_scores.append(ndcg_score)

    # Calculate the average NDCG
    average_ndcg = np.mean(ndcg_scores)
    return average_ndcg


for K in [10, 20, 40]:
    for T in [0.3, 0.5, 0.7]:
        val_gt_path = "data/TVR_Ranking/val.jsonl"
        # val_result_path = "results/mul_train/top40/val_best_TVR-Ranking_val_predictions_VCMR.json"
        val_result_path = "./results/tmp/best_val_predictions.json"
        # val_result_path = "results/mul_train/tvr-video_sub_tef-top_10-2024_02_28_02_09_16/best_val.json"
        
        pred_data_video2idx = load_json(val_result_path)
        video2idx = pred_data_video2idx["video2idx"]
        pred_data = pred_data["VCMR"]
        
        gt_data = load_jsonl(gt_path)
            
        average_ndcg = recall_iou_ndcg(val_result_path, val_gt_path, T, K)
        print(val_result_path)
        print(f"VAL Top {K}, IoU={T}, NDCG: {average_ndcg}")
        
        # test_gt_path = "data/TVR_Ranking_raw/test_raw.jsonl"
        # # test_result_path = "results/mul_train/TVR-Ranking-video_sub_tef-top_40-2024_03_05_20_17_23/test_best_TVR-Ranking_val_predictions_VCMR.json"
        # test_result_path = "results/mul_train/top1/best_test.json"
        # # test_result_path = "results/mul_train/tvr-video_sub_tef-top_10-2024_02_28_02_09_16/best_test.json"
        # average_ndcg = recall_iou_ndcg(test_result_path, test_gt_path, T, K)
        # print(test_result_path)
        # print(f"TEST Top {K}, IoU={T}, NDCG: {average_ndcg}")


