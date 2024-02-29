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


result_path = "results/mul_train/tvr-video_sub_tef-top_10-2024_02_28_02_09_16/best_tvr_val_predictions_VCMR.json"
# result_path = "results/single_train/tvr-video_sub_tef-single-2024_02_27_12_19_42/best_tvr_val_predictions_VCMR.json"
pred_data = load_json(result_path)
video2idx = pred_data["video2idx"]

gt_data = load_jsonl("data/TVR_Ranking_raw/test_tvrr.jsonl")
gt_data_tmp = pd.DataFrame(gt_data)
gt_data_tmp["start_time"] = gt_data_tmp["timestamp"].apply(lambda x: x[0])
gt_data_tmp["end_time"] = gt_data_tmp["timestamp"].apply(lambda x: x[1])
gt_data_tmp["vid"] = gt_data_tmp["vname"].map(video2idx) 
gt_data = gt_data_tmp.rename(columns={"final_match_score": "match_score"})
gt_data = gt_data[["qid", "query", "vid", "start_time", "end_time", "match_score"]]
# gt_data["match_score"] = gt_data["match_score"].replace([1, 2], 0)



###### ------------------ #######
ndcg_scores = []
for i in tqdm(range(len(pred_data["VCMR"]))):
    one_query_preds = pred_data["VCMR"][i]
    qid = one_query_preds["desc_id"]
    one_query_preds = one_query_preds["predictions"]
    one_query_preds_df = pd.DataFrame(one_query_preds, columns=["vid", "start_time", "end_time", "model_scores"])

    one_query_gts = gt_data[gt_data["qid"]==qid].sort_values(by="match_score", ascending=False).reset_index(drop=True)

    predictions_with_scores = []

    for index, pred in one_query_preds_df.iterrows():
        pred_vid, pred_st, pred_ed = pred["vid"], pred["start_time"], pred["end_time"]
        matched_rows = one_query_gts[one_query_gts["vid"] == pred_vid].reset_index(drop=True)
        
        if matched_rows.empty:
            pred["pred_match_score"] = 0
        else:
            matched_rows["iou"] = matched_rows.apply(lambda row: calculate_iou(pred_st, pred_ed, row["start_time"], row["end_time"]), axis=1)
            max_iou_idx = matched_rows["iou"].idxmax()
            max_iou_row = matched_rows.iloc[max_iou_idx]
            
            if max_iou_row["iou"] > 0:
                pred["pred_match_score"] = max_iou_row["match_score"]
                # Remove the matched ground truth row
                one_query_gts = one_query_gts.drop(index=matched_rows.index[max_iou_idx]).reset_index(drop=True)
            else:
                pred["pred_match_score"] = 0
        
        predictions_with_scores.append(pred)

    one_query_preds_df = pd.DataFrame(predictions_with_scores)

    # Assuming true_scores should come from the ground truth data filtered by qid, sorted by match_score
    # This may need adjustment based on how the ground truth data is structured and what the true scores represent
    true_scores = one_query_gts["match_score"].tolist()  # Ensure length match
    pred_scores = one_query_preds_df["pred_match_score"].tolist()[:len(true_scores)]

    ndcg_score = calculate_ndcg(pred_scores, true_scores)
    ndcg_scores.append(ndcg_score)

# Calculate the average NDCG
average_ndcg = np.mean(ndcg_scores)
print(f"Average NDCG: {average_ndcg}")
