import pandas as pd
from tqdm import tqdm, trange
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

def calculate_ndcg_iou(all_gt, all_pred, TS, KS):
    performance = defaultdict(lambda: defaultdict(list))
    performance_avg = defaultdict(lambda: defaultdict(float))
    
    for i in trange(len(all_pred), desc="Calculate NDCG,IoU: "):
        one_pred = all_pred[i]
        one_gt = all_gt[i]
        one_gt.sort(key=lambda x: x["relevance"], reverse=True)

        for T in TS:
            one_gt_drop = one_gt.copy()
            predictions_with_scores = []
            for pred in one_pred:
                pred_video_name, pred_time = pred["video_name"], pred["timestamp"]
                matched_rows = [gt for gt in one_gt_drop if gt["video_name"] == pred_video_name]
                
                if not matched_rows:
                    pred["pred_relevance"] = 0
                else:
                    ious = [calculate_iou(pred_time[0], pred_time[1], gt["timestamp"][0], gt["timestamp"][1]) for gt in matched_rows]
                    max_iou_idx = np.argmax(ious)
                    max_iou_row = matched_rows[max_iou_idx]
                    
                    if ious[max_iou_idx] > T:
                        pred["pred_relevance"] = max_iou_row["relevance"]
                        # Remove the matched ground truth row
                        original_idx = one_gt_drop.index(max_iou_row)
                        one_gt_drop.pop(original_idx)
                    else:
                        pred["pred_relevance"] = 0
                predictions_with_scores.append(pred)
            
            for K in KS:
                true_scores = [gt["relevance"] for gt in one_gt][:K]
                pred_scores = [pred["pred_relevance"] for pred in predictions_with_scores][:K]
                ndcg_score = calculate_ndcg(pred_scores, true_scores)
                performance[K][T].append(ndcg_score)

    for K, vs in performance.items():
        for T, v in vs.items():
            performance_avg[K][T] = np.mean(v)
    return performance_avg



# def calculate_ndcg_iou(gt_data, pred_data, TS, KS):
#     performance = defaultdict(lambda: defaultdict(list))
#     performance_avg = defaultdict(lambda: defaultdict(float))
    
#     for i in trange(len(pred_data), desc="Calculate NDCG,IoU: "):
#         one_pred = pred_data[i]
#         one_gt = gt_data[i]
        
#         one_pred_df = pd.DataFrame(one_pred, columns=["video_name", "start_time", "end_time", "model_scores"])
#         one_gt_df =  pd.DataFrame(one_gt, columns=["video_name", "timestamp", "relevance", "duration"])
#         one_gt_df["start_time"] = one_gt_df["timestamp"].apply(lambda x: x[0])
#         one_gt_df["end_time"] = one_gt_df["timestamp"].apply(lambda x: x[1])
#         one_gt_df = one_gt_df.sort_values(by="relevance", ascending=False).reset_index(drop=True)

#         for T in TS:
#             one_gt_drop = one_gt_df.copy()
#             predictions_with_scores = []
#             for index, pred in one_pred_df.iterrows():
#                 pred_video_name, pred_st, pred_ed = pred["video_name"], pred["start_time"], pred["end_time"]
#                 matched_rows = one_gt_drop[one_gt_drop["video_name"] == pred_video_name].reset_index(drop=True)
                
#                 if matched_rows.empty:
#                     pred["pred_relevance"] = 0
#                 else:
#                     matched_rows["iou"] = matched_rows.apply(lambda row: calculate_iou(pred_st, pred_ed, row["start_time"], row["end_time"]), axis=1)
#                     max_iou_idx = matched_rows["iou"].idxmax()
#                     max_iou_row = matched_rows.iloc[max_iou_idx]
                    
#                     if max_iou_row["iou"] > T:
#                         pred["pred_relevance"] = max_iou_row["relevance"]
#                         # Remove the matched ground truth row
#                         one_gt_drop = one_gt_drop.drop(index=matched_rows.index[max_iou_idx]).reset_index(drop=True)
#                     else:
#                         pred["pred_relevance"] = 0
                
#                 predictions_with_scores.append(pred)
#             predictions_with_scores = pd.DataFrame(predictions_with_scores)
#             for K in KS:
#                 true_scores = one_gt_df["relevance"].tolist()[:K]
#                 pred_scores = predictions_with_scores["pred_relevance"].tolist()[:K]
#                 ndcg_score = calculate_ndcg(pred_scores, true_scores)
#                 performance[K][T].append(ndcg_score)

#     for K, vs in performance.items():
#         for T, v in vs.items():
#             performance_avg[K][T] = np.mean(v)
#     return performance_avg
