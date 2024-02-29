    true_scores = one_query_gts["match_score"].tolist()[:len(predictions_with_scores)]  # Ensure length match
    pred_scores = one_query_preds_df["pred_match_score"].tolist()
    ndcg_score = calculate_ndcg(pred_scores, true_scores)
    Average NDCG: 0.2293213


gt_data["match_score"] = gt_data["match_score"].replace([1, 2], 0)
    true_scores = one_query_gts["match_score"].tolist()[:len(predictions_with_scores)]  # Ensure length match
    pred_scores = one_query_preds_df["pred_match_score"].tolist()
    ndcg_score = calculate_ndcg(pred_scores, true_scores)
    Average NDCG:  0.2065619044385435


    true_scores = one_query_gts["match_score"].tolist() # Ensure length match
    pred_scores = one_query_preds_df["pred_match_score"].tolist()[:len(true_scores)] 
    ndcg_score = calculate_ndcg(pred_scores, true_scores)
    Average NDCG: 0.1285646230537192


gt_data["match_score"] = gt_data["match_score"].replace([1, 2], 0)
    true_scores = one_query_gts["match_score"].tolist() # Ensure length match
    pred_scores = one_query_preds_df["pred_match_score"].tolist()[:len(true_scores)] 
    ndcg_score = calculate_ndcg(pred_scores, true_scores)
    Average NDCG:  0.12049033