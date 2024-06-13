from tqdm import tqdm, trange
import torch
import torch.nn.functional as F
import numpy as np

from utils.run_utils import topk_3d, generate_min_max_length_mask, extract_topk_elements
from modules.ndcg_iou import calculate_ndcg_iou

def grab_corpus_feature(model, corpus_loader, device):
    model.eval()
    all_video_feat, all_video_mask = [], []
    all_sub_feat, all_sub_mask = [], []
        
    for batch_input in tqdm(corpus_loader, desc="Compute Corpus Feature: ", total=len(corpus_loader)):
        batch_input = {k: v.to(device) for k, v in batch_input.items()}
        _video_feat, _sub_feat = model.encode_context(batch_input["video_feat"], batch_input["video_mask"],
                                                      batch_input["sub_feat"], batch_input["sub_mask"])
        
        all_video_feat.append(_video_feat.detach().cpu())
        all_video_mask.append(batch_input["video_mask"].detach().cpu())
        all_sub_feat.append(_sub_feat.detach().cpu())
        all_sub_mask.append(batch_input["sub_mask"].detach().cpu())
        
    all_video_feat = torch.cat(all_video_feat, dim=0)
    all_video_mask = torch.cat(all_video_mask, dim=0)
    all_sub_feat = torch.cat(all_sub_feat, dim=0)
    all_sub_mask = torch.cat(all_sub_mask, dim=0)

    return  { "all_video_feat": all_video_feat,
              "all_video_mask": all_video_mask,
              "all_sub_feat": all_sub_feat,
              "all_sub_mask": all_sub_mask}


def eval_epoch(model, corpus_feature, eval_loader, eval_gt, opt, corpus_video_list):
    topn_video = 100
    device = opt.device
    model.eval()
    all_video_feat = corpus_feature["all_video_feat"].to(device)
    all_video_mask = corpus_feature["all_video_mask"].to(device)
    all_sub_feat = corpus_feature["all_sub_feat"].to(device)
    all_sub_mask = corpus_feature["all_sub_mask"].to(device)
    all_query_score, all_end_prob, all_start_prob = [], [], []
    for batch_input in tqdm(eval_loader, desc="Compute Query Scores: ", total=len(eval_loader)):
        batch_input = {k: v.to(device) for k, v in batch_input.items()}
        query_scores, start_probs, end_probs = model.get_pred_from_raw_query(
            query_feat = batch_input["query_feat"],
            query_mask = batch_input["query_mask"], 
            video_feat = all_video_feat,
            video_mask = all_video_mask,
            sub_feat = all_sub_feat,
            sub_mask = all_sub_mask,
            cross=True)
        
        query_scores = torch.exp(opt.q2c_alpha * query_scores)
        start_probs = F.softmax(start_probs, dim=-1) 
        end_probs = F.softmax(end_probs, dim=-1)
        
        query_scores, start_probs, end_probs = extract_topk_elements(query_scores, start_probs, end_probs, topn_video)
        
        all_query_score.append(query_scores.detach().cpu())
        all_start_prob.append(start_probs.detach().cpu())
        all_end_prob.append(end_probs.detach().cpu())
        
    all_query_score = torch.cat(all_query_score, dim=0)
    all_start_prob = torch.cat(all_start_prob, dim=0)
    all_end_prob = torch.cat(all_end_prob, dim=0)
    
    # print("all_query_score", all_query_score.shape)
    # print("all_start_prob", all_start_prob.shape)
    # print("all_end_prob", all_end_prob.shape)
    
    average_ndcg = calculate_average_ndcg(all_start_prob, all_query_score, all_end_prob, corpus_video_list, eval_gt, opt)
    return average_ndcg

def calculate_average_ndcg(all_start_prob, all_query_score, all_end_prob, corpus_video_list, eval_gt, opt):
    topn_moment = max(opt.ndcg_topk)
    
    all_2D_map = torch.einsum("qvm,qv,qvn->qvmn", all_start_prob, all_query_score, all_end_prob)
    map_mask = generate_min_max_length_mask(all_2D_map.shape, min_l=opt.min_pred_l, max_l=opt.max_pred_l)
    all_2D_map = all_2D_map * map_mask
    
    all_pred, all_gt = [], []
    for i in trange(len(all_2D_map), desc="Collect Predictions: "):
        score_map = all_2D_map[i]
        top_score, top_idx = topk_3d(score_map, topn_moment)
        pred_videos = [corpus_video_list[i[0]] for i in top_idx]
        pre_start_time = [i[1].item() * opt.clip_length for i in top_idx]
        pre_end_time   = [i[2].item() * opt.clip_length for i in top_idx]
        
        pred_result = []
        for video_name, s, e, score, in zip(pred_videos, pre_start_time, pre_end_time, top_score):
            pred_result.append({
                "video_name": video_name,
                "timestamp": [s, e],
                "model_scores": score
            })
        all_pred.append(pred_result)
        gt_result = eval_gt[i]["relevant_moment"]
        all_gt.append(gt_result)
        
    average_ndcg = calculate_ndcg_iou(all_gt, all_pred, opt.iou_threshold, opt.ndcg_topk)
    return average_ndcg