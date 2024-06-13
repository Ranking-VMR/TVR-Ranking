from modules.dataset_tvrr import TrainDataset, QueryEvalDataset, CorpusEvalDataset
import torch
from torch.utils.data import DataLoader
from utils.tensor_utils import pad_sequences_1d
import numpy as np

def collate_fn(batch, task):
    fixed_length = 128
    batch_data = dict()

    if task == "train":
        simis = [e["simi"] for e in batch]
        batch_data["simi"] =  torch.tensor(simis)
        
        query_feat_mask = pad_sequences_1d([e["query_feat"] for e in batch], dtype=torch.float32, fixed_length=None)
        batch_data["query_feat"] = query_feat_mask[0]
        batch_data["query_mask"] = query_feat_mask[1]    
        video_feat_mask = pad_sequences_1d([e["video_feat"] for e in batch], dtype=torch.float32, fixed_length=fixed_length)
        batch_data["video_feat"] = video_feat_mask[0]
        batch_data["video_mask"] = video_feat_mask[1]
        sub_feat_mask = pad_sequences_1d([e["sub_feat"] for e in batch], dtype=torch.float32, fixed_length=fixed_length)
        batch_data["sub_feat"] = sub_feat_mask[0]
        batch_data["sub_mask"] = sub_feat_mask[1]

        st_ed_indices = [e["st_ed_indices"] for e in batch]
        batch_data["st_ed_indices"] = torch.stack(st_ed_indices, dim=0)
        match_labels = np.zeros(shape=(len(st_ed_indices), fixed_length), dtype=np.int32)
        for idx, st_ed_index in enumerate(st_ed_indices):
            st_ed = st_ed_index.cpu().numpy()
            st, ed = st_ed[0], st_ed[1]
            match_labels[idx][st:(ed + 1)] = 1
        batch_data['match_labels'] = torch.tensor(match_labels, dtype=torch.long)
        
    if task == "corpus":
        video_feat_mask = pad_sequences_1d([e["video_feat"] for e in batch], dtype=torch.float32, fixed_length=fixed_length)
        batch_data["video_feat"] = video_feat_mask[0]
        batch_data["video_mask"] = video_feat_mask[1]
        sub_feat_mask = pad_sequences_1d([e["sub_feat"] for e in batch], dtype=torch.float32, fixed_length=fixed_length)
        batch_data["sub_feat"] = sub_feat_mask[0]
        batch_data["sub_mask"] = sub_feat_mask[1]
        
    if task == "eval":
        query_feat_mask = pad_sequences_1d([e["query_feat"] for e in batch], dtype=torch.float32, fixed_length=None)
        batch_data["query_feat"] = query_feat_mask[0]
        batch_data["query_mask"] = query_feat_mask[1]    

    return  batch_data




def prepare_dataset(opt):
    train_set = TrainDataset(
        data_path=opt.train_path,
        desc_bert_path=opt.desc_bert_path,
        sub_bert_path=opt.sub_bert_path,
        max_desc_len=opt.max_desc_l,
        max_ctx_len=opt.max_ctx_l,
        video_feat_path=opt.video_feat_path,
        clip_length=opt.clip_length,
        ctx_mode=opt.ctx_mode,
        normalize_vfeat=not opt.no_norm_vfeat,
        normalize_tfeat=not opt.no_norm_tfeat)
    train_loader = DataLoader(train_set, collate_fn=lambda batch: collate_fn(batch, task='train'), batch_size=opt.bsz, num_workers=opt.num_workers, shuffle=True, pin_memory=True)
    
    corpus_set = CorpusEvalDataset(corpus_path=opt.corpus_path, max_ctx_len=opt.max_ctx_l, sub_bert_path=opt.sub_bert_path, video_feat_path=opt.video_feat_path, ctx_mode=opt.ctx_mode)
    corpus_loader = DataLoader(corpus_set, collate_fn=lambda batch: collate_fn(batch, task='corpus'), batch_size=opt.bsz, num_workers=opt.num_workers, shuffle=False, pin_memory=True)

    val_set = QueryEvalDataset(data_path=opt.val_path, desc_bert_path=opt.desc_bert_path, max_desc_len=opt.max_desc_l)
    val_loader = DataLoader(val_set, collate_fn=lambda batch: collate_fn(batch, task='eval'), batch_size=opt.bsz_eval, num_workers=opt.num_workers, shuffle=False, pin_memory=True)
    test_set = QueryEvalDataset(data_path=opt.test_path, desc_bert_path=opt.desc_bert_path, max_desc_len=opt.max_desc_l)
    test_loader = DataLoader(test_set, collate_fn=lambda batch: collate_fn(batch, task='eval'), batch_size=opt.bsz_eval, num_workers=opt.num_workers, shuffle=False, pin_memory=True)
    
    val_gt = val_set.ground_truth
    test_gt = test_set.ground_truth
    corpus_video_list = corpus_set.corpus_video_list
    return train_loader, corpus_loader, corpus_video_list, val_loader, test_loader, val_gt, test_gt
