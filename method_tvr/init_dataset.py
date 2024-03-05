from method_tvr.start_end_dataset import StartEndDataset, StartEndEvalDataset, start_end_collate
from torch.utils.data import DataLoader

def get_train_data_loader(opt, data_path):
    train_dataset = StartEndDataset(
        data_path=data_path,
        desc_bert_path_or_handler=opt.desc_bert_path,
        sub_bert_path_or_handler=opt.sub_bert_path,
        max_desc_len=opt.max_desc_l,
        max_ctx_len=opt.max_ctx_l,
        vid_feat_path_or_handler=opt.vid_feat_path,
        clip_length=opt.clip_length,
        ctx_mode=opt.ctx_mode,
        h5driver=opt.h5driver,
        data_ratio=opt.data_ratio,
        normalize_vfeat=not opt.no_norm_vfeat,
        normalize_tfeat=not opt.no_norm_tfeat)
    
    train_loader = DataLoader(train_dataset, collate_fn=start_end_collate, batch_size=opt.bsz,
                              num_workers=opt.num_workers, shuffle=True, pin_memory=opt.pin_memory)
    return train_loader

def get_eval_data(opt, data_path, data_mode):
    dataset = StartEndEvalDataset(
        data_path=data_path,
        desc_bert_path_or_handler=opt.desc_bert_path,
        sub_bert_path_or_handler=opt.sub_bert_path if "sub" in opt.ctx_mode else None,
        max_desc_len=opt.max_desc_l,
        max_ctx_len=opt.max_ctx_l,
        video_duration_idx_path=opt.video_duration_idx_path,
        vid_feat_path_or_handler=opt.vid_feat_path if "video" in opt.ctx_mode else None,
        clip_length=opt.clip_length,
        ctx_mode=opt.ctx_mode,
        data_mode=data_mode,
        h5driver=opt.h5driver,
        data_ratio=opt.data_ratio,
        normalize_vfeat=not opt.no_norm_vfeat,
        normalize_tfeat=not opt.no_norm_tfeat)
    return dataset
