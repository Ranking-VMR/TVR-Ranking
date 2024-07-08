import os, json
import torch
from tqdm import tqdm

from modules.dataset_init import prepare_dataset
from modules.infer_lib import grab_corpus_feature, eval_epoch 

from utils.basic_utils import get_logger
from utils.setup import set_seed, get_args
from utils.run_utils import prepare_optimizer, prepare_model, logger_ndcg_iou, resume_model

def main():
    opt = get_args()
    logger = get_logger(opt.results_path, opt.exp_id)
    set_seed(opt.seed)
    logger.info("Arguments:\n%s", json.dumps(vars(opt), indent=4))
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"device: {opt.device}")
    
    train_loader, corpus_loader, corpus_video_list, val_loader, test_loader, val_gt, test_gt = prepare_dataset(opt)

    model = prepare_model(opt, logger)
    # optimizer = prepare_optimizer(model, opt, len(train_loader) * opt.n_epoch)
    # start_epoch = 0
    # model, optimizer, start_epoch = resume_model(logger, opt, model, optimizer, start_epoch)
    model, _, _ = resume_model(logger, opt, model)

    model.eval()
    corpus_feature = grab_corpus_feature(model, corpus_loader, opt.device)
    val_ndcg_iou = eval_epoch(model, corpus_feature, val_loader, val_gt, opt, corpus_video_list)
    test_ndcg_iou = eval_epoch(model, corpus_feature, test_loader, test_gt, opt, corpus_video_list)

    logger_ndcg_iou(val_ndcg_iou, logger, "VAL")
    logger_ndcg_iou(test_ndcg_iou, logger, "TEST")

if __name__ == '__main__':
    main()
