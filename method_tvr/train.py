import os
import sys
sys.path.append("..")
sys.path.append(".")
import time
import json
import pprint
import random
import numpy as np
from easydict import EasyDict as EDict
from tqdm import tqdm, trange
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from method_tvr.config import BaseOptions
from method_tvr.model import ReLoCLNet
from method_tvr.init_dataset import get_train_data, get_eval_data
from method_tvr.start_end_dataset import start_end_collate, prepare_batch_inputs
from method_tvr.inference import eval_epoch, start_inference
from method_tvr.optimization import BertAdam
from utils.basic_utils import AverageMeter, get_logger
from utils.model_utils import count_parameters
from method_tvr.models.XML import XML
from utils.basic_utils import save_json, load_yaml
from torch.utils.data import DataLoader

def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)

def rm_key_from_odict(odict_obj, rm_suffix):
    """remove key entry from the OrderedDict"""
    return OrderedDict([(k, v) for k, v in odict_obj.items() if rm_suffix not in k])


def train(model, train_loader, val_data, test_data, context_data, opt, logger, writer):

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    num_train_optimization_steps = len(train_loader) * opt.n_epoch
    optimizer = BertAdam(optimizer_grouped_parameters, lr=opt.lr, weight_decay=opt.wd, warmup=opt.lr_warmup_proportion,
                         t_total=num_train_optimization_steps, schedule="warmup_linear")
    eval_step = len(train_loader) // opt.eval_num_per_epoch
    
    ########### ---------------------- ##################
    # start train
    best_val_ndcg = 0
    thresholds = [0.3, 0.5, 0.7]
    topks = [10, 20, 40]
    for epoch_i in range(0, opt.n_epoch):
        print(f"TRAIN EPOCH: {epoch_i}|{opt.n_epoch}")
        global_step = (epoch_i + 1) * len(train_loader)
        model.train()
        if opt.hard_negative_start_epoch != -1 and epoch_i >= opt.hard_negative_start_epoch:
            model.set_hard_negative(True, opt.hard_pool_size)
        if opt.train_span_start_epoch != -1 and epoch_i >= opt.train_span_start_epoch:
            if len(opt.device_ids) > 1:
                model.module.set_train_st_ed(opt.lw_st_ed)
            else:
                model.set_train_st_ed(opt.lw_st_ed)
                
        # init meters
        loss_meters = OrderedDict(loss_st_ed=AverageMeter(), loss_fcl=AverageMeter(), loss_vcl=AverageMeter(),
                                loss_neg_ctx=AverageMeter(), loss_neg_q=AverageMeter(),
                                loss_overall=AverageMeter())


        num_training_examples = len(train_loader)
        for batch_idx, batch in tqdm(enumerate(train_loader), desc="Training Iteration", total=num_training_examples):
            global_step = epoch_i * num_training_examples + batch_idx
            model.train()

            model_inputs = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory)
            loss, loss_dict = model(**model_inputs)
            optimizer.zero_grad()
            loss.backward()
            if opt.grad_clip != -1:
                nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()
            writer.add_scalar("Train/LR", float(optimizer.param_groups[0]["lr"]), global_step)
            for k, v in loss_dict.items():
                writer.add_scalar("Train/{}".format(k), v, global_step)
            for k, v in loss_dict.items():
                loss_meters[k].update(float(v))

                
            ###### ------------------- #############
            ### eval during training
            if global_step % eval_step == 0 and global_step != 0:
                model.eval()
                with torch.no_grad():
                    val_performance, val_predictions = eval_epoch(model, val_data, context_data, logger, opt,  max_after_nms=40, iou_thds=thresholds, topks=topks)
                    test_performance, test_predictions = eval_epoch(model, test_data, context_data, logger, opt,  max_after_nms=40, iou_thds=thresholds, topks=topks)
                    logger.info(f"EPOCH: {epoch_i}")
                    sum_ndcg = 0
                    line1 = ""
                    line2 = "VAL: "
                    line3 = "TEST: "
                    for K, vs in val_performance.items():
                        for T, v in vs.items():
                            sum_ndcg += v
                            line1 += f"NDCG@{K}, IoU={T}\t"
                            line2 += f" {v:.6f}"
                            
                    for K, vs in test_performance.items():
                        for T, v in vs.items():
                            line3 += f" {v:.6f}"
                logger.info(line1)
                logger.info(line2)
                logger.info(line3)
                
                
                if sum_ndcg > best_val_ndcg:
                    print("~"*40)
                    save_json(val_predictions, os.path.join(opt.results_dir, "best_val_predictions.json"))
                    save_json(test_predictions, os.path.join(opt.results_dir, "best_test_predictions.json"))
                    best_val_ndcg = sum_ndcg
                    logger.info("BEST " + line2)
                    logger.info("BEST " + line3)
                    checkpoint = {"model": model.state_dict(), "model_cfg": model.config, "epoch": epoch_i}
                    torch.save(checkpoint, opt.ckpt_filepath)
                    logger.info("save checkpoint: {}".format(opt.ckpt_filepath))
                    print("~"*40)

                logger.info("")
                        
                    # metrics = val_metrics_no_nms
                    # # early stop/ log / save model
                    # task_type = "VCMR"
                    # if task_type in metrics:
                    #     task_metrics = metrics[task_type]
                    #     for iou_thd in [0.5, 0.7]:
                    #         writer.add_scalars("Eval/{}-{}".format(task_type, iou_thd),
                    #                             {k: v for k, v in task_metrics.items() if str(iou_thd) in k},
                    #                             global_step)

                    # # use the most strict metric available
                    # stop_score = sum([metrics[opt.stop_task][e] for e in ["0.5-r1", "0.7-r1"]])
                    # if stop_score > prev_best_score:
                    #     prev_best_score = stop_score
                    #     checkpoint = {"model": model.state_dict(), "model_cfg": model.config, "epoch": epoch_i}
                    #     torch.save(checkpoint, opt.ckpt_filepath)
                        
                    #     # 
                    #     best_file_paths = [e.replace("latest", "best") for e in val_latest_file_paths]
                    #     for src, tgt in zip(val_latest_file_paths, best_file_paths):
                    #         os.renames(src, tgt)
                            
                    #     best_file_paths = [e.replace("latest", "best") for e in test_latest_file_paths]
                    #     for src, tgt in zip(test_latest_file_paths, best_file_paths):
                    #         os.renames(src, tgt)
                            
                            
                        # logger.info("The checkpoint file has been updated.")
                        
                        
                    # else:
                        # es_cnt += 1
                        # if opt.max_es_cnt != -1 and es_cnt > opt.max_es_cnt:  # early stop
                        #     with open(opt.train_log_filepath, "a") as f:
                        #         f.write("Early Stop at epoch {}".format(epoch_i))
                        #     logger.info("Early stop at {} with {} {}".format(
                        #         epoch_i, " ".join([opt.stop_task] + stop_metric_names), prev_best_score))
                        #     break


    writer.close()


def start_training():
    opt = BaseOptions().parse()
    set_seed(opt.seed)
    logger = get_logger(opt.results_dir, opt.model_name +"_"+ opt.exp_id)
    writer = SummaryWriter(opt.tensorboard_log_dir)

    train_dataset = get_train_data(opt, opt.train_path)
    train_data_loader = DataLoader(train_dataset, collate_fn=start_end_collate, batch_size=opt.bsz,
                            num_workers=opt.num_workers, shuffle=True, pin_memory=opt.pin_memory)
    
    context_data = get_eval_data(opt, opt.val_path, data_mode="context")
    val_data = get_eval_data(opt, opt.val_path, data_mode="query")
    test_data = get_eval_data(opt, opt.test_path, data_mode="query")
    
    model_name = eval(opt.model_name)
    model_config = EDict(load_yaml(opt.model_config_path))
    logger.info("{} config {}".format(model_name, model_config))
    model = model_name(model_config)
    count_parameters(model)
    
    
    # Prepare optimizer
    if opt.device.type == "cuda":
        if len(opt.device_ids) > 1:
            logger.info("Use multi GPU {}".format(opt.device_ids))
            model = torch.nn.DataParallel(model, device_ids=opt.device_ids)  # use multi GPU
        logger.info("CUDA enabled.")
        model.to(opt.device)

    train(model, train_data_loader, val_data, test_data, context_data, opt, logger, writer)
    return opt.results_dir, opt.eval_split_name, opt.eval_path, opt.debug


if __name__ == '__main__':
    model_dir, eval_split_name, eval_path, debug = start_training()
