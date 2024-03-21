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
from method_tvr.init_dataset import get_train_data_loader, get_eval_data
from method_tvr.start_end_dataset import start_end_collate, prepare_batch_inputs
from method_tvr.inference import eval_epoch, start_inference
from method_tvr.optimization import BertAdam
from utils.basic_utils import AverageMeter, get_logger
from utils.model_utils import count_parameters
from method_tvr.models.ReLoclNet import set_ReLoCLNet_Config
from method_tvr.models.XML import set_XML_Config, XML



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
    prev_best_score = 0.
    eval_step = len(train_loader) // opt.eval_num_per_epoch
    
    eval_tasks_at_training = opt.eval_tasks_at_training  # VR is computed along with VCMR
    save_submission_filename = "latest_{}_{}_predictions_{}.json".format(opt.dset_name, opt.eval_split_name,
                                                                         "_".join(eval_tasks_at_training))
    
    ########### ---------------------- ##################
    # start train
    for epoch_i in trange(0, opt.n_epoch, desc="Epoch"):
        global_step = (epoch_i + 1) * len(train_loader)
        
        training=True
        with torch.autograd.detect_anomaly():
            model.train(mode=training)
            if opt.hard_negative_start_epoch != -1 and epoch_i >= opt.hard_negative_start_epoch:
                model.set_hard_negative(True, opt.hard_pool_size)
            if opt.train_span_start_epoch != -1 and epoch_i >= opt.train_span_start_epoch:
                model.set_train_st_ed(opt.lw_st_ed)

            # init meters
            loss_meters = OrderedDict(loss_st_ed=AverageMeter(), loss_fcl=AverageMeter(), loss_vcl=AverageMeter(),
                                    loss_neg_ctx=AverageMeter(), loss_neg_q=AverageMeter(),
                                    loss_overall=AverageMeter())


            num_training_examples = len(train_loader)
            for batch_idx, batch in tqdm(enumerate(train_loader), desc="Training Iteration", total=num_training_examples):
                global_step = epoch_i * num_training_examples + batch_idx

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
                # print(eval_tasks_at_training)
                if global_step % eval_step == 0:# and global_step != 0:
                    with torch.no_grad():
                        NDCG_IOU_top10, NDCG_IOU_top20, NDCG_IOU_top40, val_prediction_paths = eval_epoch(model, val_data, context_data, logger, opt, "val_"+save_submission_filename, tasks=eval_tasks_at_training, max_after_nms=100)
                        test_metrics_no_nms, test_metrics_nms, test_latest_file_paths = eval_epoch(model, test_data, context_data, logger, opt, "test_"+save_submission_filename, tasks=eval_tasks_at_training, max_after_nms=100)
                    
                    
                    
                    logger.info(f"EPOCH: {epoch_i}")
                    logger.info(f"VAL: NDCG@10_IOU: {json.dumps(NDCG_IOU_top10)}")
                    logger.info(f"VAL: NDCG@20_IOU: {json.dumps(NDCG_IOU_top40)}")
                    logger.info(f"VAL: NDCG@40_IOU: {json.dumps(val_metrics_no_nms)}")
                    logger.info(f"TEST: metrics_no_nms: {json.dumps(test_metrics_no_nms)}")
                    logger.info(f"TEST: metrics_nms: {json.dumps(test_metrics_nms)}")
                    
                    
                    metrics = val_metrics_no_nms
                    # early stop/ log / save model
                    task_type = "VCMR"
                    if task_type in metrics:
                        task_metrics = metrics[task_type]
                        for iou_thd in [0.5, 0.7]:
                            writer.add_scalars("Eval/{}-{}".format(task_type, iou_thd),
                                                {k: v for k, v in task_metrics.items() if str(iou_thd) in k},
                                                global_step)

                    # use the most strict metric available
                    stop_score = sum([metrics[opt.stop_task][e] for e in ["0.5-r1", "0.7-r1"]])
                    if stop_score > prev_best_score:
                        prev_best_score = stop_score
                        checkpoint = {"model": model.state_dict(), "model_cfg": model.config, "epoch": epoch_i}
                        torch.save(checkpoint, opt.ckpt_filepath)
                        
                        # 
                        best_file_paths = [e.replace("latest", "best") for e in val_latest_file_paths]
                        for src, tgt in zip(val_latest_file_paths, best_file_paths):
                            os.renames(src, tgt)
                            
                        best_file_paths = [e.replace("latest", "best") for e in test_latest_file_paths]
                        for src, tgt in zip(test_latest_file_paths, best_file_paths):
                            os.renames(src, tgt)
                            
                            
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

    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Metrics] {eval_metrics_str}\n"

    train_data_loader = get_train_data_loader(opt, opt.train_path)
    context_data = get_eval_data(opt, opt.val_path, data_mode="context")
    val_data = get_eval_data(opt, opt.val_path, data_mode="query")
    test_data = get_eval_data(opt, opt.test_path, data_mode="query")
    
    model_name = eval(opt.model_name)
    model_config = eval("set_"+opt.model_name+"_Config")(opt)
    logger.info("{} config {}".format(model_name, model_config))
    model = model_name(model_config)
    count_parameters(model)
    
    # model_config = EDict(
    #     visual_input_size=opt.vid_feat_size,
    #     sub_input_size=opt.sub_feat_size,  # for both desc and subtitles
    #     query_input_size=opt.q_feat_size,  # for both desc and subtitles
    #     hidden_size=opt.hidden_size,  # hidden dimension
    #     conv_kernel_size=opt.conv_kernel_size,
    #     conv_stride=opt.conv_stride,
    #     max_ctx_l=opt.max_ctx_l,
    #     max_desc_l=opt.max_desc_l,
    #     input_drop=opt.input_drop,
    #     drop=opt.drop,
    #     n_heads=opt.n_heads,  # self-att heads
    #     initializer_range=opt.initializer_range,  # for linear layer
    #     ctx_mode=opt.ctx_mode,  # video, sub or video_sub
    #     margin=opt.margin,  # margin for ranking loss
    #     ranking_loss_type=opt.ranking_loss_type,  # loss type, 'hinge' or 'lse'
    #     lw_neg_q=opt.lw_neg_q,  # loss weight for neg. query and pos. context
    #     lw_neg_ctx=opt.lw_neg_ctx,  # loss weight for pos. query and neg. context
    #     lw_fcl=opt.lw_fcl,  # loss weight for frame level contrastive learning
    #     lw_vcl=opt.lw_vcl,  # loss weight for video level contrastive learning
    #     lw_st_ed=0,  # will be assigned dynamically at training time
    #     use_hard_negative=False,  # reset at each epoch
    #     hard_pool_size=opt.hard_pool_size)
    # logger.info("model_config {}".format(model_config))
    # model = ReLoCLNet(model_config)
    # count_parameters(model)
    
    # Prepare optimizer
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
        if len(opt.device_ids) > 1:
            logger.info("Use multi GPU", opt.device_ids)
            model = torch.nn.DataParallel(model, device_ids=opt.device_ids)  # use multi GPU

    train(model, train_data_loader, val_data, test_data, context_data, opt, logger, writer)
    return opt.results_dir, opt.eval_split_name, opt.eval_path, opt.debug


if __name__ == '__main__':
    model_dir, eval_split_name, eval_path, debug = start_training()
