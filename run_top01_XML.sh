 CUDA_VISIBLE_DEVICES=2 \
   python method_tvr/train.py \
      --model_name XML \
      --dset_name TVR-Ranking \
      --eval_split_name val \
      --nms_thd -1 \
      --results_root results/mul_train \
      --train_path data/TVR_Ranking_1/train.jsonl \
      --val_path data/TVR_Ranking_1/val.jsonl \
      --test_path data/TVR_Ranking_1/test.jsonl \
      --clip_length 1.5 \
      --vid_feat_size 1024 \
      --ctx_mode video_sub_tef \
      --no_norm_vfeat \
      --max_pred_l 16\
      --sub_feat_size 768\
      --video_duration_idx_path ./data/common_data/video_corpus.json \
      --vid_feat_path /storage_fast/rjliang/tvr/feat/video_feature/tvr_i3d_rgb600_avg_cl-1.5.h5 \
      --desc_bert_path /storage/rjliang/TVRR/ReLoCLNet/data/common_data/query_bert.h5 \
      --sub_bert_path /storage_fast/rjliang/tvr/feat/bert_feature/sub_query/tvr_sub_pretrained_w_sub_query_max_cl-1.5.h5\
      --eval_tasks_at_training VCMR \
      --eval_num_per_epoch 1 \
      --n_epoch 300 \
      --exp_id top_1