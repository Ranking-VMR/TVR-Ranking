 CUDA_VISIBLE_DEVICES=0 \
   python method_tvr/train.py \
      --dset_name tvr \
      --eval_split_name val \
      --nms_thd -1 \
      --results_root results/single_train \
      --train_path data/tvr_train_release.jsonl \
      --video_duration_idx_path ./data/TVR_Ranking/video_corpus.json \
      --clip_length 1.5 \
      --vid_feat_size 1024 \
      --ctx_mode video_sub_tef \
      --no_norm_vfeat \
      --eval_path data/tvr_val_release.jsonl\
      --max_pred_l 16\
      --sub_feat_size 768\
      --vid_feat_path /storage_fast/rjliang/tvr/feat/video_feature/tvr_i3d_rgb600_avg_cl-1.5.h5 \
      --desc_bert_path /storage_fast/rjliang/tvr/feat/bert_feature/query_only/tvr_query_pretrained_w_query.h5 \
      --sub_bert_path /storage_fast/rjliang/tvr/feat/bert_feature/sub_query/tvr_sub_pretrained_w_sub_query_max_cl-1.5.h5\
      --eval_tasks_at_training VCMR \
      --exp_id demo

      # qsub -I -l select=1:ngpus=1 -P gs_slab -q slab_gpu8