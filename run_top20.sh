python train.py \
    --results_path results/tvr_ranking \
    --train_path data/TVR_Ranking/train_top20.json \
    --val_path data/TVR_Ranking/val.json \
    --test_path data/TVR_Ranking/test.json \
    --corpus_path ./data/TVR_Ranking/video_corpus.json \
    --desc_bert_path /home/renjie.liang/datasets/TVR_Ranking/features/query_bert.h5 \
    --video_feat_path /home/share/czzhang/Dataset/TVR/TVR_feature/video_feature/tvr_i3d_rgb600_avg_cl-1.5.h5 \
    --sub_bert_path /home/share/czzhang/Dataset/TVR/TVR_feature/bert_feature/sub_query/tvr_sub_pretrained_w_sub_query_max_cl-1.5.h5\
    --n_epoch 100 \
    --eval_num_per_epoch 1 \
    --seed 2024 \
    --exp_id new_version

# qsub -I -l select=1:ngpus=1 -P gs_slab -q slab_gpu8
# cd /home/renjie.liang/11_TVR-Ranking/ReLoCLNet; conda activate py11; sh run_top20.sh 
    # --hard_negative_start_epoch 0 \
    # --no_norm_vfeat \
    # --use_hard_negative