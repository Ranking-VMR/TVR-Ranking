python train.py \
    --results_path      results/tvr_ranking \
    --train_path        data/TVR_Ranking/train_top40.json \
    --val_path          data/TVR_Ranking/val.json \
    --test_path         data/TVR_Ranking/test.json \
    --corpus_path       data/TVR_Ranking/video_corpus.json \
    --desc_bert_path    data/features/query_bert.h5 \
    --video_feat_path   data/features/tvr_i3d_rgb600_avg_cl-1.5.h5 \
    --sub_bert_path     data/features/tvr_sub_pretrained_w_sub_query_max_cl-1.5.h5 \
    --n_epoch               100 \
    --eval_num_per_epoch    2 \
    --seed                  2024 \
    --bsz                   512 \
    --exp_id                top40

# qsub -I -l select=1:ngpus=1 -P gs_slab -q gpu8
# cd /home/renjie.liang/11_TVR-Ranking/ReLoCLNet; conda activate py11; sh run_top40.sh 