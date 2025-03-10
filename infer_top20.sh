python infer.py \
    --results_path      results/tvr_ranking \
    --train_path        data/TVR_Ranking/train_top20.json \
    --val_path          data/TVR_Ranking/val.json \
    --test_path         data/TVR_Ranking/test.json \
    --corpus_path       data/TVR_Ranking/video_corpus.json \
    --desc_bert_path    data/features/query_bert.h5 \
    --video_feat_path   data/features/tvr_i3d_rgb600_avg_cl-1.5.h5 \
    --sub_bert_path     data/features/tvr_sub_pretrained_w_sub_query_max_cl-1.5.h5 \
    --checkpoint        results/tvr_ranking/top20/best_model.pt \
    --exp_id            top20_infer

