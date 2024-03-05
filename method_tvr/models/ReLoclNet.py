from easydict import EasyDict as EDict



def set_ReLoCLNet_Config(opt):
    model_config = EDict(
        visual_input_size=opt.vid_feat_size,
        sub_input_size=opt.sub_feat_size,  # for both desc and subtitles
        query_input_size=opt.q_feat_size,  # for both desc and subtitles
        hidden_size=opt.hidden_size,  # hidden dimension
        conv_kernel_size=opt.conv_kernel_size,
        conv_stride=opt.conv_stride,
        max_ctx_l=opt.max_ctx_l,
        max_desc_l=opt.max_desc_l,
        input_drop=opt.input_drop,
        drop=opt.drop,
        n_heads=opt.n_heads,  # self-att heads
        initializer_range=opt.initializer_range,  # for linear layer
        ctx_mode=opt.ctx_mode,  # video, sub or video_sub
        margin=opt.margin,  # margin for ranking loss
        ranking_loss_type=opt.ranking_loss_type,  # loss type, 'hinge' or 'lse'
        lw_neg_q=opt.lw_neg_q,  # loss weight for neg. query and pos. context
        lw_neg_ctx=opt.lw_neg_ctx,  # loss weight for pos. query and neg. context
        lw_fcl=opt.lw_fcl,  # loss weight for frame level contrastive learning
        lw_vcl=opt.lw_vcl,  # loss weight for video level contrastive learning
        lw_st_ed=0,  # will be assigned dynamically at training time
        use_hard_negative=False,  # reset at each epoch
        hard_pool_size=opt.hard_pool_size)
    return model_config