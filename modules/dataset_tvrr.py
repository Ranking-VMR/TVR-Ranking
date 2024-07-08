import h5py
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.basic_utils import load_json, load_json, l2_normalize_np_array, uniform_feature_sampling
from utils.tensor_utils import pad_sequences_1d



class TrainDataset(Dataset):

    def __init__(self, data_path, desc_bert_path, sub_bert_path, max_desc_len,
                 max_ctx_len, video_feat_path, clip_length, ctx_mode, normalize_vfeat=True,
                 normalize_tfeat=True):

        self.annotations = self.expand_annotations(load_json(data_path))

        self.max_desc_len = max_desc_len
        self.max_ctx_len = max_ctx_len
        self.clip_length = clip_length

        # prepare desc data
        self.use_video = "video" in ctx_mode
        self.use_sub = "sub" in ctx_mode
        
        self.desc_bert_h5 = h5py.File(desc_bert_path, "r")
        if self.use_video:
            self.vid_feat_h5 = h5py.File(video_feat_path, "r")
        if self.use_sub:
            self.sub_bert_h5 = h5py.File(sub_bert_path, "r")

        self.normalize_vfeat = normalize_vfeat
        self.normalize_tfeat = normalize_tfeat

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        raw_data = self.annotations[index]
        # initialize with basic data
        # meta = dict(query_id=raw_data["query_id"], desc=raw_data["query"], vid_name=raw_data["video_name"],
        #             duration=raw_data["duration"], ts=raw_data["timestamp"], simi=raw_data["similarity"], caption=raw_data["caption"])
        
        '''
        return a dictionary:
        {
            "simi":
            "query_feat":
            "video_feat":
            "sub_feat":
            "st_ed_indices":
        }
        
        '''
        query_id=raw_data["query_id"]
        video_name=raw_data["video_name"]
        timestamp = raw_data["timestamp"]
        duration = raw_data["duration"]
        
        model_inputs = dict()
        model_inputs["simi"] = raw_data["similarity"]
        model_inputs["query_feat"] = self.get_query_feat_by_query_id(query_id)

        ctx_l = 0
        if self.use_video:
            video_feat = uniform_feature_sampling(self.vid_feat_h5[video_name][:], self.max_ctx_len)
            if self.normalize_vfeat:
                video_feat = l2_normalize_np_array(video_feat)
            model_inputs["video_feat"] = torch.from_numpy(video_feat)
            ctx_l = len(video_feat)
        else:
            model_inputs["video_feat"] = torch.zeros((2, 2))

        if self.use_sub:  # no need for ctx feature, as the features are already contextualized
            sub_feat = uniform_feature_sampling(self.sub_bert_h5[video_name][:], self.max_ctx_len)
            if self.normalize_tfeat:
                sub_feat = l2_normalize_np_array(sub_feat)
            model_inputs["sub_feat"] = torch.from_numpy(sub_feat)
            ctx_l = len(sub_feat)
        else:
            model_inputs["sub_feat"] = torch.zeros((2, 2))

        model_inputs["st_ed_indices"] = self.get_st_ed_label(timestamp, max_idx=ctx_l - 1)
        return model_inputs

    def get_st_ed_label(self, ts, max_idx):
        """
        Args:
            ts: [st (float), ed (float)] in seconds, ed > st
            max_idx: length of the video
        Returns:
            [st_idx, ed_idx]: int,
        Given ts = [3.2, 7.6], st_idx = 2, ed_idx = 6,
        clips should be indexed as [2: 6), the translated back ts should be [3:9].
        """
        st_idx = min(math.floor(ts[0] / self.clip_length), max_idx)
        ed_idx = min(math.ceil(ts[1] / self.clip_length), max_idx)  # -1
        return torch.tensor([st_idx, ed_idx], dtype=torch.long)

    def get_query_feat_by_query_id(self, query_id):
        query_feat = self.desc_bert_h5[str(query_id)][:self.max_desc_len]
        if self.normalize_tfeat:
            query_feat = l2_normalize_np_array(query_feat)
        return torch.from_numpy(query_feat)

    def expand_annotations(self, annotations):
        new_annotations = []
        for i in annotations:
            query = i["query"]
            query_id = i["query_id"]
            for moment in  i["relevant_moment"]:
                moment.update({'query': query, 'query_id': query_id})
                new_annotations.append(moment)
        return new_annotations


class QueryEvalDataset(Dataset):
    def __init__(self, data_path, desc_bert_path, max_desc_len, normalize_tfeat=True):

        self.max_desc_len = max_desc_len
        self.desc_bert_h5 = h5py.File(desc_bert_path, "r")

        self.annotations = load_json(data_path)
        self.normalize_tfeat = normalize_tfeat
        self.ground_truth = self.get_relevant_moment_gt()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        raw_data = self.annotations[index]
        query_id = raw_data["query_id"]
        query = raw_data["query"]
        model_inputs =  {"query_id": query_id,  
                         "query_feat": self.get_query_feat_by_query_id(query_id)}
        return model_inputs

    def get_query_feat_by_query_id(self, query_id):
        query_feat = self.desc_bert_h5[str(query_id)][:self.max_desc_len]
        if self.normalize_tfeat:
            query_feat = l2_normalize_np_array(query_feat)
        return torch.from_numpy(query_feat)

    def get_relevant_moment_gt(self):
        gt_all = {}
        for data in self.annotations:
            gt_all[data["query_id"]] = data["relevant_moment"]
            # gt_all.append({
            #     "query_id": data["query_id"],
            #     "relevant_moment": data["relevant_moment"]})
        return gt_all

    def get_st_ed_label(self, ts, max_idx):
        st_idx = min(math.floor(ts[0] / self.clip_length), max_idx)
        ed_idx = min(math.ceil(ts[1] / self.clip_length), max_idx)
        return torch.tensor([st_idx, ed_idx], dtype=torch.long)

    
class CorpusEvalDataset(Dataset):
    def __init__(self, corpus_path, max_ctx_len, sub_bert_path, video_feat_path, ctx_mode,
                 normalize_vfeat=True, normalize_tfeat=True):
        self.normalize_vfeat = normalize_vfeat
        self.normalize_tfeat = normalize_tfeat

        self.max_ctx_len = max_ctx_len
        
        video_data = load_json(corpus_path)
        self.video_data = [{"vid_name": k, "duration": v} for k, v in video_data.items()]
        self.corpus_video_list = list(video_data.keys())


        self.use_video = "video" in ctx_mode
        self.use_sub = "sub" in ctx_mode

        if self.use_video:
            self.vid_feat_h5 = h5py.File(video_feat_path, "r")
        if self.use_sub:
            self.sub_bert_h5 = h5py.File(sub_bert_path, "r")

    def __len__(self):
        return len(self.video_data)

    def __getitem__(self, index):
        """No need to batch, since it has already been batched here"""
        raw_data = self.video_data[index]
        # initialize with basic data
        duration = raw_data["duration"]
        video_name = raw_data["vid_name"]
        meta = dict(vid_name=raw_data["vid_name"], duration=raw_data["duration"])
        model_inputs = dict()

        if self.use_video:
            video_feat = uniform_feature_sampling(self.vid_feat_h5[meta["vid_name"]][:], self.max_ctx_len)
            if self.normalize_vfeat:
                video_feat = l2_normalize_np_array(video_feat)
            model_inputs["video_feat"] = torch.from_numpy(video_feat)
        else:
            model_inputs["video_feat"] = torch.zeros((2, 2))

        if self.use_sub:  # no need for ctx feature, as the features are already contextualized
            sub_feat = uniform_feature_sampling(self.sub_bert_h5[meta["vid_name"]][:], self.max_ctx_len)
            if self.normalize_tfeat:
                sub_feat = l2_normalize_np_array(sub_feat)
            model_inputs["sub_feat"] = torch.from_numpy(sub_feat)
        else:
            model_inputs["sub_feat"] = torch.zeros((2, 2))
        return model_inputs
