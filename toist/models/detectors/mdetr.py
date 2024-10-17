# Copyright (c) Pengfei Li. All Rights Reserved
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MDETR model and criterion classes.
"""

import torch
import torch.distributed
import torch.nn.functional as F
from torch import nn


from scipy.optimize import linear_sum_assignment


from ..detectors.kmeans import kmeans, kmeans_predict



class ClusterCriterion(nn.Module):
    def __init__(self, feature_dim, memory_size, cluster_num, task_count, args):
        super().__init__()
        self.args = args

        self.feature_dim = feature_dim
        self.memory_size = memory_size
        self.cluster_num = cluster_num
        self.task_count = task_count

        self.temp_feature_idx_list = [torch.zeros([args.train_batch_size, feature_dim+1]).cuda()
                                        for _ in range(torch.distributed.get_world_size())]

        feature_bank = torch.randn([task_count, memory_size, feature_dim])
        self.register_buffer("feature_bank", feature_bank)

        cluster_centers = torch.randn([task_count, cluster_num, feature_dim])
        self.register_buffer("cluster_centers", cluster_centers)

        update_count = torch.zeros([task_count])
        self.register_buffer("update_count", update_count)
        full_label = torch.zeros([task_count])
        self.register_buffer("full_label", full_label)

    def syn_memory(self):
        world_size = torch.distributed.get_world_size()

        torch.distributed.all_reduce(self.feature_bank)
        self.feature_bank /= world_size

        torch.distributed.all_reduce(self.cluster_centers)
        self.cluster_centers /= world_size

    def update_memory_queue(self, feature_idx_list):
        for feature_data in self.temp_feature_idx_list:
            feature_data *= 0
        torch.distributed.all_gather(self.temp_feature_idx_list, feature_idx_list)
        temp_feature_idx_tensor = torch.cat(self.temp_feature_idx_list, dim=0)

        new_feature_list = {}
        for i in range(self.task_count):
            new_feature_list[i] = []

        for i in range(len(temp_feature_idx_tensor)):
            new_task_idx = temp_feature_idx_tensor[i][-1]
            if new_task_idx == -1:  # empty
                continue
            new_feature = temp_feature_idx_tensor[i][:-1]
            new_feature_list[int(new_task_idx)].append(new_feature)

        for i in range(self.task_count):
            feature_length = len(new_feature_list[i])
            if feature_length == 0:
                continue
            feature_to_update = torch.stack(new_feature_list[i])

            if self.full_label[i] == 0:
                feature_to_remain = self.feature_bank[i][feature_length:].clone()
                self.feature_bank[i][:-feature_length] = feature_to_remain
                self.feature_bank[i][-feature_length:] = feature_to_update

                if self.update_count[i] > self.memory_size:
                    self.full_label[i] = 1
                self.update_count[i] += feature_length
            else:
                if self.args.fifo_memory:
                    feature_to_remain = self.feature_bank[i][feature_length:].clone()
                    self.feature_bank[i][:-feature_length] = feature_to_remain
                    self.feature_bank[i][-feature_length:] = feature_to_update
                else:   # replace nearest one
                    l1_dist = torch.cdist(feature_to_update, self.feature_bank[i], p=1).cpu()
                    indices = linear_sum_assignment(l1_dist)

                    for j in range(len(indices[1])):
                        self.feature_bank[i][indices[1][j]] = feature_to_update[indices[0][j]]

    def update_memory(self, memory_cache_noun, targets_noun, captions_noun):
        text_feature = torch.permute(memory_cache_noun['text_memory'], (1,0,2)) # BS x (num_tokens) x feature_dim
        normalized_text_emb = text_feature
        bs = normalized_text_emb.shape[0]

        tokenized = memory_cache_noun["tokenized"]

        # text token feature average
        token_feature_all_noun = torch.zeros([bs, normalized_text_emb.shape[-1]]).to(normalized_text_emb.device) # BS x hdim
        for i, tgt in enumerate(targets_noun):   # batchsize
            feature_i = []
            cur_tokens = [tgt["noun_tokens_positive"][j] for j in range(len(tgt["noun_tokens_positive"]))]
            for j, tok_list in enumerate(cur_tokens):   # bboxes in a sample
                pos_true = torch.zeros(normalized_text_emb.shape[1])
                for (beg, end) in tok_list:
                    beg_pos = tokenized.char_to_token(i, beg)
                    end_pos = tokenized.char_to_token(i, end - 1)
                    if beg_pos is None:
                        try:
                            beg_pos = tokenized.char_to_token(beg + 1)
                            if beg_pos is None:
                                beg_pos = tokenized.char_to_token(beg + 2)
                        except:
                            beg_pos = None
                    if end_pos is None:
                        try:
                            end_pos = tokenized.char_to_token(end - 2)
                            if end_pos is None:
                                end_pos = tokenized.char_to_token(end - 3)
                        except:
                            end_pos = None
                    if beg_pos is None or end_pos is None:
                        continue

                    assert beg_pos is not None and end_pos is not None

                    pos_true[beg_pos : end_pos + 1] = 1
                temp_token_feature = normalized_text_emb[i][pos_true.nonzero().reshape(-1)].mean(0)
                feature_i.append(temp_token_feature)
            if len(feature_i) > 0:
                feature_i = torch.stack(feature_i, dim=0)
                token_feature_all_noun[i] = feature_i.mean(0)

        all_mask = torch.zeros(bs, dtype=torch.bool)
        for i in range(len(targets_noun)):
            if len(targets_noun[i]['boxes']) == 0:
                all_mask[i] = True
        all_mask = all_mask.to(normalized_text_emb.device)

        task_idx_list = torch.ones(bs) * -1
        feature_list = torch.zeros([bs, normalized_text_emb.shape[-1]])
        task_idx_list = task_idx_list.to(normalized_text_emb.device)
        feature_list = feature_list.to(normalized_text_emb.device)
        for i in range(bs):
            if all_mask[i]:
                continue
            task_idx = int(targets_noun[i]['dataset_name'].split('_')[1]) - 1 # count from 0
            task_idx_list[i] = task_idx
            feature_list[i] = token_feature_all_noun[i].clone().detach()

        # update
        feature_idx_list = torch.cat([feature_list, task_idx_list.reshape(-1,1)], dim=-1)
        self.update_memory_queue(feature_idx_list)

        # reture
        memory_cache_noun['img_memory_mod'] = memory_cache_noun['img_memory'].clone()
        for i in range(bs):
            if all_mask[i]:
                continue

            task_idx = int(targets_noun[i]['dataset_name'].split('_')[1]) - 1 # count from 0
            cluster_center_choice, cluster_center_feature = self.memory_cluster(token_feature_all_noun[i].clone().detach(), task_idx)

            select_feature = self.cluster_centers[task_idx, cluster_center_choice]

            cur_tokens = [targets_noun[i]["noun_tokens_positive"][j] for j in range(len(targets_noun[i]["noun_tokens_positive"]))]
            pos_true = torch.zeros(normalized_text_emb.shape[1])
            for j, tok_list in enumerate(cur_tokens):   # bboxes in a sample
                for (beg, end) in tok_list:
                    beg_pos = tokenized.char_to_token(i, beg)
                    end_pos = tokenized.char_to_token(i, end - 1)
                    if beg_pos is None:
                        try:
                            beg_pos = tokenized.char_to_token(beg + 1)
                            if beg_pos is None:
                                beg_pos = tokenized.char_to_token(beg + 2)
                        except:
                            beg_pos = None
                    if end_pos is None:
                        try:
                            end_pos = tokenized.char_to_token(end - 2)
                            if end_pos is None:
                                end_pos = tokenized.char_to_token(end - 3)
                        except:
                            end_pos = None
                    if beg_pos is None or end_pos is None:
                        continue

                    assert beg_pos is not None and end_pos is not None

                    pos_true[beg_pos : end_pos + 1] = 1

            memory_cache_noun['img_memory_mod'][-len(memory_cache_noun['text_memory']):,i,:][pos_true.nonzero().reshape(-1)] = select_feature

        memory_cache_noun['full_label'] = self.full_label
        memory_cache_noun['update_count'] = self.update_count
        return memory_cache_noun

    def memory_cluster(self, feature_to_cluster, task_idx):
        memory_feature = self.feature_bank[task_idx]
        device = memory_feature.device

        cluster_ids_x, new_cluster_centers = kmeans(
            X=memory_feature, 
            init_cluster_centers=self.cluster_centers[task_idx].clone(),
            num_clusters=self.cluster_num, 
            distance='euclidean', 
            device=device,
            full_label=self.full_label[task_idx].item()
        )
        self.cluster_centers[task_idx] = new_cluster_centers

        cluster_ids_y = kmeans_predict(
            feature_to_cluster.reshape(1,-1), new_cluster_centers, 'euclidean', device=device, full_label=self.full_label[task_idx].item()
        )

        cluster_center_choice = cluster_ids_y[0]
        cluster_center_feature = new_cluster_centers[cluster_center_choice]

        return cluster_center_choice, cluster_center_feature

    def forward(self, memory_cache_sth, targets_sth, captions_sth):
        text_feature = torch.permute(memory_cache_sth['text_memory'], (1,0,2)) # BS x (num_tokens) x feature_dim

        normalized_text_emb = text_feature
        bs = normalized_text_emb.shape[0]

        tokenized = memory_cache_sth["tokenized"]

        memory_cache_sth['img_memory_mod'] = memory_cache_sth['img_memory'].clone()    # transformer decoder input

        loss_cluster_choice = torch.tensor(0.).to(normalized_text_emb.device)
        loss_cluster_feature = torch.tensor(0.).to(normalized_text_emb.device)
        loss_count = 0
        for i in range(bs):
            pos_true = torch.zeros(normalized_text_emb.shape[1])

            anno_name = 'something'
            begin_idx = captions_sth[i].find(anno_name)
            end_idx = begin_idx + len(anno_name)

            beg_pos = tokenized.char_to_token(i, begin_idx)
            end_pos = tokenized.char_to_token(i, end_idx - 1)
            pos_true[beg_pos : end_pos + 1] = 1

            temp_token_feature = normalized_text_emb[i][pos_true.nonzero().reshape(-1)].mean(0)

            task_idx = int(targets_sth[i]['dataset_name'].split('_')[1]) - 1 # count from 0
            cluster_center_choice, cluster_center_feature = self.memory_cluster(temp_token_feature.clone().detach(), task_idx)

            select_feature = self.cluster_centers[task_idx, cluster_center_choice]

            memory_cache_sth['img_memory_mod'][-len(memory_cache_sth['text_memory']):,i,:][pos_true.nonzero().reshape(-1)] = select_feature

            # loss
            loss_cluster_feature_i = F.mse_loss(temp_token_feature, \
                                        cluster_center_feature)
            loss_cluster_feature += loss_cluster_feature_i

            loss_count += 1

        if loss_count:
            loss_cluster_choice /= loss_count
            loss_cluster_feature /= loss_count

        return memory_cache_sth, {"loss_cluster_choice": loss_cluster_choice, "loss_cluster_feature": loss_cluster_feature}

    def infer_choice(self, memory_cache_sth, dataset_name_list, captions):
        text_feature = torch.permute(memory_cache_sth['text_memory'], (1,0,2)) # BS x (num_tokens) x feature_dim

        normalized_text_emb = text_feature
        bs = normalized_text_emb.shape[0]

        tokenized = memory_cache_sth["tokenized"]
        
        memory_cache_sth['img_memory_mod'] = memory_cache_sth['img_memory'].clone()    # transformer decoder input

        for i in range(bs):
            pos_true = torch.zeros(normalized_text_emb.shape[1])

            anno_name = 'something'
            begin_idx = captions[i].find(anno_name)
            end_idx = begin_idx + len(anno_name)

            beg_pos = tokenized.char_to_token(i, begin_idx)
            end_pos = tokenized.char_to_token(i, end_idx - 1)
            pos_true[beg_pos : end_pos + 1] = 1

            temp_token_feature = normalized_text_emb[i][pos_true.nonzero().reshape(-1)].mean(0)

            task_idx = int(dataset_name_list[i].split('_')[1]) - 1 # count from 0
            cluster_center_choice, cluster_center_feature = self.memory_cluster(temp_token_feature.clone().detach(), task_idx)

            select_feature = self.cluster_centers[task_idx, cluster_center_choice]

            memory_cache_sth['img_memory_mod'][-len(memory_cache_sth['text_memory']):,i,:][pos_true.nonzero().reshape(-1)] = select_feature

        return memory_cache_sth
    
class ClusterCriterion(nn.Module):
    def __init__(self,feature_dim,memory_size, cluster_num, task_count,batch_size):
        self.feature_dim = feature_dim
        self.memory_size = memory_size
        self.cluster_num = cluster_num
        self.task_count = task_count

        self.temp_feature_idx_list = torch.zeros([batch_size, feature_dim+1]).cuda()

        feature_bank = torch.randn([task_count, memory_size, feature_dim])
        self.register_buffer("feature_bank", feature_bank)

        cluster_centers = torch.randn([task_count, cluster_num, feature_dim])
        self.register_buffer("cluster_centers", cluster_centers)

    def update_memory_queue(self, feature_idx_list):
        for feature_data in feature_idx_list:
            task_idx=
            l1_dist = torch.cdist(feature_data, self.feature_bank[task_idx=], p=1).cpu()
            indices = linear_sum_assignment(l1_dist)
            self.feature_bank[i][indices[1][j]] = feature_to_update[indices[0][j]]

    def update_memory(self, memory_cache_noun, targets_noun, captions_noun):
        self.update_memory_queue(feature_idx_list)
        cluster_center_choice,_= self.memory_cluster()
        select_feature = self.cluster_centers[task_idx, cluster_center_choice]
