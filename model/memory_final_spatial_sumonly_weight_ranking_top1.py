import torch
import torch.autograd as ag
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import functools
import random
from torch.nn import functional as F

def random_uniform(shape, low, high, cuda):
    x = torch.rand(*shape)
    result_cpu = (high - low) * x + low
    if cuda:
        return result_cpu.cuda()
    else:
        return result_cpu
    
def distance(a, b):
    return torch.sqrt(((a - b) ** 2).sum()).unsqueeze(0)

def distance_batch(a, b):
    bs, _ = a.shape
    result = distance(a[0], b)
    for i in range(bs-1):
        result = torch.cat((result, distance(a[i], b)), 0)
        
    return result

def multiply(x): #to flatten matrix into a vector 
    return functools.reduce(lambda x,y: x*y, x, 1)

def flatten(x):
    """ Flatten matrix into a vector """
    count = multiply(x.size())
    return x.resize_(count)

def index(batch_size, x):
    idx = torch.arange(0, batch_size).long() 
    idx = torch.unsqueeze(idx, -1)
    return torch.cat((idx, x), dim=1)

def MemoryLoss(memory):

    m, d = memory.size()
    memory_t = torch.t(memory)
    similarity = (torch.matmul(memory, memory_t))/2 + 1/2 # 30X30
    identity_mask = torch.eye(m).cuda()
    sim = torch.abs(similarity - identity_mask)
    
    return torch.sum(sim)/(m*(m-1))


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Memory(nn.Module):
    def __init__(self, memory_size, feature_dim, key_dim, temp_update, temp_gather):
        super(Memory, self).__init__()
        # Constants（保持你的接口与成员不变）
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.key_dim = key_dim
        self.temp_update = temp_update  # 用于控制动量（见 update）
        self.temp_gather = temp_gather  # 用于相似度缩放（见 _scaled_similarity）

    # ---------------------------
    # 工具：数值稳定的 scaled dot-product 相似度
    # ---------------------------
    @torch.no_grad()
    def _scaled_similarity(self, A, B, temperature=1.0):
        # A: (..., d), B: (m, d) -> S: (..., m)
        d = A.size(-1)
        scale = math.sqrt(d) * max(temperature, 1e-6)
        return torch.matmul(A, B.t()) / scale

    # ---------------------------
    # 维持原接口：hard_neg_mem / random_pick_memory
    # 仅做小幅鲁棒化（device-aware）
    # ---------------------------
    def hard_neg_mem(self, mem, i):
        # 按原语义：与 self.keys_var 做相似度，mask 第 i 列取 top1
        similarity = torch.matmul(mem, torch.t(self.keys_var))
        similarity[:, i] = -1
        _, max_idx = torch.topk(similarity, 1, dim=1)
        return self.keys_var[max_idx]

    def random_pick_memory(self, mem, max_indices):
        m, d = mem.size()
        device = max_indices.device
        output = []
        for i in range(m):
            flattened_indices = (max_indices == i).nonzero()
            a, _ = flattened_indices.size()
            if a != 0:
                # 用 torch.randint 代替 numpy 以避免 CPU↔GPU 往返
                number = torch.randint(low=0, high=a, size=(1,), device=device)
                output.append(flattened_indices[number, 0])
            else:
                output.append(torch.tensor(-1, device=device))
        return torch.stack(output)

    # ---------------------------
    # get_update_query：向量化重构（去掉 for 循环）
    # ---------------------------
    @torch.no_grad()
    def get_update_query(self, mem, max_indices, update_indices, score, query, train):
        """
        原逻辑：对每个 memory 槽 i，把归属到该槽的样本 query 按权重 score[:, i] 做加权求和。
        这里用 index_add_ 在 GPU 上一次性完成聚合，速度更快、数值更稳。
        """
        # mem: (m, d) | max_indices: (N,1) | score: (N,m) | query: (N,d)
        N, d = query.size()
        m = mem.size(0)
        device = query.device
        g = max_indices.view(-1)                         # (N,)
        w = score.gather(1, g.view(-1, 1)).view(-1)     # (N,)

        # 加权样本（逐样本）
        q_w = query * w.view(-1, 1)                      # (N, d)

        # 归约到各 memory 槽：accum[i] = sum_{n:g_n=i} q_w[n]
        query_update = torch.zeros(m, d, device=device, dtype=query.dtype)
        query_update.index_add_(0, g, q_w)               # 关键加速点

        return query_update

    # ---------------------------
    # 相似度得分：引入 scaled dot-product
    # ---------------------------
    def get_score(self, mem, query):
        bs, h, w, d = query.size()
        m = mem.size(0)
        logits = self._scaled_similarity(query.reshape(-1, d), mem, temperature=self.temp_gather)  # (N, m)
        score_query = F.softmax(logits, dim=0)
        score_memory = F.softmax(logits, dim=1)
        return score_query, score_memory

    # ---------------------------
    # forward：保持你的入参/出参不变
    # ---------------------------
    def forward(self, query, keys, train=True):
        batch_size, dims, h, w = query.size()  # b x d x h x w
        query = F.normalize(query, dim=1)
        query = query.permute(0, 2, 3, 1).contiguous()  # b x h x w x d

        if train:
            separateness_loss, compactness_loss = self.gather_loss(query, keys, train)
            updated_query, softmax_score_query, softmax_score_memory = self.read(query, keys)
            updated_memory = self.update(query, keys, train)
            return (updated_query, updated_memory,
                    softmax_score_query, softmax_score_memory,
                    separateness_loss, compactness_loss)
        else:
            compactness_loss, query_re, top1_keys, keys_ind = self.gather_loss(query, keys, train)
            updated_query, softmax_score_query, softmax_score_memory = self.read(query, keys)
            updated_memory = keys
            return (updated_query, updated_memory,
                    softmax_score_query, softmax_score_memory,
                    query_re, top1_keys, keys_ind, compactness_loss)

    # ---------------------------
    # update：动量式融合（MoCo风格），不改返回
    # ---------------------------
    def update(self, query, keys, train):
        b, h, w, d = query.size()
        score_q, score_m = self.get_score(keys, query)
        query_reshape = query.contiguous().view(b * h * w, d)

        _, gathering_indices = torch.topk(score_m, 1, dim=1)   # (N,1)
        _, updating_indices  = torch.topk(score_q, 1, dim=0)   # 与原签名保持

        query_update = self.get_update_query(
            keys, gathering_indices, updating_indices, score_q, query_reshape, train
        )

        # 将 temp_update 作为动量的控制旋钮：mmt ∈ [0, 0.999]
        mmt = 1.0 / max(self.temp_update, 1e-6)
        mmt = max(min(mmt, 0.999), 0.0)

        updated_memory = keys * mmt + query_update * (1.0 - mmt)
        updated_memory = F.normalize(updated_memory, dim=1)
        return updated_memory.detach()

    # ---------------------------
    # 损失：接口与语义保持不变
    # ---------------------------
    def pointwise_gather_loss(self, query_reshape, keys, gathering_indices, train):
        n, dims = query_reshape.size()
        loss_mse = torch.nn.MSELoss(reduction='none')
        return loss_mse(query_reshape, keys[gathering_indices].squeeze(1).detach())

    def gather_loss(self, query, keys, train):
        b, h, w, d = query.size()
        if train:
            loss_tri = torch.nn.TripletMarginLoss(margin=1.0)
            loss_mse = torch.nn.MSELoss()
            score_q, score_m = self.get_score(keys, query)
            q = query.contiguous().view(b * h * w, d)
            _, idx = torch.topk(score_m, 2, dim=1)
            pos = keys[idx[:, 0]]
            neg = keys[idx[:, 1]]
            top1_loss = loss_mse(q, pos.detach())
            gathering_loss = loss_tri(q, pos.detach(), neg.detach())
            return gathering_loss, top1_loss
        else:
            loss_mse = torch.nn.MSELoss()
            score_q, score_m = self.get_score(keys, query)
            q = query.contiguous().view(b * h * w, d)
            _, idx = torch.topk(score_m, 1, dim=1)
            gathering_loss = loss_mse(q, keys[idx].squeeze(1).detach())
            return gathering_loss, q, keys[idx].squeeze(1).detach(), idx[:, 0]

    # ---------------------------
    # read：保持原逻辑，仅复用 get_score（已含缩放）
    # ---------------------------
    def read(self, query, updated_memory):
        b, h, w, d = query.size()
        score_q, score_m = self.get_score(updated_memory, query)
        q = query.contiguous().view(b * h * w, d)
        concat_memory = torch.matmul(score_m.detach(), updated_memory)  # (N,d)
        updated_query = torch.cat((q, concat_memory), dim=1).view(b, h, w, 2 * d)
        updated_query = updated_query.permute(0, 3, 1, 2)
        return updated_query, score_q, score_m

    
    
