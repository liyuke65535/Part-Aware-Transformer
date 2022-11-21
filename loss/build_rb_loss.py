import torch
from torch import nn
import torch.nn.functional as F

def euclidean_distance(qf, gf, reduction='mean'):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    if reduction == 'sum':
        return dist_mat.sum()
    if reduction == 'mean':
        return dist_mat.mean()
    print("No such reduction!!")

def cosine_similarity(qf, gf, reduction='sum'):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot)
    dist_mat = torch.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = torch.arccos(dist_mat)
    if reduction == 'sum':
        return dist_mat.sum()
    if reduction == 'mean':
        return dist_mat.mean()
    print("No such redution way!")
    
class RB_LOSS:
    def __init__(self, loss_type='COS', reduction='mean', normalize=True, device='cuda'):
        self.loss_type = loss_type
        self.reduction = reduction
        self.normalize = normalize
        self.loss_cnt = torch.tensor(0.).to(device)

    def kl_loss(self, x, y, tao):
        self.loss_cnt += F.kl_div(
            F.log_softmax(x / tao, dim=1),
            F.log_softmax(y / tao, dim=1),
            reduction=self.reduction,
            log_target=True
        )
        
    def cosine_loss(self, x, y):
        self.loss_cnt += cosine_similarity(x, y, reduction=self.reduction)
        
    def euclidean_loss(self, x, y):
        self.loss_cnt += euclidean_distance(x, y, reduction=self.reduction)
        
    def reset_loss(self):
        self.loss_cnt = 0