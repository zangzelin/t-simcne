import inspect

import torch
import torch.nn.functional as F
from torch import nn

from .base import LossBase
import scipy
import numpy as np


class InfoNCECosine(nn.Module):
    def __init__(
        self,
        temperature: float = 0.5,
        reg_coef: float = 0,
        reg_radius: float = 200,
    ):
        super().__init__()
        self.temperature = temperature
        self.reg_coef = reg_coef
        self.reg_radius = reg_radius

    def forward(self, features, backbone_features=None, labels=None):
        # backbone_features and labels are unused
        batch_size = features.size(0) // 2

        a = features[:batch_size]
        b = features[batch_size:]

        # mean deviation from the sphere with radius `reg_radius`
        vecnorms = torch.linalg.vector_norm(features, dim=1)
        target = torch.full_like(vecnorms, self.reg_radius)
        penalty = self.reg_coef * F.mse_loss(vecnorms, target)

        a = F.normalize(a)
        b = F.normalize(b)

        cos_aa = a @ a.T / self.temperature
        cos_bb = b @ b.T / self.temperature
        cos_ab = a @ b.T / self.temperature

        # mean of the diagonal
        tempered_alignment = cos_ab.trace() / batch_size

        # exclude self inner product
        self_mask = torch.eye(batch_size, dtype=bool, device=cos_aa.device)
        cos_aa.masked_fill_(self_mask, float("-inf"))
        cos_bb.masked_fill_(self_mask, float("-inf"))
        logsumexp_1 = torch.hstack((cos_ab.T, cos_bb)).logsumexp(dim=1).mean()
        logsumexp_2 = torch.hstack((cos_aa, cos_ab)).logsumexp(dim=1).mean()
        raw_uniformity = logsumexp_1 + logsumexp_2

        loss = -(tempered_alignment - raw_uniformity / 2) + penalty
        return loss


class InfoNCECauchy(nn.Module):
    def __init__(self, temperature: float = 1, exaggeration: float = 1):
        super().__init__()
        self.temperature = temperature
        self.exaggeration = exaggeration

    def forward(self, features, backbone_features=None, labels=None):
        # backbone_features and labels are unused
        batch_size = features.size(0) // 2

        a = features[:batch_size]
        b = features[batch_size:]

        sim_aa = 1 / (torch.cdist(a, a) * self.temperature).square().add(1)
        sim_bb = 1 / (torch.cdist(b, b) * self.temperature).square().add(1)
        sim_ab = 1 / (torch.cdist(a, b) * self.temperature).square().add(1)

        tempered_alignment = torch.diagonal(sim_ab).log().mean()

        # exclude self inner product
        self_mask = torch.eye(batch_size, dtype=bool, device=sim_aa.device)
        sim_aa.masked_fill_(self_mask, 0.0)
        sim_bb.masked_fill_(self_mask, 0.0)

        logsumexp_1 = torch.hstack((sim_ab.T, sim_bb)).sum(1).log_().mean()
        logsumexp_2 = torch.hstack((sim_aa, sim_ab)).sum(1).log_().mean()

        raw_uniformity = logsumexp_1 + logsumexp_2

        loss = -(self.exaggeration * tempered_alignment - raw_uniformity / 2)
        return loss



class InfoNCEZL(nn.Module):
    def __init__(self, temperature: float = 1, exaggeration: float = 1):
        super().__init__()
        self.temperature = temperature
        self.exaggeration = exaggeration

    def _DistanceSquared(self, x, y=None, metric="euclidean"):
        if metric == "euclidean":
            if y is not None:
                m, n = x.size(0), y.size(0)
                xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
                yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
                dist = xx + yy
                dist = torch.addmm(dist, mat1=x, mat2=y.t(), beta=1, alpha=-2)
                dist = dist.clamp(min=1e-12)
            else:
                m, n = x.size(0), x.size(0)
                xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
                yy = xx.t()
                dist = xx + yy
                dist = torch.addmm(dist, mat1=x, mat2=x.t(), beta=1, alpha=-2)
                dist = dist.clamp(min=1e-12)
                dist[torch.eye(dist.shape[0]) == 1] = 1e-12
        return dist

    def _CalGamma(self, v):
        a = scipy.special.gamma((v + 1) / 2)
        b = np.sqrt(v * np.pi) * scipy.special.gamma(v / 2)
        out = a / b
        return out

    def _Similarity_old(self, dist, gamma, v=100, h=1, pow=2):
        dist_rho = dist

        dist_rho[dist_rho < 0] = 0
        Pij = (
            gamma
            * torch.tensor(2 * 3.14)
            * gamma
            * torch.pow((1 + dist_rho / v), exponent=-1 * (v + 1))
        )
        return Pij
    
    def _TwowaydivergenceLoss(self, P_, Q_, select=None):
        EPS = 1e-5
        losssum1 = P_ * torch.log(Q_ + EPS)
        losssum2 = (1 - P_) * torch.log(1 - Q_ + EPS)
        losssum = -1 * (losssum1 + losssum2)
        return losssum.mean()
    
    def forward(self, features, backbone_features=None, labels=None):
        # backbone_features and labels are unused
        v_input=100
        v_latent=0.01
        
        batch_size = features.size(0) // 2


        data_1 = backbone_features[:batch_size]
        dis_P = self._DistanceSquared(data_1, data_1)
        latent_data_1 = features[:batch_size]
        dis_P_2 = dis_P  # + nndistance.reshape(1, -1)
        gamma = self._CalGamma(v_input)
        P_2 = self._Similarity_old(dist=dis_P_2, gamma=gamma, v=v_input)
        latent_data_2 = features[batch_size:]
        dis_Q_2 = self._DistanceSquared(latent_data_1, latent_data_2)
        Q_2 = self._Similarity_old(
            dist=dis_Q_2,
            gamma=self._CalGamma(v_latent),
            v=v_latent,
        )
        loss = self._TwowaydivergenceLoss(P_=P_2, Q_=Q_2)

        # import pdb; pdb.set_trace()
        
        # backbone_features_a = backbone_features[:batch_size]
        # backbone_features_b = backbone_features[batch_size:]
        
        # sim_baa = 1 / (torch.cdist(backbone_features_a, backbone_features_a) * self.temperature).square().add(1)
        # sim_bbb = 1 / (torch.cdist(backbone_features_b, backbone_features_b) * self.temperature).square().add(1)
        # sim_bab = 1 / (torch.cdist(backbone_features_a, backbone_features_b) * self.temperature).square().add(1)
        
        # p = torch.diagonal(sim_bab)
        
        
        # features_a = features[:batch_size]
        # features_b = features[batch_size:]

        # sim_aa = 1 / (torch.cdist(features_a, features_a) * self.temperature).square().add(1)
        # sim_bb = 1 / (torch.cdist(features_b, features_b) * self.temperature).square().add(1)
        # sim_ab = 1 / (torch.cdist(features_a, features_b) * self.temperature).square().add(1)

        # tempered_alignment = (torch.diagonal(sim_ab).log()).mean()

        # # exclude self inner product
        # self_mask = torch.eye(batch_size, dtype=bool, device=sim_aa.device)
        # sim_aa.masked_fill_(self_mask, 0.0)
        # sim_bb.masked_fill_(self_mask, 0.0)

        # logsumexp_1 = torch.hstack((sim_ab.T, sim_bb)).sum(1).log_().mean()
        # logsumexp_2 = torch.hstack((sim_aa, sim_ab)).sum(1).log_().mean()

        # raw_uniformity = logsumexp_1 + logsumexp_2

        # loss = -(self.exaggeration * tempered_alignment - raw_uniformity / 2)
        return loss

class InfoNCEGaussian(InfoNCECauchy):
    def forward(self, features, backbone_features=None, labels=None):
        # backbone_features and labels are unused
        batch_size = features.size(0) // 2

        a = features[:batch_size]
        b = features[batch_size:]

        sim_aa = -(torch.cdist(a, a) * self.temperature).square()
        sim_bb = -(torch.cdist(b, b) * self.temperature).square()
        sim_ab = -(torch.cdist(a, b) * self.temperature).square()

        tempered_alignment = sim_ab.trace() / batch_size

        # exclude self inner product
        self_mask = torch.eye(batch_size, dtype=bool, device=sim_aa.device)
        sim_aa.masked_fill_(self_mask, float("-inf"))
        sim_bb.masked_fill_(self_mask, float("-inf"))

        logsumexp_1 = torch.hstack((sim_ab.T, sim_bb)).logsumexp(1).mean()
        logsumexp_2 = torch.hstack((sim_aa, sim_ab)).logsumexp(1).mean()

        raw_uniformity = logsumexp_1 + logsumexp_2

        loss = -(tempered_alignment - raw_uniformity / 2)
        return loss


class InfoNCELoss(LossBase):
    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)

        metric = self.metric
        if metric == "cosine":
            self.cls = InfoNCECosine
        elif metric == "euclidean":  # actually Cauchy
            self.cls = InfoNCECauchy
        elif metric == "gauss":
            self.cls = InfoNCEGaussian
        else:
            raise ValueError(f"Unknown {metric = !r} for InfoNCE loss")

    def get_deps(self):
        supdeps = super().get_deps()
        return [inspect.getfile(self.cls)] + supdeps

    def compute(self):
        self.criterion = self.cls(**self.kwargs)
