"""CompoundE benchmark variants from ACL 2023 Table 11.

Independent adaptation because the public repo only exposes the OGB tail variant.
Normalization, angle range and adversarial loss follow public model.py. Rotation
layout follows appendix E: adjacent 2D blocks for FB; real/imag halves for WN.
The complete FB form shares one rotation parameter across the head and tail,
as stated by the appendix's relation-embedding visualization description.
No claim of performance equivalence until full training is measured.
"""
import torch
from torch import nn
from torch.nn import functional as F


class CompoundE(nn.Module):
    def __init__(self, nentity, nrelation, dim, gamma, variant):
        super().__init__()
        assert dim % 2 == 0
        assert variant in ['fb_complete_srt', 'wn_rst_st', 'ogb_tail_str']
        self.variant, self.gamma, self.dim = variant, gamma, dim
        self.embedding_range = (gamma + 2) / dim
        self.entity = nn.Embedding(nentity, dim)
        # Retain the upstream d-dimensional angle storage (half is used).
        # FB/WN use head scale/translation, a shared/head rotation, and tail
        # scale/translation: five d-dimensional components in total.
        self.relation = nn.Embedding(nrelation, dim * (3 if variant == 'ogb_tail_str' else 5))
        nn.init.uniform_(self.entity.weight, -self.embedding_range, self.embedding_range)
        nn.init.uniform_(self.relation.weight, -self.embedding_range, self.embedding_range)

    def rotate(self, x, angle):
        angle = angle[..., :self.dim // 2] * (torch.pi / self.embedding_range)
        if self.variant == 'wn_rst_st':
            re, im = x.chunk(2, -1)
            return torch.cat([re * angle.cos() - im * angle.sin(),
                              re * angle.sin() + im * angle.cos()], -1)
        re, im = x[..., 0::2], x[..., 1::2]
        return torch.stack([re * angle.cos() - im * angle.sin(),
                            re * angle.sin() + im * angle.cos()], -1).flatten(-2)

    def forward(self, triples, candidates=None, direction='tail'):
        h, r, t = triples.unbind(-1)
        h, t = self.entity(h).unsqueeze(1), self.entity(t).unsqueeze(1)
        if candidates is not None:
            if direction == 'head':
                h = self.entity(candidates)
            else:
                t = self.entity(candidates)
        h, t = F.normalize(h, dim=-1), F.normalize(t, dim=-1)
        parts = self.relation(r).unsqueeze(1).split(self.dim, -1)
        if self.variant == 'ogb_tail_str':
            s, tr, a = parts
            t = s * (self.rotate(t, a) + tr)
        else:
            sh, th, angle, st, tt = parts
            if self.variant == 'fb_complete_srt':
                h = sh * self.rotate(h + th, angle)
                t = st * self.rotate(t + tt, angle)
            else:
                h, t = self.rotate(sh * (h + th), angle), st * (t + tt)
        return self.gamma - (h - t).abs().sum(-1)
