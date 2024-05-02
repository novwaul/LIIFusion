import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord

from models.prior_injector import PriorInjector
from models.denoiser import Denoiser
from models.diffusion import GaussianDiffusion

@register('liifusion_stage2')
class LIIF(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True, mode='test'):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        self.n_feats = 64
        self.n_convs = 7
        self.n_fcns = 2
        self.steps = 4
        self.mode = mode

        self.encoder = models.make(encoder_spec)

        if imnet_spec is not None:
            imnet_in_dim = self.encoder.out_dim
            if self.feat_unfold:
                imnet_in_dim *= 9
            imnet_in_dim += 2 # attach coord
            if self.cell_decode:
                imnet_in_dim += 2
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
        else:
            self.imnet = None

        self.prior_encoder = self.gen_encoder(6)
        
        self.prior_injector = PriorInjector()

        self.condition_encoder = self.gen_encoder(3)
        denoiser = Denoiser(n_feats=4*self.n_feats, steps=self.steps)
        self.prior_generator = GaussianDiffusion(denoiser, steps=self.steps)
    
    def gen_encoder(self, inc):
        encoder = nn.Sequential(
            nn.Conv2d(inc, self.n_feats, kernel_size=3, padding=1),
            *[nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, padding=1) for _ in range(self.n_convs)],
            nn.Conv2d(self.n_feats, self.n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.n_feats * 2, self.n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.n_feats * 2, self.n_feats * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            *[nn.Sequential(nn.Linear(self.n_feats * 4, self.n_feats * 4), nn.LeakyReLU(0.1, True),) for _ in range(self.n_fcns)]
        )
        return encoder

    def gen_feat(self, inp, ref):
        feat = self.encoder(inp)
        cond = self.condition_encoder(inp)

        *_, h, w = ref.shape
        upsampled_inp = nn.functional.interpolate(inp, size=(h,w), mode='bicubic')
        if cond.requires_grad:
            self.prior = self.prior_encoder(torch.cat((ref, upsampled_inp), dim=1)).detach()
            noise = torch.randn_like(self.prior)
        else:
            self.prior = None
            noise = torch.randn((inp.shape[0], 4*self.n_feats), device=inp.device)

        self.sampled_prior = self.prior_generator(noise, cond)
        self.feat = self.prior_injector(feat, self.sampled_prior)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat

        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        
        if self.prior == None:
            return ret
        else:
            L_diff = torch.abs((self.prior - self.sampled_prior))
            return ret, L_diff

    def forward(self, inp, ref, coord, cell):
        self.gen_feat(inp, ref)
        return self.query_rgb(coord, cell)
