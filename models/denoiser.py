import torch
import torch.nn as nn
import torch.nn.functional as F

class Denoiser(nn.Module):
    def __init__(self, n_feats=256, n_denoise_res=5, steps=5):
        super().__init__()
        self.max_period = steps * 10
        resmlp = [
            nn.Linear(n_feats*2+1, n_feats),
            nn.LeakyReLU(0.1, True),
        ]
        for _ in range(n_denoise_res):
            resmlp.append(ResMLP(n_feats))
        self.resmlp=nn.Sequential(*resmlp)

    def forward(self, x, t, c):
        t=t.float()
        t =t/self.max_period
        t=t.view(-1,1)
        c = torch.cat([c,t,x],dim=1)
        
        fea = self.resmlp(c)

        return fea 

class ResMLP(nn.Module):
    def __init__(self, n_feats=256):
        super().__init__()
        self.resmlp = nn.Sequential(
            nn.Linear(n_feats , n_feats),
            nn.LeakyReLU(0.1, True),
        )
    def forward(self, x):
        res=self.resmlp(x)
        return res