from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianDiffusion(nn.Module):
    def __init__(self, model, steps):
        super().__init__()
        # register common variables
        self.steps = steps
        self.model = model
        # self.register_buffer("betas", torch.linspace(start=0.1, end=0.99, steps=self.steps))  # noise schedule
        # self.register_buffer("betas", torch.linspace(start=1e-3, end=0.05, steps=self.steps))  # noise schedule
        self.register_buffer("betas", torch.linspace(start=5e-4, end=3e-3, steps=self.steps))  # noise schedule
        self.register_buffer("alphas", 1. - self.betas)
        self.register_buffer("alphas_bar", torch.cumprod(self.alphas, dim=0))

        # register variables to calculate loss
        self.register_buffer("sqrt_alphas_bar", torch.sqrt(self.alphas_bar))
        self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(1. - self.alphas_bar))

        # register variables to sample images
        self.register_buffer("one_over_sqrt_alphas", torch.sqrt(1. / self.alphas))
        self.register_buffer("betas_over_sqrt_one_minus_alphas_bar", self.betas / torch.sqrt(1. - self.alphas_bar)) # 1 - alpha = beta
        self.register_buffer("sigmas", torch.sqrt(self.betas))

    def _extract(self, values, times, dimension_num):
        B, *_ = times.shape
        selected_values = torch.gather(values, dim=0, index=times)
        return selected_values.reshape((B, *[1 for _ in range(dimension_num-1)])) # to broadcast coefficients

    def forward(self, x_T, c):
        # Algorithm 2
        B, *_ = x_T.shape
        dimension_num = len(x_T.shape)
        x_t = x_T
        for step in reversed(range(self.steps)):
            times = step * x_t.new_ones((B, ), dtype=torch.int64)
            epsilon = self.model(x_t, times, c)
            one_over_sqrt_alpha = self._extract(self.one_over_sqrt_alphas, times, dimension_num)
            beta_over_sqrt_one_minus_alpha_bar = self._extract(self.betas_over_sqrt_one_minus_alphas_bar, times, dimension_num)
            # get x_{t-1} from x_{t} and set to new x_{t}
            x_t = one_over_sqrt_alpha * ( x_t - beta_over_sqrt_one_minus_alpha_bar * epsilon )

        x_0 = x_t
        return x_0



