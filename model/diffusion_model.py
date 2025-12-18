import math
import numpy as np
import torch
import torch.nn as nn

class Diffusion:
    def __init__(self, timesteps: int = 100, schedule: str = 'linear',
                 start: float = 1e-4, end: float = 5e-2,
                 scale_param: float = 1e-5, num_tau: int = 10):
        self.timesteps = timesteps
        self.schedule = schedule
        self.start = start
        self.end = end
        self.scale_param = scale_param
        self.num_tau = num_tau

        # Precompute beta/alpha sequences on CPU; will move to device when used
        self.betas = self.make_beta_schedule(schedule=self.schedule,
                                             n_timesteps=self.timesteps,
                                             start=self.start, end=self.end)
        self.alphas = 1.0 - self.betas
        self.alphas_prod = torch.cumprod(self.alphas, dim=0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1.0 - self.alphas_prod)

    @staticmethod
    def make_beta_schedule(schedule: str = 'linear',
                           n_timesteps: int = 1000,
                           start: float = 1e-5, end: float = 1e-2
                           , s: float = 0.008) -> torch.Tensor:
        """
        Create beta schedule tensor of shape (n_timesteps,).
        """
        if schedule == 'linear':
            betas = torch.linspace(start, end, n_timesteps)
        elif schedule == "quad":
            betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
        elif schedule == "sigmoid":
            tmp = torch.linspace(-6, 6, n_timesteps)
            betas = torch.sigmoid(tmp) * (end - start) + start
        elif schedule == "cosine":
            """
            cosine schedule
            as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
            """
            steps = n_timesteps + 1
            t = torch.linspace(0, n_timesteps, steps) / n_timesteps
            alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        return betas

    @staticmethod
    def extract(input: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Gather values from `input` at indices `t` and reshape to broadcast with `x`.
        """
        shape = x.shape
        out = torch.gather(input, 0, t.to(input.device))
        reshape = [t.shape[0]] + [1] * (len(shape) - 1)
        return out.reshape(*reshape)

    def p_sample(self, model, x, mask, cur_y, t: int):
        """
        Single denoising step corresponding to original `p_sample`.
        - model: should provide `generate_noise(cur_y, beta, x, mask)`
        - x, mask: context inputs
        - cur_y: current noisy tensor
        - t: timestep index (int)
        """
        # prepare tensors on the same device as cur_y
        device = cur_y.device
        betas = self.betas.to(device)
        alphas = self.alphas.to(device)
        alphas_bar_sqrt = self.alphas_bar_sqrt.to(device)
        one_minus_alphas_bar_sqrt = self.one_minus_alphas_bar_sqrt.to(device)

        t_tensor = torch.tensor([t], dtype=torch.long, device=device)
        # eps factor and mean calculation follow original code logic
        eps_factor = ((1 - self.extract(alphas, t_tensor, cur_y)) /
                      self.extract(one_minus_alphas_bar_sqrt, t_tensor, cur_y))

        # beta passed to model should have batch dimension
        beta_for_model = self.extract(betas, t_tensor.repeat(x.shape[0]), cur_y)
        eps_theta = model.generate_noise(cur_y, beta_for_model, x, mask)

        mean = (1.0 / self.extract(alphas, t_tensor, cur_y).sqrt()) * (cur_y - (eps_factor * eps_theta))

        # sampling noise
        z = torch.randn_like(cur_y).to(device)
        sigma_t = self.extract(betas, t_tensor, cur_y).sqrt()
        sample = mean + sigma_t * z * self.scale_param

        return sample

    def batch_denoise_loop(self, model, x, mask, loc, n, tau_steps: int = None):
        """
        Batch denoising loop
        Splits `loc` into two halves, runs denoising for each half across tau steps,
        and returns concatenated result in the same ordering as original code.
        """
        if tau_steps is None:
            tau_steps = self.num_tau

        sample_num = int(n / 2)
        cur_y_first = loc[:, :sample_num]
        cur_y_second = loc[:, sample_num:]

        for tau in reversed(range(tau_steps)):
            cur_y_first = self.p_sample(model, x, mask, cur_y_first, tau)
            cur_y_second = self.p_sample(model, x, mask, cur_y_second, tau)

        prediction_total = torch.cat((cur_y_second, cur_y_first), dim=1)
        return prediction_total
