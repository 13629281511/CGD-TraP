
import os
import argparse
import numpy as np

from omegaconf import OmegaConf
from build_trainer import Trainer
from model.diffusion_model import Diffusion
from model.model_initializer import TrajModel


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',    type=str,     default='')
    parser.add_argument('--mode',         type=str,     default='', help='train or test')
    parser.add_argument('--log_dir',      type=str,     default='./logs')
    parser.add_argument('--manual_seed',  type=int,     default=42)
    parser.add_argument('--eval_iter',    type=int,     default=2)
    parser.add_argument('--cfg',          type=str,     default='')
    parser.add_argument('--model_ckpt',   type=str,     default='')

    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)

    cfg.mode = args.mode
    cfg.data_path = args.data_path
    cfg.log_dir = args.log_dir
    cfg.manual_seed = args.manual_seed
    cfg.eval_iter = args.eval_iter

    diffusion = Diffusion(
        timesteps=cfg.timesteps,
        schedule=cfg.scheduler,
        scale_param=cfg.scale_param,
        num_tau=cfg.num_tau,
    )
    model = TrajModel(obs_len=cfg.obs_len, fut_len=cfg.pred_len, num_sample=cfg.num_sample,
                      latent_dim=cfg.latent_dim, context_dim=cfg.context_dim, mamba_dim=cfg.mamba_dim,
                      n_layer=cfg.num_layers, data_type=cfg.dataset_type, fusion_way='cat')

    # Count the number of total model parameters
    model_parameters = filter(lambda a: a.requires_grad, model.parameters())
    parameters = sum([np.prod(a.size()) for a in model_parameters])
    print('>>>>>> Total trainable parameters are: {:.2f}M'.format(parameters / 1e6))

    # Log prepare
    log_path = cfg.log_dir + '/' + cfg.dataset_type
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log = open(os.path.join(log_path, 'log.txt'), 'a+')

    OmegaConf.save(cfg, os.path.join(log_path, 'configs.yaml'))

    # Train Stage
    if cfg.mode == 'train':
        trainer = Trainer(
                cfg=cfg,
                model=model,
                diffusion=diffusion,
                log=log,
                log_path=log_path
            )
        trainer.loop()

