import numpy as np
import random
import torch
import argparse
from omegaconf import OmegaConf
from model.diffusion_model import Diffusion
from tool.tools import SDDdata_process, NBAdata_process, scale_sample_prediction
from model.model_initializer import TrajModel
from dataset.data_process import NBADataset, SDDDataset
from torch.utils.data import DataLoader
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def test_process(args, test_loader, model, diffusion, traj_scale=None):

    model.eval()
    exp = 'Test'
    if not args.dataset_type == 'NBA':
        performance = {'FDE': 0, 'ADE': 0,
                       'JADE': 0, 'JFDE': 0}
    else:
        performance = {'FDE': [0, 0, 0, 0],
                       'ADE': [0, 0, 0, 0],
                       'JADE': [0, 0, 0, 0],
                       'JFDE': [0, 0, 0, 0]}
    samples = 0
    count = 0
    with torch.no_grad():
        for data in test_loader:
            if  args.dataset_type == 'SDD':
                traj_scale = args.data_scale
                past_traj, fut_traj, traj_mask = SDDdata_process(args, data, mode='test')
                seq_start_end_list = None
            elif args.dataset_type == 'NBA':
                traj_scale = args.data_scale
                traj_mask, past_traj, fut_traj = NBAdata_process(data)
                seq_start_end_list = None

            data_dict = model(cfg,
                            past_traj,
                            fut_traj,
                            seq_start_end_list,
                            traj_mask,
                            exp)
            init_traj = scale_sample_prediction(data_dict['sample'], data_dict['var'], data_dict['mean'])

            pred_traj = diffusion.batch_denoise_loop(model, past_traj, traj_mask, init_traj,
                                                 cfg.num_sample)

            fut_traj = fut_traj.unsqueeze(1).repeat(1, args.num_sample, 1, 1)
            # b*n, K, T, 2
            distances = torch.norm(fut_traj - pred_traj, dim=-1) * traj_scale

            if args.dataset_type == 'NBA':
                for time_i in range(1, 5):
                    ade = (distances[:, :, :5 * time_i]).mean(dim=-1).min(dim=-1)[0].sum()
                    fde = (distances[:, :, 5 * time_i - 1]).min(dim=-1)[0].sum()
                    jade = (distances[:, :, :5 * time_i]).mean(dim=-1).mean(dim=0).min()
                    jfde = (distances[:, :, 5 * time_i - 1]).mean(dim=0).min()

                    performance['ADE'][time_i-1] += ade.item()
                    performance['FDE'][time_i-1] += fde.item()
                    performance['JADE'][time_i-1] += jade.item()
                    performance['JFDE'][time_i-1] += jfde.item()
            else:
                ade = distances.mean(dim=-1).min(dim=-1)[0].sum()
                fde = distances[:, :, -1].min(dim=-1)[0].sum()
                jade = distances.mean(dim=-1).mean(dim=0).min()
                jfde = distances[:, :, -1].mean(dim=0).min()

                performance['ADE'] += ade.item()
                performance['FDE'] += fde.item()
                performance['JADE'] += jade.item()
                performance['JFDE'] += jfde.item()

            samples += distances.shape[0]
            count += 1

    return performance, samples


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',    type=str,     default='')
    parser.add_argument('--mode',         type=str,     default='test', help='train or test')
    parser.add_argument('--log_dir',      type=str,     default='./logs')
    parser.add_argument('--manual_seed',  type=int,     default=0)
    parser.add_argument('--cfg',          type=str,     default='')
    parser.add_argument('--model_ckpt',   type=str,     default='')

    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg)
    cfg.mode = args.mode
    cfg.data_path = args.data_path
    cfg.log_dir = args.log_dir
    cfg.manual_seed = args.manual_seed
    cfg.model_path = args.model_ckpt

    def prepare_seed(rand_seed):
        np.random.seed(rand_seed)
        random.seed(rand_seed)
        torch.manual_seed(rand_seed)
        torch.cuda.manual_seed_all(rand_seed)

    prepare_seed(0)

    # Prepare for Loading Datasets
    if cfg.dataset_type == 'SDD':
        test_dataset  = SDDDataset(data_path=cfg.data_path, obs_len=cfg.obs_len, pred_len=cfg.pred_len, flip_aug=False, mode='test')
    elif cfg.dataset_type == 'NBA':
        test_dataset  = NBADataset(obs_len=cfg.obs_len, pred_len=cfg.pred_len, training=False)

    # Package dataset for training and testing
    test_loader  = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    diffusion = Diffusion(
        timesteps=cfg.timesteps,
        schedule=cfg.scheduler,
        scale_param=cfg.scale_param,
        num_tau=cfg.num_tau,
    )
    model = TrajModel(obs_len=cfg.obs_len, fut_len=cfg.pred_len, num_sample=cfg.num_sample,
                      latent_dim=cfg.latent_dim, context_dim=cfg.context_dim, n_layer=cfg.num_layers,
                      fusion_way='cat').to(device)

    # load pretrained model
    checkpoint_path = cfg.model_path
    model_cp = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(model_cp['model_dict'])

    performance, samples = test_process(args=cfg, test_loader=test_loader, model=model, diffusion=diffusion)
    if not cfg.dataset_type == 'NBA':
        print('Evaluation\t--ADE: {:.4f}\t--FDE: {:.4f}\t--JADE: {:.4f}\t--JFDE: {:.4f}'.format(
            performance['ADE'] / samples, performance['FDE'] / samples,
            performance['JADE'] / len(test_loader), performance['JFDE'] / len(test_loader)))
    else:
        for time_i in range(4):
            print('--ADE({}s): {:.4f}\t--FDE({}s): {:.4f}\t--JADE({}s): {:.4f}\t--JFDE({}s): {:.4f}'.format(
                time_i + 1, performance['ADE'][time_i] / samples,
                time_i + 1, performance['FDE'][time_i] / samples,
                time_i + 1, performance['JADE'][time_i] / len(test_loader),
                time_i + 1, performance['JFDE'][time_i] / len(test_loader)))

