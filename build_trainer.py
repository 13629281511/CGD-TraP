import numpy as np
import random
import torch
import time
import copy
from torch import nn
from tqdm import tqdm
from datasets.data_process import NBADataset, SDDDataset
from torch.utils.data import DataLoader
from tool.tools import SDDdata_process, NBAdata_process, print_log, scale_sample_prediction


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


def prepare_seed(rand_seed):
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)


class Trainer:
    def __init__(self, cfg, model, diffusion, log, log_path):
        super().__init__()

        # Setting device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cfg = cfg
        self.diffusion = diffusion
        self.model = model.to(self.device)
        self.log = log
        self.log_path = log_path

        if self.cfg.ema is True:
            self.ema = EMA(0.995)
            self.ema_model = copy.deepcopy(model).eval().requires_grad_(False)
            self.ema_setup = (self.cfg.ema, self.ema, self.ema_model)
        else:
            self.ema_model = None
            self.ema_setup = None

        self.iter = 0
        self.init_loss = 100000

    def initialization(self):
        # initialize the SEED
        prepare_seed(self.cfg.manual_seed)
        # Prepare for Loading Dataset
        if self.cfg.dataset_type == 'SDD':
            self.train_dataset = SDDDataset(data_path=self.cfg.data_path, obs_len=self.cfg.obs_len, pred_len=self.cfg.pred_len,
                                       flip_aug=True, mode='train')
            self.test_dataset = SDDDataset(data_path=self.cfg.data_path, obs_len=self.cfg.obs_len, pred_len=self.cfg.pred_len,
                                      flip_aug=False, mode='test')
        elif self.cfg.dataset_type == 'NBA':
            self.train_dataset = NBADataset(obs_len=self.cfg.obs_len, pred_len=self.cfg.pred_len, training=True)
            self.test_dataset = NBADataset(obs_len=self.cfg.obs_len, pred_len=self.cfg.pred_len, training=False)

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.cfg.decay_step, gamma=self.cfg.decay_gamma)

        self.criterion = nn.MSELoss()
        self.temporal_reweight = torch.FloatTensor([21 - i for i in range(1, 21)]).cuda().unsqueeze(0).unsqueeze(0) / 10    # sdd:13, nba:21
        self.count = 0
        self.best_ade = 100000
        self.best_fde = 100000
        self.best_jade = 100000
        self.best_jfde = 100000

    def prepare_to_train(self):
        self.model.train()
        self.train_loss = 0
        # Package dataset for training and testing
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=1)
        self.test_loader = DataLoader(self.test_dataset, batch_size=256, shuffle=False, num_workers=1)

    def begin_to_train(self):
        exp = 'Train'
        loss_total, loss_dt, loss_dc, loss_kld, loss_conf, loss_sc, count = 0, 0, 0, 0, 0, 0, 0
        train_loader = tqdm(self.train_loader)
        for _, data in enumerate(train_loader):
            if self.cfg.dataset_type == 'SDD':
                past_traj, fut_traj, traj_mask = SDDdata_process(self.cfg, data, mode='train')
                seq_start_end_list = None
            elif self.cfg.dataset_type == 'NBA':
                traj_mask, past_traj, fut_traj = NBAdata_process(self.cfg, data)
                seq_start_end_list = None

            data_dict = self.model(self.cfg,
                                    past_traj,
                                    fut_traj,
                                    seq_start_end_list,
                                    traj_mask,
                                    exp)
            init_traj = scale_sample_prediction(data_dict['sample'], data_dict['var'], data_dict['mean'])

            pred_traj = self.diffusion.batch_denoise_loop(self.model, past_traj, traj_mask, init_traj,
                                                 self.cfg.num_sample)

            pred_traj_std = pred_traj.reshape(-1, self.cfg.num_sample, self.cfg.pred_len * 2).std(dim=-1)
            fut_traj_std = fut_traj.reshape(-1, self.cfg.pred_len * 2)[:, None].std(dim=-1)
            loss_sample_consistency = torch.norm((pred_traj_std - fut_traj_std), p=2, dim=-1).mean()

            loss_kld_ = -0.5 * torch.sum(
                1 + data_dict['var'] - data_dict['mean'].sum(dim=-1).mean(dim=-1).pow(2).reshape(-1, 1)
                - data_dict['var'].exp())
            dest_loss_dist = self.criterion(data_dict['dest'], fut_traj[:, -1])
            if self.cfg.dataset_type == 'SDD':
                marginal_loss_dist = ((pred_traj - fut_traj.unsqueeze(dim=1)).norm(p=2, dim=-1)).sum(axis=-1).min(dim=1)[
                    0].mean()
                joint_loss_dist = ((pred_traj - fut_traj.unsqueeze(dim=1)).norm(p=2, dim=-1)).sum(axis=-1).mean(dim=0).min()
            elif self.cfg.dataset_type == 'NBA':
                marginal_loss_dist = \
                ((pred_traj - fut_traj.unsqueeze(dim=1)).norm(p=2, dim=-1) * self.temporal_reweight).sum(axis=-1).min(dim=1)[
                    0].mean()
                joint_loss_dist = ((pred_traj - fut_traj.unsqueeze(dim=1)).norm(p=2, dim=-1) * self.temporal_reweight).sum(axis=-1).mean(
                    dim=0).min()
            loss_uncertainty = (torch.exp(-data_dict['var'])
                                * (pred_traj - fut_traj.unsqueeze(dim=1)).norm(p=2, dim=-1).mean(dim=(1, 2))
                                + data_dict['var']).mean()

            loss = marginal_loss_dist * self.cfg.hyper_param1 + joint_loss_dist * self.cfg.hyper_param2 + loss_uncertainty + loss_kld_ * 100 + loss_sample_consistency + dest_loss_dist * 20  # for training SDD
            # loss = marginal_loss_dist * self.cfg.hyper_param1 + joint_loss_dist * self.cfg.hyper_param2 + loss_uncertainty + loss_kld_ * 100 + loss_sample_consistency + dest_loss_dist * 5  # for training NBA

            loss_kld += loss_kld_.item() * 100
            loss_dt += marginal_loss_dist.item() * self.cfg.hyper_param1 + joint_loss_dist.item() * self.cfg.hyper_param2 + dest_loss_dist.item() * 5
            loss_dc += loss_uncertainty.item()
            loss_sc += loss_sample_consistency.item()
            loss_total += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            self.optimizer.step()

            args_ema, ema, ema_model = self.ema_setup[0], self.ema_setup[1], self.ema_setup[2]
            if args_ema is True:
                ema.step_ema(ema_model, self.model)

            count += 1

        self.iter += 1
        print_log('[{}] Epoch: {}\tTotal_loss: {:.6f}\tDist_loss: {:.6f}\tUncertainty_loss: {:.6f}\tKLD_loss: {:.6f}\tSC_loss: {:.6f}'.format(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            self.iter, loss_total/count, loss_dt/count, loss_dc/count, loss_kld/count, loss_sc/count), self.log)
        self.train_loss = loss_total/count

    def prepare_to_val(self):
        self.scheduler.step()
        self.eval_model = self.ema_model if (self.cfg.ema and self.ema_model is not None) else self.model
        self.eval_model.eval()
        if not self.cfg.dataset_type == 'NBA':
            self.performance = {'APD': 0, 'FDE': 0, 'ADE': 0,
                           'JADE': 0, 'JFDE': 0}
        else:
            self.performance = {'FDE': [0, 0, 0, 0],
                           'ADE': [0, 0, 0, 0],
                           'JADE': [0, 0, 0, 0],
                           'JFDE': [0, 0, 0, 0]}

    def begin_to_val(self):
        exp = 'Test'
        samples = 0
        val_count = 0
        if self.iter % self.cfg.eval_iter == 0 and self.iter != 0:
            with torch.no_grad():
                for data in self.test_loader:
                    if self.cfg.dataset_type == 'SDD':
                        traj_scale = self.cfg.data_scale
                        past_traj, fut_traj, traj_mask = SDDdata_process(self.cfg, data, mode='test')
                        seq_start_end_list = None
                    elif self.cfg.dataset_type == 'NBA':
                        traj_scale = self.cfg.data_scale
                        traj_mask, past_traj, fut_traj = NBAdata_process(self.cfg, data)
                        seq_start_end_list = None

                    data_dict = self.model(self.cfg,
                                           past_traj,
                                           fut_traj,
                                           seq_start_end_list,
                                           traj_mask,
                                           exp)
                    init_traj = scale_sample_prediction(data_dict['sample'], data_dict['var'], data_dict['mean'])

                    pred_traj = self.diffusion.batch_denoise_loop(self.model, past_traj, traj_mask, init_traj, self.cfg.num_sample)

                    fut_traj = fut_traj.unsqueeze(1).repeat(1, self.cfg.num_sample, 1, 1)

                    distances = torch.norm(fut_traj - pred_traj, dim=-1) * traj_scale

                    if self.cfg.dataset_type == 'NBA':
                        for time_i in range(1, 5):
                            ade = (distances[:, :, :5 * time_i]).mean(dim=-1).min(dim=-1)[0].sum()
                            fde = (distances[:, :, 5 * time_i - 1]).min(dim=-1)[0].sum()
                            jade = (distances[:, :, :5 * time_i]).mean(dim=-1).mean(dim=0).min()
                            jfde = (distances[:, :, 5 * time_i - 1]).mean(dim=0).min()

                            self.performance['ADE'][time_i - 1] += ade.item()
                            self.performance['FDE'][time_i - 1] += fde.item()
                            self.performance['JADE'][time_i - 1] += jade.item()
                            self.performance['JFDE'][time_i - 1] += jfde.item()
                    else:
                        if pred_traj.shape[1] == 1:
                            diversity = 0.0
                        dist_diverse = torch.pdist(pred_traj.reshape(pred_traj.shape[1], -1))
                        diversity = dist_diverse.mean()
                        ade = distances.mean(dim=-1).min(dim=-1)[0].sum()
                        fde = distances[:, :, -1].min(dim=-1)[0].sum()
                        jade = distances.mean(dim=-1).mean(dim=0).min()
                        jfde = distances[:, :, -1].mean(dim=0).min()

                        self.performance['APD'] += diversity.item()
                        self.performance['ADE'] += ade.item()
                        self.performance['FDE'] += fde.item()
                        self.performance['JADE'] += jade.item()
                        self.performance['JFDE'] += jfde.item()

                    samples += distances.shape[0]
                    val_count += 1

            if not self.cfg.dataset_type == 'NBA':
                if self.best_ade > self.performance['ADE'] / samples:
                    self.best_ade = self.performance['ADE'] / samples
                if self.best_fde > self.performance['FDE'] / samples:
                    self.best_fde = self.performance['FDE'] / samples
                print_log('Evaluation\t--APD: {:.4f}\t--ADE: {:.4f}\t--FDE: {:.4f}\t--JADE: {:.4f}\t--JFDE: {:.4f}'.format(
                        self.performance['APD'] / len(self.test_loader), self.performance['ADE'] / samples, self.performance['FDE'] / samples,
                    self.performance['JADE'] / len(self.test_loader), self.performance['JFDE'] / len(self.test_loader)), self.log)
                if self.iter % 10 == 0:
                    print_log(
                        'Epoch {}: |----Best ADE: {:.4f}----|----Best FDE: {:.4f}----|\t'.format(
                            self.iter, self.best_ade, self.best_fde), self.log)
            else:
                for time_i in range(4):
                    print_log('--ADE({}s): {:.4f}\t--FDE({}s): {:.4f}\t--JADE({}s): {:.4f}\t--JFDE({}s): {:.4f}'.format(
                        time_i+1, self.performance['ADE'][time_i] / samples,
                        time_i+1, self.performance['FDE'][time_i] / samples,
                        time_i+1, self.performance['JADE'][time_i] / len(self.test_loader),
                        time_i+1, self.performance['JFDE'][time_i] / len(self.test_loader)), self.log)

    def after_val(self):
        # save model parameters
        if self.init_loss > self.train_loss:
            self.init_loss = self.train_loss
            model_path = self.log_path + '/' + 'model_opt'
            if self.cfg.ema is True:
                model_state = {'model_dict': self.ema_model.state_dict()}
                torch.save(model_state, model_path)
                print("model saved in {}".format(model_path))
            else:
                model_state = {'model_dict': self.model.state_dict()}
                torch.save(model_state, model_path)
                print("model saved in {}".format(model_path))

    def loop(self):
        self.initialization()
        for self.iter in range(0, self.cfg.num_epoch):
            self.prepare_to_train()
            self.begin_to_train()
            self.prepare_to_val()
            self.begin_to_val()
            self.after_val()
