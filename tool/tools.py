import numpy as np
import torch


def NBAdata_process(args, data):
    """
    :param data: input one batch dataset
    :return: traj_mask, past_traj, fut_traj
    """
    traj_mean = [14, 7.5]
    traj_mean = torch.FloatTensor(traj_mean).cuda().unsqueeze(0).unsqueeze(0).unsqueeze(0)

    batch_size = data['pre_motion_3D'].shape[0]

    traj_mask = torch.zeros(batch_size*11, batch_size*11).cuda()
    for i in range(batch_size):
        traj_mask[i*11:(i+1)*11, i*11:(i+1)*11] = 1
    initial_pos = data['pre_motion_3D'].cuda()[:, :, -1:]
    # augment input: absolute position, relative position, velocity
    past_traj_abs = ((data['pre_motion_3D'].cuda() - traj_mean)/args.data_scale).contiguous().view(-1, 10, 2)
    past_traj_rel = ((data['pre_motion_3D'].cuda() - initial_pos)/args.data_scale).contiguous().view(-1, 10, 2)
    past_traj_vel = torch.cat((past_traj_rel[:, 1:] - past_traj_rel[:, :-1], torch.zeros_like(past_traj_rel[:, -1:])), dim=1)
    past_traj = torch.cat((past_traj_abs, past_traj_rel, past_traj_vel), dim=-1)

    fut_traj = ((data['fut_motion_3D'].cuda() - initial_pos)/args.data_scale).contiguous().view(-1, 20, 2)

    return traj_mask, past_traj, fut_traj


def SDDdata_process(args, data, mode='train'):
    """
    :param data: input one batch dataset
    :return: past_traj, fut_traj, traj_mask
    """
    if mode == 'train':
        traj_mean = [700, 842]
    elif mode == 'test':
        traj_mean = [783, 914]
    traj_mean = torch.FloatTensor(traj_mean).cuda().unsqueeze(0).unsqueeze(0)

    past_traj = data['past_traj'].type(torch.float).cuda()
    fut_traj = data['fut_traj'].type(torch.float).cuda()

    batch_size = past_traj.shape[0]
    traj_mask = torch.zeros(batch_size, batch_size).cuda()
    for i in range(batch_size):
        traj_mask[i:(i + 1), i:(i + 1)] = 1

    initial_pos_ = past_traj[:, -1:]
    # augment input: absolute position, relative position, velocity
    past_traj_abs = ((past_traj - traj_mean) / args.data_scale)
    past_traj_rel = ((past_traj - initial_pos_) / args.data_scale)
    past_traj_vel = torch.cat((past_traj_rel[:, 1:] - past_traj_rel[:, :-1], torch.zeros_like(past_traj_rel[:, -1:])),
                              dim=1)
    past_traj = torch.cat((past_traj_abs, past_traj_rel, past_traj_vel), dim=-1)
    fut_traj = (fut_traj - initial_pos_) / args.data_scale

    return past_traj, fut_traj, traj_mask


def print_log(print_str, log, same_line=False, display=True):
    '''
    print a string to a log file

    parameters:
        print_str:          a string to print
        log:                a opened file to save the log
        same_line:          True if we want to print the string without a new next line
        display:            False if we want to disable to print the string onto the terminal
    '''
    if display:
        if same_line:
            print('{}'.format(print_str), end='')
        else:
            print('{}'.format(print_str))

    if same_line:
        log.write('{}'.format(print_str))
    else:
        log.write('{}\n'.format(print_str))
    log.flush()


def scale_sample_prediction(sample_prediction: torch.Tensor,
                            variance_estimation: torch.Tensor,
                            mean_estimation: torch.Tensor,
                            eps: float = 1e-8) -> torch.Tensor:
    """
    Scale `sample_prediction` by exp(variance_estimation/2) and normalize by per-sample std.
    Args:
        sample_prediction: Tensor of shape (B, N, T, 2) or similar.
        variance_estimation: Tensor of shape (B, ...) matching batch dim B.
        eps: small value to avoid division by zero.
    Returns:
        Scaled and normalized sample_prediction with same shape as input.
    """
    # expand scaling to match sample_prediction spatial dims
    scaling = torch.exp(variance_estimation / 2)[..., None, None]
    # compute per-sample std over dim=1 then mean over spatial dims (1,2) as original
    per_sample_std = sample_prediction.std(dim=1).mean(dim=(1, 2))
    denom = per_sample_std[:, None, None, None]

    sample = scaling * sample_prediction / denom
    initialized_traj = sample + mean_estimation[:, None]

    return initialized_traj
