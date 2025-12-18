import numpy as np
import torch
import torch.nn as nn

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from .layers import MLP
from .intention_layer import IntentionEncoder
from .interaction_layer import InteractionEncoder
from .layers import PositionalEncoding, ConcatSquashLinear
from .mamba_model import Mamba, Block


class TrajModel(nn.Module):
	def __init__(self, obs_len=8, fut_len=12, num_sample=20, latent_dim=128, context_dim=256, mamba_dim=128, n_layer=2, data_type=None, fusion_way='cat'):
		super(TrajModel, self).__init__()
		# define feature encoder
		self.obs_len = obs_len
		self.fut_len = fut_len
		self.n = num_sample
		self.stride = fut_len * 2
		self.input_dim = obs_len * 6
		self.hidden_dim = self.input_dim * 4
		self.latent_dim = latent_dim
		self.mamba_dim = mamba_dim
		self.output_dim = self.stride * num_sample
		self.fut_len = fut_len
		self.fusion_way = fusion_way
		self.dataset_type = data_type

		self.int_dim = latent_dim
		self.inter_dim = latent_dim
		self.fused_dim = self.int_dim + self.inter_dim

		self.int_encoder = IntentionEncoder(obs_len=obs_len, latent_dim=latent_dim)
		self.ext_encoder = InteractionEncoder(obs_len=obs_len, latent_dim=latent_dim,
											  n_units=[latent_dim, latent_dim // 2, latent_dim], n_heads=[4, 1], dropout=0,
											  alpha=0.2)
		if fusion_way == 'att':
			self.cross_att_fusion = nn.MultiheadAttention(embed_dim=latent_dim,
                                                num_heads=2,
                                                batch_first=True)

		# define spatial-temporal selective state space model
		if not self.dataset_type == 'NBA':
			self.ST_S3M1 = Block(dim=mamba_dim, mixer_cls=Mamba, norm_cls=RMSNorm, fused_add_norm=True)
			self.ST_S3M2 = Block(dim=mamba_dim, mixer_cls=Mamba, norm_cls=RMSNorm, fused_add_norm=True)
			self.fusion_act = nn.SiLU()

		self.scale_encoder = MLP(1, 32, hid_feat=(4, 16), activation=nn.ReLU())
		self.mean_decoder = MLP(self.fused_dim * 2, self.stride, hid_feat=(512, 256, 128), activation=nn.ReLU())
		self.var_decoder = MLP(self.fused_dim * 2, 1, hid_feat=(512, 256, 128), activation=nn.ReLU())
		self.sample_decoder = MLP(self.fused_dim * 2 + 32, self.output_dim, hid_feat=(1024, 1024), activation=nn.ReLU())

		# define denoise sampling model
		self.context_dim = context_dim
		self.tf_layer = n_layer

		self.context_encoder = Block(dim=6, mixer_cls=Mamba, norm_cls=RMSNorm, fused_add_norm=True)
		self.hidden_encoder = nn.Linear(self.obs_len * 6, 256, bias=False)

		self.pos_emb = PositionalEncoding(d_model=2 * context_dim, dropout=0.1, max_len=24)
		self.concat1 = ConcatSquashLinear(2, 2 * context_dim, context_dim + 3)
		self.concat2 = ConcatSquashLinear(2 * context_dim, context_dim, context_dim + 3)
		self.concat3 = ConcatSquashLinear(context_dim, context_dim // 2, context_dim + 3)

		self.trans_block1 = Block(dim=2 * context_dim, mixer_cls=Mamba, norm_cls=RMSNorm, fused_add_norm=True)
		self.trans_block2 = Block(dim=2 * context_dim, mixer_cls=Mamba, norm_cls=RMSNorm, fused_add_norm=True)
		self.trans_act = nn.SiLU()

		self.out = ConcatSquashLinear(context_dim // 2, 2, context_dim + 3)

	def forward(self, cfg, x, y, seq_start_end_list=None, mask=None, exp=''):
		'''
		x(past_traj): batch size, t_p, 6
		seq_start_end_list: N * (start, end)
		mask: N * N
		'''
		past_traj_rel = x[:, :, 2:4]
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		if exp == 'Train':
			end_point = y[:, -1]
			mu, logvar, pred_dest, encoded_intention = self.int_encoder(past_traj_rel, end_point, mask, exp)
		else:
			des_gt = y[:, -1]
			des_gt = des_gt.cpu().numpy()
			all_guesses = []
			all_fde = []
			for _ in range(self.n):
				coarse_pred_dest = self.int_encoder(past_traj_rel, None, mask, exp)
				coarse_pred_dest = coarse_pred_dest.cpu().numpy()
				all_guesses.append(coarse_pred_dest)
				fde = np.linalg.norm(coarse_pred_dest - des_gt, axis=1)
				all_fde.append(fde)
			all_fde = np.array(all_fde)
			all_guesses = np.array(all_guesses)
			indices = np.argmin(all_fde, axis=0)

			best_guess_dest = all_guesses[indices, np.arange(past_traj_rel.shape[0]), :]
			best_guess_dest = torch.tensor(best_guess_dest, dtype=torch.float32).cuda()
			pred_dest = best_guess_dest
			encoded_intention = self.int_encoder.function(past_traj_rel, mask, None, best_guess_dest, exp)
			mu = None
			logvar = None

		encoded_interaction = self.ext_encoder(cfg, past_traj_rel, seq_start_end_list)
		# select one way to fuse multiple features, 'cat' or 'add' or 'att'
		if self.fusion_way == 'cat':
			if self.dataset_type == 'NBA':
				fused_features = torch.cat((encoded_intention, encoded_interaction), dim=-1)
			else:
				h_mix = torch.cat((encoded_intention, encoded_interaction), dim=-1)
				h1, r1 = self.ST_S3M1(h_mix.reshape(h_mix.size(0), self.obs_len, -1))
				h2, r2 = self.ST_S3M2(h1, r1)
				fused_features = self.fusion_act(h2 + r2)
				fused_features = fused_features.contiguous().view(x.size(0), -1)
		elif self.fusion_way == 'add':
			fused_features = encoded_interaction + encoded_intention
		elif self.fusion_way == 'att':
			encoded_interaction = encoded_interaction.unsqueeze(1)
			encoded_intention = encoded_intention.unsqueeze(1)
			fused_features = self.cross_att_fusion(encoded_intention, encoded_interaction)
			fused_features = fused_features.squeeze(0)

		guess_mean = self.mean_decoder(fused_features).contiguous().view(-1, self.fut_len, 2)
		guess_var = self.var_decoder(fused_features)

		guess_scale_feat = self.scale_encoder(guess_var)
		var_total = torch.cat((fused_features, guess_scale_feat), dim=-1)
		guess_sample = self.sample_decoder(var_total).reshape(x.size(0), self.n, self.fut_len, 2)

		data_dict = {
			'sample': guess_sample,
			'mean': guess_mean,
			'var': guess_var,
			'dest': pred_dest,
			'mu': mu,
			'logvar': logvar
		}
		return data_dict

	def generate_noise(self, current_y, beta, past_context, mask=None):
		"""
        Generate denoised trajectory representations.

        Parameters:
          current_y: Current sample/noise tensor
          beta: Scalar time embedding, shape (B,) or (B,1,1)
          past_context: Past trajectory context, passed to self.context_encoder
          mask: Currently unused, kept for compatibility

        Returns:
          The generated result from the final linear.batch_generate
        """
		# Ensure beta is on the correct device and shape
		batch_size = current_y.size(0)
		beta = beta.view(batch_size, 1, 1).to(current_y.device)  # (B,1,1)

		# Encode context (context_encoder returns context_h, context_r)
		context_h, context_r = self.context_encoder(past_context)
		context = self.hidden_encoder((context_h+context_r).reshape((context_h+context_r).size(0), -1)).unsqueeze(1)

		# Build time embedding and concatenate with context
		time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B,1,3)
		n_half = int(self.n / 2)
		ctx_emb = torch.cat([time_emb, context], dim=-1).repeat(1, n_half, 1).unsqueeze(2)  # (B, n_half, 1, feat)

		# First-level concat_batch_generate encodes current_y into the required feature dimension
		x = self.concat1.batch_generate(ctx_emb, current_y).contiguous().view(-1, self.n, self.fused_dim)
		final_emb = x.permute(1, 0, 2)  # (seq_len, batch', dim)
		final_emb = self.pos_emb(final_emb)

		trans_h1, trans_r1 = self.trans_block1(final_emb)
		trans_h2, trans_r2 = self.trans_block2(trans_h1, trans_r1)
		trans = self.trans_act(trans_h2 + trans_r2)
		# Restore batch dimension and reshape for subsequent concat_batch_generate usage
		trans = trans.permute(1, 0, 2).contiguous().view(-1, n_half, self.fut_len, self.fused_dim)

		# Conditional generation through subsequent concat-squash layers
		trans = self.concat2.batch_generate(ctx_emb, trans)
		trans = self.concat3.batch_generate(ctx_emb, trans)

		# Final linear generation
		outs = self.out.batch_generate(ctx_emb, trans)

		return outs

