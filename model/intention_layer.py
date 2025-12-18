import math
import torch
import torch.nn as nn
from torch.nn import Module, Linear
import torch.nn.functional as F


class Encoder_pasttraj(nn.Module):
	def __init__(self, past_len, latent_dim=128):
		super(Encoder_pasttraj, self).__init__()
		self.encode_past = nn.Linear(past_len*2, latent_dim, bias=False)
		self.layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=2, dim_feedforward=latent_dim)
		self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=2)

	def forward(self, h, mask):
		'''
		h: batch_size, t, 2
		'''
		h_feat = self.encode_past(h.reshape(h.size(0), -1)).unsqueeze(1)
		# n_samples, 1, 64
		h_feat_ = self.transformer_encoder(h_feat, mask)
		h_feat = h_feat + h_feat_

		return h_feat


class SelfAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SelfAttention, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.query = nn.Linear(in_dim, out_dim, bias=False)
        self.key = nn.Linear(in_dim, out_dim, bias=False)
        self.value = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, hidden_state):
        """
        hidden_state: [B, 1, in_dim]
        return: [B, 1, out_dim]
        """
        assert hidden_state.shape[1] == 1, "The shape of dimension 1 must be 1"

        # [B, 1, C] -> [B, 1, out_dim]
        q = self.query(hidden_state)
        k = self.key(hidden_state)
        v = self.value(hidden_state)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.out_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)

        output = torch.matmul(attn_weights, v)
        return output


class CrossAttention(nn.Module):
	def __init__(self, in_dim, out_dim, in_q_dim, hid_q_dim):
		super(CrossAttention, self).__init__()
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.in_q_dim = in_q_dim
		self.hid_q_dim = hid_q_dim
		# define q k v
		self.query = nn.Linear(in_q_dim, hid_q_dim, bias=False)
		self.key = nn.Linear(in_dim, out_dim, bias=False)
		self.value = nn.Linear(in_dim, out_dim, bias=False)

	def forward(self, hidden_state_query, hidden_state_key):
		"""
		hidden_state: [B, 1, in_dim]
		return: [B, 1, out_dim]
		"""
		assert hidden_state_query.shape[1] == 1, 'The shape of dimesion 2 must be 1'
		assert hidden_state_key.shape[1] == 1, 'The shape of dimesion 2 must be 1'
		hidden_state_query = hidden_state_query.permute(1, 0, 2)
		hidden_state_key = hidden_state_key.permute(1, 0, 2)

		x = self.query(hidden_state_query)
		y = self.key(hidden_state_key)

		attn_scores = torch.matmul(x, y.transpose(-2, -1)) / (
					self.out_dim ** 0.5)
		attn_weights = F.softmax(attn_scores, dim=-1)

		V = self.value(hidden_state_key)
		output = torch.bmm(attn_weights, V)

		return output


class MLP(nn.Module):
	def __init__(self, in_feat, out_feat, hid_feat=(1024, 512), activation=None, dropout=-1):
		super(MLP, self).__init__()
		dims = (in_feat, ) + hid_feat + (out_feat, )

		self.layers = nn.ModuleList()
		for i in range(len(dims) - 1):
			self.layers.append(nn.Linear(dims[i], dims[i + 1]))

		self.activation = activation if activation is not None else lambda x: x
		self.dropout = nn.Dropout(dropout) if dropout != -1 else lambda x: x

	def forward(self, x):
		for i in range(len(self.layers)):
			x = self.activation(x)
			x = self.dropout(x)
			x = self.layers[i](x)
		return x


class IntentionEncoder(nn.Module):
	def __init__(self, obs_len, latent_dim):
		super(IntentionEncoder, self).__init__()
		self.obs_encoder = Encoder_pasttraj(obs_len, latent_dim)
		self.int_encoder = MLP(in_feat=2, out_feat=latent_dim, hid_feat=(1024, 512), activation=nn.ReLU())
		self.cross_att = CrossAttention(in_dim=latent_dim, out_dim=latent_dim, in_q_dim=latent_dim, hid_q_dim=latent_dim)
		self.ln = nn.LayerNorm(latent_dim*2)
		self.drop = nn.Dropout(p=0.2, inplace=True)
		self.latent_encoder = MLP(in_feat=latent_dim, out_feat=32, hid_feat=(256, 128), activation=nn.ReLU())
		self.decoder = MLP(in_feat=latent_dim+16, out_feat=2, hid_feat=(256, 128), activation=nn.ReLU())

	def forward(self, past_traj, end_point, traj_mask, exp):

		obs_features = self.obs_encoder(past_traj, traj_mask)
		if exp == 'Train':
			des_features = self.int_encoder(end_point)
			des_features = des_features.unsqueeze(1)

			x = self.cross_att(des_features, obs_features)
			x = x.permute(1, 0, 2)
			latent_inputs = x.squeeze(1)
			latent = self.latent_encoder(latent_inputs)

			mu = latent[:, 0:16]
			logvar = latent[:, 16:]

			var = logvar.mul(0.5).exp_()
			eps = torch.DoubleTensor(var.size()).normal_()
			eps = eps.cuda()
			z = eps.mul(var).add_(mu)
		elif exp == 'Test':
			z = torch.Tensor(past_traj.size(0), 16)
			z.normal_(0, 1.3)

		z = z.to(torch.float32)
		decoder_input = torch.cat((obs_features.squeeze(1), z.cuda()), dim=1)
		estimated_dest = self.decoder(decoder_input)

		int_features = self.function(past_traj, traj_mask, obs_features, estimated_dest, exp)

		if exp == 'Train':
			return mu, logvar, estimated_dest, int_features
		return estimated_dest

	def function(self, past_traj, traj_mask, past_encoded, estimated_dest, exp):
		if not exp == 'Train':
			obs_features = self.obs_encoder(past_traj, traj_mask)
		else:
			obs_features = past_encoded
		pred_des_features = self.int_encoder(estimated_dest)
		pred_des_features = pred_des_features.unsqueeze(1)

		x = self.cross_att(pred_des_features, obs_features)
		x = x.permute(1, 0, 2)
		int_features = torch.cat((pred_des_features, x), dim=-1)
		int_features = int_features.squeeze(1)

		int_features = self.drop(self.ln(int_features))

		return int_features

