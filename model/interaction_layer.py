import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module, Linear
import torch.nn.functional as F


# this efficient implementation comes from https://github.com/xptree/DeepInf/
class BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
        super(BatchMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out
        self.w = nn.Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = nn.Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = nn.Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter("bias", None)

        nn.init.xavier_uniform_(self.w, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)

    def forward(self, h):
        bs, n = h.size()[:2]
        h_prime = torch.matmul(h.unsqueeze(1), self.w)
        attn_src = torch.matmul(h_prime, self.a_src)
        attn_dst = torch.matmul(h_prime, self.a_dst)
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(
            0, 1, 3, 2
        )
        attn = self.leaky_relu(attn)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime)
        if self.bias is not None:
            return output + self.bias, attn
        else:
            return output, attn

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.n_head)
            + " -> "
            + str(self.f_in)
            + " -> "
            + str(self.f_out)
            + ")"
        )


class GAT(nn.Module):
    def __init__(self, n_units, n_heads, dropout=0.2, alpha=0.2):
        super(GAT, self).__init__()
        self.n_layer = len(n_units) - 1
        self.dropout = dropout
        self.layer_stack = nn.ModuleList()

        for i in range(self.n_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(
                BatchMultiHeadGraphAttention(
                    n_heads[i], f_in=f_in, f_out=n_units[i + 1], attn_dropout=dropout
                )
            )

        self.norm_list = [
            torch.nn.InstanceNorm1d(256).cuda(),
            torch.nn.InstanceNorm1d(512).cuda(),
        ]

    def forward(self, x):
        x = x.unsqueeze(0)
        bs, n = x.size()[:2]
        for i, gat_layer in enumerate(self.layer_stack):
            x = self.norm_list[i](x.permute(0, 2, 1)).permute(0, 2, 1)
            x, attn = gat_layer(x)
            if i + 1 == self.n_layer:
                x = x.squeeze(dim=1)
            else:
                x = F.elu(x.transpose(1, 2).contiguous().view(bs, n, -1))
                x = F.dropout(x, self.dropout, training=False)
        else:
            return x


class GATEncoder(nn.Module):
    def __init__(self, n_units, n_heads, dropout, alpha):
        super(GATEncoder, self).__init__()
        self.gat_net = GAT(n_units, n_heads, dropout, alpha)

    def forward(self, args, obs_traj_embedding, seq_start_end):
        if args.dataset_type == 'ETH':
            graph_embeded_data = []
            for (start, end) in seq_start_end:
                curr_seq_embedding_traj = obs_traj_embedding[start:end, :]
                curr_seq_graph_embedding = self.gat_net(curr_seq_embedding_traj)
                graph_embeded_data.append(curr_seq_graph_embedding)
            graph_embeded_data = torch.cat(graph_embeded_data, dim=1)
        else:
            graph_embeded_data = self.gat_net(obs_traj_embedding)

        return graph_embeded_data


class TALayer(nn.Module):
    def __init__(self, time, augment=4):
        super(TALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(time, time * augment, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(time * augment, time, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, t, _ = x.size()
        y = self.avg_pool(x).view(b, t)
        y = self.fc(y).view(b, t, 1)
        return x * y.expand_as(x)


class InteractionEncoder(nn.Module):
    def __init__(self, obs_len, latent_dim, n_units, n_heads, dropout, alpha):
        super(InteractionEncoder, self).__init__()
        self.obs_len = obs_len
        self.traj_lstm_input_size = 2*self.obs_len
        self.traj_lstm_hidden_size = latent_dim
        self.graph_lstm_hidden_size = latent_dim
        self.graph_network_out_dims = latent_dim

        self.gatencoder = GATEncoder(n_units=n_units, n_heads=n_heads, dropout=dropout, alpha=alpha)
        self.traj_lstm_model = nn.LSTMCell(self.traj_lstm_input_size, self.traj_lstm_hidden_size)
        self.graph_lstm_model = nn.LSTMCell(self.graph_network_out_dims, self.graph_lstm_hidden_size)
        # self.TAlayer = TALayer(time=obs_len)

    def init_hidden_traj_lstm(self, batch):
        return (
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
        )

    def init_hidden_graph_lstm(self, batch):
        return (
            torch.randn(batch, self.graph_lstm_hidden_size).cuda(),
            torch.randn(batch, self.graph_lstm_hidden_size).cuda(),
        )

    def forward(self, args, past_traj_rel, seq_start_end_list):

        batchsize = past_traj_rel.shape[0]
        traj_lstm_h_t, traj_lstm_c_t = self.init_hidden_traj_lstm(batchsize)
        graph_lstm_h_t, graph_lstm_c_t = self.init_hidden_graph_lstm(batchsize)

        # Time Attention
        # tem_features = self.TAlayer(past_traj_rel)
        # traj_lstm_input = torch.cat((past_traj_rel, tem_features), dim=1)
        traj_lstm_h_t, traj_lstm_c_t = self.traj_lstm_model(past_traj_rel.contiguous().view(batchsize, -1), (traj_lstm_h_t, traj_lstm_c_t))

        graph_lstm_input = self.gatencoder(args, traj_lstm_h_t, seq_start_end_list)  # shape=(1, bs*n, 32)

        graph_lstm_input = graph_lstm_input.squeeze(0)
        graph_lstm_h_t, graph_lstm_c_t = self.graph_lstm_model(graph_lstm_input, (graph_lstm_h_t, graph_lstm_c_t))
        encoded_interaction_hidden = torch.cat((traj_lstm_h_t, graph_lstm_h_t), dim=1)

        return encoded_interaction_hidden


