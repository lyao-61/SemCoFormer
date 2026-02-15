import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math, copy, time
from torch import Tensor
from typing import Optional
#from mamba_ssm import Mamba


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class get_derivatives(nn.Module):
    noise_type = "general"
    sde_type = "stratonovich"

    def __init__(self, d_model=256, diff_feat=512, brownian_size=1):
        super(get_derivatives, self).__init__()
        self.drift = nn.Sequential(nn.Linear(d_model, diff_feat), nn.Tanh(),
                                   nn.Linear(diff_feat, diff_feat), nn.Tanh(),
                                   nn.Linear(diff_feat, d_model))
        self.diffusion = nn.Sequential(nn.Linear(d_model, diff_feat), nn.Tanh(),
                                       nn.Linear(diff_feat,diff_feat), nn.Tanh(),
                                       nn.Linear(diff_feat,d_model * brownian_size))
        self.d_model = d_model
        self.diff_feat = diff_feat
        self.brownian_size = brownian_size

    def f(self, t, y):
        out = self.drift(y)
        return out

    def g(self, t, y):
        out = self.diffusion(y).view(-1, self.d_model, self.brownian_size)
        return out


class VisionLanguageFusionModule(nn.Module):
    def __init__(self, VLdmodel, VLnhead, visual_dim, dropout):
        super().__init__()
        self.input_proj = nn.Linear(visual_dim,VLdmodel)  #(config["feat_dim"],config["d_model"])
        self.output_proj = nn.Linear(VLdmodel, visual_dim)
        self.multihead_attn = nn.MultiheadAttention(
            VLdmodel, VLnhead, dropout=dropout
        )
        self.vid_embed_proj = nn.Conv2d(5, 15680, kernel_size=1)
        self.top_k = 50
        self.km = None

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt,
        memory,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):

        # if tgt[1] is not None and memory[1] is not None:
        #     min_dim = min(tgt.size(1), memory.size(1))
        #     tgt = tgt[:, :min_dim]
        #     memory = memory[:, :min_dim]
        tgt_proj = self.input_proj(tgt).transpose(0, 1)
        memory_proj = self.input_proj(memory).transpose(0, 1)
        tgt2, weight = self.multihead_attn(
            query=self.with_pos_embed(tgt_proj, query_pos),
            key=self.with_pos_embed(memory_proj, pos),
            value=memory_proj,
            attn_mask=None,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt2 = tgt2.transpose(0, 1)     # [B, T, 256]
        tgt2 = self.output_proj(tgt2)  # [B, T, 1536]
        tgt = tgt * tgt2
        return tgt

class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # [d_model//2]

        pe = torch.zeros(1, max_len, d_model)  # [1, max_len, d_model]
        pe[0, :, 0::2] = torch.sin(position * div_term)  # even dims
        pe[0, :, 1::2] = torch.cos(position * div_term)  # odd dims

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor, indices=None) -> Tensor:
        if indices is None:
            x = x + self.pe[:, :x.size(1)]
        else:
            pos = torch.cat([self.pe[:, index] for index in indices])
            x = x + pos
        return self.dropout(x)


class PositionEncoder(nn.Module):
    def __init__(self, d_model, length):
        super().__init__()
        freqs = torch.Tensor(
            [10000 ** (-i / d_model) if i % 2 == 0 else -10000 ** ((1 - i) / d_model) for i in range(d_model)]) \
            .unsqueeze(dim=1)
        phases = torch.Tensor([0 if i % 2 == 0 else math.pi / 2 for i in range(d_model)]).unsqueeze(dim=1)
        pos = torch.arange(length).repeat(d_model, 1).to(torch.float)
        self.pos_encoding = torch.sin(torch.add(torch.mul(pos, freqs), phases))
        self.pos_encoding = self.pos_encoding.transpose(1, 0)
        self.pos_encoding = nn.Parameter(self.pos_encoding, requires_grad=False)

    def forward(self, x):
        return x + self.pos_encoding


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        #x = self.dropout(x)
        return x



class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=None, element_width=None, head_width=None):
        super().__init__()
        self.dropout = dropout
        self.element_width = element_width
        self.head_width = head_width

    # Q: [nb, nh, len1, hid1], K: [nb, nh, len2, hid2], V: [nb, nh, len2, hid2], mask: [nb, len2]
    def forward(self, Q, K, V, mask=None):
        if self.head_width is None:
            if self.element_width is not None:
                K = F.pad(K, (0, 0, self.element_width, self.element_width))
                V = F.pad(V, (0, 0, self.element_width, self.element_width))
                mask = F.pad(mask, (self.element_width, self.element_width))
            out = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))  # [nb, nh, len1, len2]
            if mask is not None:
                mask = mask.unsqueeze(1)  # mask: [nb, 1, len2]
                mask = mask.unsqueeze(1)  # mask: [nb, 1, 1, len2]
                out = out.masked_fill(mask == 0, -1e30)  # [nb, nh, len1, len2]
            if self.element_width is not None:
                mask = torch.zeros(out.size(-2), out.size(-1))
                for i in range(self.element_width, out.size(-2) - self.element_width):
                    mask[i, i - self.element_width:i + self.element_width + 1] = 1
                mask = mask.unsqueeze(0)
                mask = mask.unsqueeze(0)
                mask = mask.cuda()
                out = out.masked_fill(mask == 0, -1e30)
            attn = F.softmax(out, dim=-1)
            if self.dropout is not None:
                attn = self.dropout(attn)
            out = torch.matmul(attn, V)  # [nb, nh, len1, hid2]
        else:
            if self.element_width is not None:
                K = F.pad(K, (0, 0, self.element_width, self.element_width))
                V = F.pad(V, (0, 0, self.element_width, self.element_width))
                mask = F.pad(mask, (self.element_width, self.element_width))
            if self.head_width is not None:
                K = F.pad(K, (0, 0, 0, 0, self.head_width, self.head_width))
                V = F.pad(V, (0, 0, 0, 0, self.head_width, self.head_width))
                Q = F.pad(Q, (0, 0, 0, 0, self.head_width, self.head_width))
            element_mask = None
            if mask is not None:
                mask = mask.unsqueeze(1)  # mask: [nb, 1, len2]
                mask = mask.unsqueeze(1)  # mask: [nb, 1, 1, len2]
            if self.element_width is not None:
                element_mask = torch.zeros(Q.size(-2), K.size(-2))
                for i in range(self.element_width, Q.size(-2) - self.element_width):
                    element_mask[i, i - self.element_width:i + self.element_width + 1] = 1
                element_mask = element_mask.unsqueeze(0)  # [1, len1, len2]
                element_mask = element_mask.unsqueeze(0)  # [1, 1, len1, len2]
                element_mask = element_mask.cuda()

            num_heads = Q.size(1)
            attn_matrices = []
            K_T = K.transpose(-2, -1)
            for h in range(self.head_width, num_heads - self.head_width):
                attn_matrix = torch.matmul(Q[:, h:h + 1], K_T[:, h - self.head_width:h + self.head_width]) / math.sqrt(
                    Q.size(-1))
                # h->h-n...h+n: [nb, nrh, len1, len2]
                if mask is not None:
                    attn_matrix = attn_matrix.masked_fill(mask == 0, -1e30)
                if element_mask is not None:
                    attn_matrix = attn_matrix.masked_fill(element_mask == 0, -1e30)
                attn_matrices.append(attn_matrix)

            # softmax
            for i, h in enumerate(range(self.head_width, num_heads - self.head_width)):
                attn_matrix = attn_matrices[i]
                nb, nrh, len1, len2 = attn_matrix.shape
                attn_score = F.softmax(attn_matrix.transpose(1, 2).contiguous().view(nb, len1, nrh * len2), -1)
                attn_score = attn_score.transpose(1, 2).contiguous().view(nb, nrh, len1, len2)
                attn_matrices[i] = attn_score

            outs = []
            for i, h in enumerate(range(self.head_width, num_heads - self.head_width)):
                out = torch.matmul(attn_matrices[i],
                                   V[:, h - self.head_width:h + self.head_width])  # [nb, nrh, len1, hid2]
                out = torch.sum(out, 1)  # [nb, len1, hid2]
                outs.append(out)
                # print(out.shape)
            out = torch.stack(outs, 0).transpose(0, 1)
            # outs: [nb, nh, len1, hid2]
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout, element_width=None, head_width=None):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(dropout)
        self.element_width = element_width
        self.head_width = head_width
        self.attn = ScaledDotProductAttention(self.dropout, element_width=element_width, head_width=head_width)

    def forward(self, Q, K, V, mask=None):
        # Q: [nb, len1, d_model], K: [nb, len2, d_model], V: [nb, len2, d_model]
        num_batches = Q.size(0)
        Q, K, V = [
            l(x).view(num_batches, -1, self.num_heads, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (Q, K, V))
        ]
        # Q: [nb, nh, len1, d_k], K: [nb, nh, len2, d_k], V: [nb, nh, len2, d_k]
        x = self.attn(Q, K, V, mask)
        x = x.transpose(1, 2).contiguous().view(num_batches, -1, self.num_heads * self.d_k)
        # [nb, len1, d_model]
        return self.linears[-1](x)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout=0.1):
        super().__init__()
        self.size = size
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        res = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, mask)
        x = self.dropout(x)
        x = res + x

        res = x
        x = self.feed_forward(self.norm2(x))
        x = self.dropout(x)
        x = res + x
        return x


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

#
# class MyEncoder(nn.Module):
#     def __init__(self, layers):
#         super().__init__()
#         self.layers = layers
#         self.norm = LayerNorm(layers[0].size)
#
#     def forward(self, x, mask):
#         for layer in self.layers:   # ori
#             x = layer(x, mask)
#         return self.norm(x)

class MyEncoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(layers[0].size if hasattr(layers[0], 'size') else layers[0].norm1.normalized_shape[0])

    def forward(self, x, mask):
        for layer in self.layers:
            if hasattr(layer, 'forward') and 'Sparse_Self_Attention' in layer.__class__.__name__:
                #  Sparse_Self_Attention
                B, T, C = x.shape
                x_reshaped = x.permute(0, 2, 1).unsqueeze(2)   # [B, C, 1, T]
                x_reshaped = layer(x_reshaped)                 # -> [B, C, 1, T]
                x = x_reshaped.squeeze(2).permute(0, 2, 1)     # -> [B, T, C]
            else:
                x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size,
                 self_attn,
                 src_attn,
                 feed_forward, dropout=0.1):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.src_attn = src_attn
        self.norms = clones(LayerNorm(size), 3)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, target_mask):
        res = x
        x = self.norms[0](x)
        x = self.self_attn(x, x, x, target_mask)
        x = self.dropout(x)
        x = res + x

        res = x
        x = self.norms[1](x)
        x = self.src_attn(x, memory, memory, src_mask)
        x = self.dropout(x)
        x = res + x

        res = x
        x = self.feed_forward(self.norms[2](x))
        x = self.dropout(x)
        x = res + x
        return x


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, target_mask)
        return self.norm(x)


class MyDecoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(layers[0].size)

    def forward(self, x, memory, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, target_mask)
        return self.norm(x)

class WrapperModel(nn.Module):
    def __init__(self, model, config, device):
        super().__init__()
        self.model = model
        self.config = config
        self.device = device

    def forward(self, x):  # x ÊÇ dummy ÊäÈë
        B, T = x.shape[:2]
        D = self.config["feat_dims"] * 4 + self.config["semantic_dim"]
        T2 = self.config["max_target"]
        D2 = self.config["semantic_dim"]  # Ä¿±êÌØÕ÷Î¬¶È

        src = torch.randn(B, T, D).to(self.device)
        tgt = torch.randn(B, T2, D2).to(self.device)
        src_mask = torch.ones(B, T).bool().to(self.device)
        tgt_mask = torch.ones(B, T2).bool().to(self.device)

        pred, target, loss = self.model(src, src_mask, tgt, tgt_mask, epoch=0, device=self.device)
        return pred  # ·µ»ØÓÃÓÚÍ³¼Æ MACs µÄÕÅÁ¿

class QuadrangleEncoderWrapper(nn.Module):
    def __init__(self, attn_module, norm_dim, h, w):
        super().__init__()
        self.attn = attn_module
        self.norm1 = nn.LayerNorm(norm_dim)
        self.norm2 = nn.LayerNorm(norm_dim)
        self.ffn = nn.Sequential(
            nn.Linear(norm_dim, norm_dim * 4),
            nn.GELU(),
            nn.Linear(norm_dim * 4, norm_dim),
        )
        self.h = h
        self.w = w

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), self.h, self.w)
        x = x + self.ffn(self.norm2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, src_size, tgt_size, num_heads, d_model, d_ff, n_encoder_layers, n_decoder_layers):
        super().__init__()
        self.encoder = Encoder(EncoderLayer(src_size,
                                            MultiHeadAttention(num_heads, d_model),
                                            PositionWiseFeedForward(d_model, d_ff)), n_encoder_layers)
        self.decoder = Decoder(DecoderLayer(tgt_size,
                                            MultiHeadAttention(num_heads, d_model),
                                            MultiHeadAttention(num_heads, d_model),
                                            PositionWiseFeedForward(d_model, d_ff)), n_decoder_layers)

    def forward(self, src, src_mask, tgt, tgt_mask):
        x = self.encoder(src, src_mask)  # [nb, len1, hid]
        x = self.decoder(tgt, x, src_mask, tgt_mask)  # [nb, len2, hid]
        return x


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, num_vetex, act=F.relu, dropout=0.5, bias=True):
        super(GraphConvolution, self).__init__()

        self.alpha = 1.

        self.act = act
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.randn(output_dim))
        else:
            self.bias = None

        for w in [self.weight]:
            nn.init.xavier_normal_(w)

    def normalize(self, m):
        rowsum = torch.sum(m, 0)
        r_inv = torch.pow(rowsum, -0.5)
        r_mat_inv = torch.diag(r_inv).float()

        m_norm = torch.mm(r_mat_inv, m)
        m_norm = torch.mm(m_norm, r_mat_inv)

        return m_norm

    def forward(self, adj, x):

        x = self.dropout(x)

        # K-ordered Chebyshev polynomial
        adj_norm = self.normalize(adj)
        sqr_norm = self.normalize(torch.mm(adj, adj))
        m_norm = self.alpha * adj_norm + (1. - self.alpha) * sqr_norm

        x_tmp = torch.einsum('abcd,de->abce', x, self.weight)
        x_out = torch.einsum('ij,abid->abjd', m_norm, x_tmp)
        if self.bias is not None:
            x_out += self.bias

        x_out = self.act(x_out)

        return x_out


class StandConvolution(nn.Module):
    def __init__(self, dims, num_classes, dropout):
        super(StandConvolution, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=1, stride=2),
            nn.InstanceNorm2d(dims[1]),
            nn.ReLU(inplace=True),
            # nn.AvgPool2d(3, stride=2),
            nn.Conv2d(dims[1], dims[2], kernel_size=1, stride=2),
            nn.InstanceNorm2d(dims[2]),
            nn.ReLU(inplace=True),
            # nn.AvgPool2d(3, stride=2),
            nn.Conv2d(dims[2], dims[3], kernel_size=1, stride=2),
            nn.InstanceNorm2d(dims[3]),
            nn.ReLU(inplace=True),
            # nn.AvgPool2d(3, stride=2)
        )

        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.dropout(x.permute(0, 3, 1, 2))
        x_tmp = self.conv(x)
        x_out = self.fc(x_tmp.view(x.size(0), -1))

        return x_out


class StandRecurrent(nn.Module):
    def __init__(self, dims, num_classes, dropout):
        super(StandRecurrent, self).__init__()

        self.lstm = nn.LSTM(dims[0] * 45, dims[1], batch_first=True,
                            dropout=0)
        self.fc = nn.Linear(dims[1], num_classes)

    def forward(self, x):
        x_tmp, _ = self.lstm(x.contiguous().view(x.size(0), x.size(1), -1))
        x_out = self.fc(x_tmp[:, -1])

        return x_out


class FC(nn.Module):
    def __init__(self, in_features, out_features, relu=True):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


if __name__ == '__main__':
    nb = 1
    nh = 1
    len1 = 4
    len2 = 3

    a = torch.randn(128, 10, 128)
    a_mask = torch.ones(10, 12)
    print(a_mask)
    b = torch.randn(128, 10, 256)
    model = Transformer(src_size=10,tgt_size=10,num_heads=8,d_model=256,d_ff=256,n_encoder_layers=2,n_decoder_layers=2)
    output = model(a,a_mask,b,a_mask)

    print(a.size())
    print(b.size())
