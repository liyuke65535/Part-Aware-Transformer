import copy
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from torch.nn import ModuleList
from torch.autograd import Variable
import torch.nn.functional as F
from typing import Optional
import math

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class PositionalEncoding(Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0)
        self.pe = nn.parameter.Parameter(pe.unsqueeze(0), requires_grad=False)
        # self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class EncoderLayer(Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation=F.relu, layer_norm_eps=1e-5, norm_first=False) -> None:
        super(EncoderLayer, self).__init__()

        self.pos_embed = PositionalEncoding(d_model) # sin pos
        # self.pos_embed = nn.Parameter(torch.zeros(1, 128, 256)) # learnable pos
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(EncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src.flatten(2).transpose(1,2) # B,C,H,W -> B,HW,C
        x = self.pos_embed(x)
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x.transpose(1,2).reshape(src.size())

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

# class Encoder(Module):
#     __constants__ = ['norm']

#     def __init__(self, encoder_layer=EncoderLayer, d_model=[256,512,1024,2048], num_layers=3, norm=None):
#         super(Encoder, self).__init__()
#         self.block1 = encoder_layer(d_model=[d_model[0]])
#         self.block2 = encoder_layer(d_model=[d_model[1]])
#         self.block3 = encoder_layer(d_model=[d_model[2]])
#         if num_layers == 4:
#             self.block4 = encoder_layer(d_model=[d_model[3]])

#         self.norm = norm

#     def forward(self, src, mask=None, src_key_padding_mask=None):
#         output = self.pos_embed(src)
#         # output = src + self.pos_embed
#         # output = src
#         outputs = []

#         for mod in self.layers:
#             output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
#             outputs.append(output)

#         if self.norm is not None:
#             for i in len(outputs):
#                 outputs[i] = self.norm(outputs[i])

#         outputs = torch.cat(outputs, dim=-1)
#         return outputs # bs,hw,dim256*nlayer8
#         # return output # bs,hw,dim256

# if __name__ == '__main__':
#     inputs = torch.rand([64, 32,256])
#     layer = EncoderLayer(d_model=256, nhead=8)
#     net = Encoder(layer, num_layers=8)
#     print(net)
#     out = net(inputs)
#     print(out.size())