import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm


# from torch.nn import functional as F
from ml.dp.base import BaseModel


def parse_layer(layer_str, init_dim=None, last_dim=None):
    def _get_args(__l: str, prefix):
        __l = __l.replace(prefix, '')
        return __l

    def replace_args(_lst, _args, _pos):
        if _args:
            _lst[_pos] = _args
        return _lst

    def prepare_args(_layer_str, _prefix, _id_init, _id_last):
        args = _get_args(_layer_str, _prefix)
        args, _kwargs = eval(args)
        args = list(args)
        args = replace_args(args, init_dim, _id_init)
        args = replace_args(args, last_dim, _id_last)
        return args, _kwargs

    layer_str = layer_str.lstrip(' ')
    if layer_str.startswith('Conv1d'):
        prep_args, kwargs = prepare_args(layer_str, 'Conv1d', 0, 1)
        return torch.nn.Conv1d(*prep_args, **kwargs)
    if layer_str.startswith('Linear'):
        prep_args, kwargs = prepare_args(layer_str, 'Linear', 0, 1, )
        return torch.nn.Linear(*prep_args, **kwargs)


def str_to_pytorch(inp: str, init_dim, last_dim):
    inp = inp.split(';')
    if len(inp) == 1:
        return [parse_layer(inp[0], init_dim, int(last_dim))]
    out = [parse_layer(inp[0], init_dim)]
    for layer in inp[1:-1]:
        out.append(parse_layer(layer))
    out.append(parse_layer(inp[-1], None, int(last_dim)))
    return out


# def init_weights(m):
#     if type(m) == nn.Linear:
#         torch.nn.init.kaiming_normal_(m.weight)
#         if m.bias is not None:
#             torch.nn.init.zeros_(m.bias)

class LSTM(nn.Module):

    def __init__(self, device, lstm_width, lstm_count=1, splitted_input_features=None, features_encoder=None,
                 input_dimension=None, bidirectional=False, simple_head=False, attention=False,
                 num_heads=1):
        super(LSTM, self).__init__()
        lstm_width = int(lstm_width)
        lstm_count = int(lstm_count)
        if splitted_input_features:
            in_dim = len(splitted_input_features)
        elif input_dimension:
            in_dim = input_dimension
        else:
            in_dim = 3
        if not features_encoder:
            self.embedding = nn.Linear(in_dim, lstm_width) # wrong code has to be conv not linear
        else:
            self.embedding = nn.Sequential(*str_to_pytorch(features_encoder, in_dim, lstm_width))

        self.LSTMS = torch.nn.LSTM(lstm_width, lstm_width, lstm_count, bidirectional=bidirectional)

        if simple_head:
            self.regressor_head = nn.Sequential(
                nn.Linear(lstm_width * (2 if bidirectional else 1) * (2 if attention else 1), 1),
                nn.Sigmoid()
            )
        else:
            self.regressor_head = nn.Sequential(
                nn.Linear(lstm_width * (2 if bidirectional else 1) * (2 if attention else 1), 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

        self.LSTMS = self.LSTMS.to(device)
        self.regressor_head = self.regressor_head.to(device)
        self.embedding = self.embedding.to(device)
        self.attention = attention
        if self.attention:
            self.nn_attention = nn.MultiheadAttention(lstm_width * (2 if bidirectional else 1), num_heads)

    def forward(self, src):
        if isinstance(self.embedding, torch.nn.Linear):
            embed_src = self.embedding(src)
            embed_src = embed_src.permute(1, 0, 2)
        else:
            src = src.permute(0, 2, 1)
            embed_src = self.embedding(src)
            embed_src = embed_src.permute(2, 0, 1)
        hidden_states = self.LSTMS(embed_src)
        if self.attention:
            attn = self.nn_attention(hidden_states[0], hidden_states[0], hidden_states[0])
            value = torch.cat((hidden_states[0], attn[0]), 2)
            value = self.regressor_head(value)
        else:
            value = self.regressor_head(hidden_states[0])
        # value = self.regressor(h2)
        # return h1, value.view(-1)
        # return value.view(-1)
        return torch.squeeze(value).T
