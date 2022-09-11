import torch
import torch.nn as nn
import torch.nn.functional as F
from ml.dp.base import BaseModel
from nn.pytorch.pytorch_modules import LSTM
from nn.pytorch.pytorch_modules import str_to_pytorch


class DpModel(LSTM):

    def __init__(self, lstm_width=5, lstm_count=1, bidirectional=False, conv_encoding=True, simple_head=False, device=None, splitted_input_features=None, input_dimension=None, attention=False, num_heads=1):
        if conv_encoding:
            fe = 'Conv1d((None, 32, 3), dict(padding=1));Conv1d((32, 64, 3), dict(padding=1));' \
             'Conv1d((64, 128, 3), dict(padding=1));Conv1d((128, 256, 3), dict(padding=1));' \
             'Conv1d((256, None, 3), dict(padding=1))'
        else:
            fe = None
        super().__init__(lstm_width=lstm_width, lstm_count=lstm_count, device=device,
                         features_encoder=fe, splitted_input_features=splitted_input_features,
                         bidirectional=bidirectional, input_dimension=input_dimension,
                         simple_head=simple_head, attention=attention, num_heads=num_heads)


class LinearOnly(BaseModel):

    def __init__(self, width=256, device=None, splitted_input_features=None, input_dimension=None):
        super(LinearOnly, self).__init__()
        if splitted_input_features:
            in_dim = len(splitted_input_features)
        elif input_dimension:
            in_dim = input_dimension
        else:
            in_dim = 3
        fe = 'Conv1d((None, 32, 3), dict(padding=1));Conv1d((32, 64, 3), dict(padding=1));' \
             'Conv1d((64, 128, 3), dict(padding=1));Conv1d((128, 256, 3), dict(padding=1));' \
             'Conv1d((256, None, 3), dict(padding=1))'
        self.embedding = nn.Sequential(*str_to_pytorch(fe, in_dim, width))
        self.regressor_head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(width, 128),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.regressor_head = self.regressor_head.to(device)
        self.embedding = self.embedding.to(device)

    def forward(self, src):
        src = src.permute(0, 2, 1)
        embed_src = self.embedding(src)
        embed_src = embed_src.permute(2, 0, 1)
        value = self.regressor_head(embed_src)
        # value = self.regressor(h2)
        # return h1, value.view(-1)
        # return value.view(-1)
        return torch.squeeze(value).T

