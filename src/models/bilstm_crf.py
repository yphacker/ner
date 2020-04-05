# coding=utf-8
# author=yphacker

from torch.nn import LayerNorm
import torch.nn as nn
from conf import config
from conf import model_config_lstm as model_config
from models.crf import CRF
from utils.data_utils import load_vocab


class SpatialDropout(nn.Dropout2d):
    def __init__(self, p=0.6):
        super(SpatialDropout, self).__init__(p=p)

    def forward(self, x):
        x = x.unsqueeze(2)  # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x


class Model(nn.Module):
    def __init__(self, ):
        super(Model, self).__init__()
        self.hidden_size = model_config.hidden_size
        self.embedding = nn.Embedding(config.num_vocab, config.embed_dim)
        self.bilstm = nn.LSTM(input_size=config.embed_dim, hidden_size=self.hidden_size,
                              batch_first=True, num_layers=2, dropout=model_config.dropout,
                              bidirectional=True)
        # self.dropout = SpatialDropout(drop_p)
        self.dropout = nn.Dropout(model_config.dropout)
        self.layer_norm = LayerNorm(self.hidden_size * 2)
        self.classifier = nn.Linear(self.hidden_size * 2, config.num_labels)
        self.crf = CRF(tagset_size=config.num_labels, tag_dictionary=config.label2id, is_bert=True)

    def forward(self, input_ids, attention_mask, input_lens, labels=None):
        embs = self.embedding(input_ids)
        embs = self.dropout(embs)
        embs = embs * attention_mask.float().unsqueeze(2)
        seqence_output, _ = self.bilstm(embs)
        seqence_output = self.layer_norm(seqence_output)
        logits = self.classifier(seqence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf.calculate_loss(logits, tag_list=labels, lengths=input_lens)
            outputs = (loss,) + outputs
        return outputs
