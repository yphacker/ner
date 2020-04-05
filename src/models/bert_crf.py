# coding=utf-8
# author=yphacker

import torch
import torch.nn as nn
from transformers import BertModel
from conf import config
from conf import model_config_bert as model_config
from models.crf import CRF


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(model_config.pretrain_model_path)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, config.num_labels)
        self.crf = CRF(tagset_size=config.num_labels, tag_dictionary=config.label2id, is_bert=True)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, input_lens=None, labels=None):
        outputs = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf.calculate_loss(logits, tag_list=labels, lengths=input_lens)
            outputs = (loss,) + outputs
        return outputs
