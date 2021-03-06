# coding=utf-8
# author=yphacker

import os
from conf import config

pretrain_model_name = 'bert-base-chinese'
pretrain_model_path = os.path.join(config.pretrain_model_path, pretrain_model_name)

hidden_size = 768
dropout = 0.1

learning_rate = 1e-3
adjust_lr_num = 2
