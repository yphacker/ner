# coding=utf-8
# author=yphacker

import os
import json
import pickle as pkl
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.utils import NerTokenizer

from conf import config


def build_vocab():
    tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
    vocab_dic = {}

    train_df = pd.read_csv(config.train_path)
    texts = train_df['text'].values.tolist()
    for text in texts:
        if not text:
            continue
        for word in tokenizer(text):
            vocab_dic[word] = vocab_dic.get(word, 0) + 1
    vocab_list = vocab_dic.keys()
    word2id = {word: i + 2 for i, word in enumerate(vocab_list)}
    word2id["<SEP>"] = 0
    word2id["<CLS>"] = 1
    print(len(word2id))
    pkl.dump(word2id, open(config.vocab_path, 'wb'))
    return vocab_dic


def load_vocab():
    if os.path.exists(config.vocab_path):
        word2id = pkl.load(open(config.vocab_path, 'rb'))
    else:
        word2id = build_vocab()
    id2word = {v: k for k, v in word2id.items()}
    return word2id, id2word


class MyDataset(Dataset):

    def __init__(self, df, mode='train'):
        self.mode = mode
        self.tokenizer = lambda x: [y for y in x]
        self.word2id, _ = load_vocab()
        self.x_data = []
        self.y_data = []
        for i, row in df.iterrows():
            x, y = self.row_to_tensor(row)
            self.x_data.append(x)
            self.y_data.append(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def row_to_tensor(self, row):
        text = row["text"]
        x_data = list(text)

        y_data = ['O'] * len(x_data)
        if self.mode == 'train':
            label_entities = row['label']
            label_entities = json.loads(label_entities)
            for key, value in label_entities.items():
                for sub_name, sub_index in value.items():
                    for start_index, end_index in sub_index:
                        assert ''.join(x_data[start_index:end_index + 1]) == sub_name
                        if start_index == end_index:
                            y_data[start_index] = 'S-' + key
                        else:
                            y_data[start_index] = 'B-' + key
                            y_data[start_index + 1:end_index + 1] = ['I-' + key] * (len(sub_name) - 1)

        tokens = self.tokenizer(x_data)
        label_ids = [config.label2id[label] for label in y_data]
        # input_mask = [1] * len(input_ids)
        # input_len = len(label_ids)
        # # Zero-pad up to the sequence length.
        # padding_length = config.max_seq_len - len(input_ids)
        #
        # pad_token = 0
        # input_ids += [pad_token] * padding_length
        # input_mask += [0] * padding_length
        # label_ids += [pad_token] * padding_length

        sep_token = "[SEP]"
        cls_token = "[CLS]"
        pad_token = 0

        tokens += [sep_token]
        label_ids += [config.label2id[sep_token]]

        input_ids = [cls_token] + tokens
        label_ids = [config.label2id[cls_token]] + label_ids

        input_ids = [self.word2id[word] for word in x_data]
        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1] * len(input_ids)
        input_len = len(label_ids)
        # Zero-pad up to the sequence length.
        padding_length = config.max_seq_len - len(input_ids)

        input_ids += [pad_token] * padding_length
        input_mask += [0] * padding_length
        # pad_token代表标签X
        label_ids += [pad_token] * padding_length

        x_data = input_ids, input_mask, input_len
        y_data = label_ids
        return x_data, y_data

    def __len__(self):
        return len(self.y_data)


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    x_data = [item[0] for item in batch]
    y_data = [item[1] for item in batch]

    input_lens = [x[2] for x in x_data]
    max_len = max(input_lens)

    input_ids = [x[0][:max_len] for x in x_data]
    input_mask = [x[1][:max_len] for x in x_data]
    label_ids = [x[:max_len] for x in y_data]

    x_tensor = torch.tensor(input_ids, dtype=torch.long), \
               torch.tensor(input_mask, dtype=torch.long), \
               torch.tensor(input_lens, dtype=torch.long)
    y_tensor = torch.tensor(label_ids, dtype=torch.long)
    return x_tensor, y_tensor


if __name__ == '__main__':
    word2id, id2word = load_vocab()
    print(len(word2id))
