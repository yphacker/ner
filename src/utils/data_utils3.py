# coding=utf-8
# author=yphacker

import json
import torch
from torch.utils.data import Dataset
from conf import config_span as config
from conf import model_config_bert_span as model_config
from utils.utils import NerTokenizer, get_entities


# 适用于bert+span

class MyDataset(Dataset):

    def __init__(self, df, mode='train'):
        self.mode = mode
        self.tokenizer = NerTokenizer.from_pretrained(model_config.pretrain_model_path)
        self.pad_idx = self.tokenizer.pad_token_id
        self.x_data = []
        self.y_data = []
        for i, row in df.iterrows():
            x, y = self.row_to_tensor(self.tokenizer, row)
            self.x_data.append(x)
            self.y_data.append(y)

    def __getitem__(self, index):
        return self.x_data[index], (self.y_data[index][0],  self.y_data[index][1])

    def get_subjects(self):
        return [item[2] for item in self.y_data]

    def row_to_tensor(self, tokenizer, row):
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

        tokens = tokenizer.tokenize(x_data)
        start_ids = [0] * len(tokens)
        end_ids = [0] * len(tokens)
        subjects = get_entities(y_data, id2label=None, markup='bios')
        subjects_id = []
        for subject in subjects:
            label = subject[0]
            start = subject[1]
            end = subject[2]
            start_ids[start] = config.label2id[label]
            end_ids[end] = config.label2id[label]
            subjects_id.append((config.label2id[label], start, end))

        # Account for [CLS] and [SEP] with "- 2".
        special_tokens_count = 2
        if len(tokens) > config.max_seq_len - special_tokens_count:
            tokens = tokens[: (config.max_seq_len - special_tokens_count)]
            start_ids = start_ids[: (config.max_seq_len - special_tokens_count)]
            end_ids = end_ids[: (config.max_seq_len - special_tokens_count)]
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        sep_token = "[SEP]"
        cls_token = "[CLS]"
        pad_token = 0
        pad_token_segment_id = 0
        sequence_a_segment_id = 0
        cls_token_segment_id = 0

        tokens += [sep_token]
        start_ids += [0]
        end_ids += [0]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        tokens = [cls_token] + tokens
        start_ids = [0] + start_ids
        end_ids = [0] + end_ids
        segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1] * len(input_ids)
        input_len = len(input_ids)
        # Zero-pad up to the sequence length.
        padding_length = config.max_seq_len - len(input_ids)

        input_ids += [pad_token] * padding_length
        input_mask += [0] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length
        start_ids += ([0] * padding_length)
        end_ids += ([0] * padding_length)

        # assert len(input_ids) == config.max_seq_len
        # assert len(input_mask) == config.max_seq_len
        # assert len(segment_ids) == config.max_seq_len
        # assert len(start_ids) == config.max_seq_len
        # assert len(end_ids) == config.max_seq_len

        # print("*** Example ***")
        # print("tokens: {}".format(text))
        # print("tokens: {}".format(tokens))
        # print("tokens: {}".format(" ".join([str(x) for x in tokens])))
        # print("input_ids: {}".format(" ".join([str(x) for x in input_ids])))
        # print("input_mask: {}".format(" ".join([str(x) for x in input_mask])))
        # print("segment_ids: {}".format(" ".join([str(x) for x in segment_ids])))
        # print("start_ids: {}".format(" ".join([str(x) for x in start_ids])))
        # print("end_ids: {}".format(" ".join([str(x) for x in end_ids])))

        # x_tensor = torch.tensor(input_ids, dtype=torch.long), \
        #            torch.tensor(input_mask, dtype=torch.long), \
        #            torch.tensor(segment_ids, dtype=torch.long), \
        #            torch.tensor(input_len, dtype=torch.long)
        # y_tensor = torch.tensor(start_ids, dtype=torch.long), \
        #            torch.tensor(end_ids, dtype=torch.long), \
        #            subjects_id
        x_tensor = input_ids, input_mask, segment_ids, input_len,
        y_tensor = start_ids, end_ids, subjects_id
        return x_tensor, y_tensor

    def __len__(self):
        return len(self.y_data)


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    x_data = [item[0] for item in batch]
    y_data = [item[1] for item in batch]

    input_lens = [x[3] for x in x_data]
    max_len = max(input_lens)

    input_ids = [x[0][:max_len] for x in x_data]
    input_mask = [x[1][:max_len] for x in x_data]
    segment_ids = [x[2][:max_len] for x in x_data]
    start_ids = [x[0][:max_len] for x in y_data]
    end_ids = [x[1][:max_len] for x in y_data]

    x_tensor = torch.tensor(input_ids, dtype=torch.long), \
               torch.tensor(input_mask, dtype=torch.long), \
               torch.tensor(segment_ids, dtype=torch.long), \
               torch.tensor(input_lens, dtype=torch.long)
    y_tensor = torch.tensor(start_ids, dtype=torch.long), \
               torch.tensor(end_ids, dtype=torch.long)
    return x_tensor, y_tensor


if __name__ == "__main__":
    pass
