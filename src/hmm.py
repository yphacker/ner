# coding=utf-8
# author=yphacker

import gc
import os
import time
import json
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from importlib import import_module
from conf import config
from utils.metrics_utils import get_score
from utils.data_utils import load_vocab

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


def get_inputs(batch_x, batch_y=None):
    input_ids, attention_mask, token_type_ids = batch_x
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    token_type_ids = token_type_ids.to(device)
    if batch_y is not None:
        batch_y = batch_y.to(device)
    return dict(input_ids=input_ids, attention_mask=attention_mask,
                token_type_ids=token_type_ids, labels=batch_y)


class MyDataset(Dataset):

    def __init__(self, df, mode='train'):
        self.mode = mode
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
            label_entities = row.get('label', None)
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

        x_data = [self.word2id[word] for word in x_data]
        y_data = [config.label2id[label] for label in y_data]
        return x_data, y_data

    def __len__(self):
        return len(self.y_data)


def train(train_data, val_data, fold_idx=None):
    train_dataset = MyDataset(train_data)
    val_dataset = MyDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    from models.hmm import HMM
    word2id, id2word = load_vocab()
    model = HMM(len(config.label2id), len(word2id))

    if fold_idx is None:
        print('start')
        model_save_path = os.path.join(config.model_path, '{}.bin'.format(model_name))
    else:
        print('start fold: {}'.format(fold_idx + 1))
        model_save_path = os.path.join(config.model_path, '{}_fold{}.bin'.format(model_name, fold_idx))

    word_id_list = train_dataset.x_data
    label_id_list = train_dataset.y_data
    model.train(word_id_list, label_id_list)

    y_pred_list = model.predict(train_dataset.x_data)
    train_score = get_score(train_dataset.y_data, y_pred_list)
    y_pred_list = model.predict(val_dataset.x_data)
    val_score = get_score(val_dataset.y_data, y_pred_list)
    msg = 'train score: {0:>6.2%}, val score: {1:>6.2%}'
    print(msg.format(train_score, val_score))
    # train score: 53.84%, val score: 45.80%

    # del model
    # gc.collect()
    #
    # if fold_idx is not None:
    #     model_score[fold_idx] = best_val_score


def eval():
    pass


def predict():
    pass
    # model = x.Model().to(device)
    # model.load_state_dict(torch.load(model_config.model_save_path))
    # model.eval()
    # test_df = pd.read_csv(config.test_path)
    # # submission = pd.read_csv(config.sample_submission_path)
    #
    # test_dataset = MyDataset(test_df, 'test')
    # test_iter = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    #
    # results = []
    # with torch.no_grad():
    #     for batch_x, _ in test_iter:
    #         inputs = get_inputs(batch_x)
    #         outputs = model(**inputs)
    #         logits = outputs[0]
    #         # preds = np.argmax(preds, axis=2).tolist()
    #         preds = torch.argmax(logits, dim=1)
    #         preds = preds.cpu().data.numpy()[0][1:-1]  # [CLS]XXXX[SEP]
    #         tags = [args.id2label[x] for x in preds]
    #         label_entities = get_entities(preds, config.id2label)
    #         json_d = {}
    # #         json_d['id'] = step
    # #         json_d['tag_seq'] = " ".join(tags)
    # #         json_d['entities'] = label_entities
    # #         results.append(json_d)
    # # print(" ")
    # # output_predic_file = os.path.join(pred_output_dir, prefix, "test_prediction.json")
    # # output_submit_file = os.path.join(pred_output_dir, prefix, "test_submit.json")
    # # with open(output_predic_file, "w") as writer:
    # #     for record in results:
    # #         writer.write(json.dumps(record) + '\n')
    # # test_text = []
    # # with open(os.path.join(args.data_dir,"test.json"), 'r') as fr:
    # #     for line in fr:
    # #         test_text.append(json.loads(line))
    # # test_submit = []
    # # for x, y in zip(test_text, results):
    # #     json_d = {}
    # #     json_d['id'] = x['id']
    # #     json_d['label'] = {}
    # #     entities = y['entities']
    # #     words = list(x['text'])
    # #     if len(entities) != 0:
    # #         for subject in entities:
    # #             tag = subject[0]
    # #             start = subject[1]
    # #             end = subject[2]
    # #             word = "".join(words[start:end + 1])
    # #             if tag in json_d['label']:
    # #                 if word in json_d['label'][tag]:
    # #                     json_d['label'][tag][word].append([start, end])
    # #                 else:
    # #                     json_d['label'][tag][word] = [[start, end]]
    # #             else:
    # #                 json_d['label'][tag] = {}
    # #                 json_d['label'][tag][word] = [[start, end]]
    # #     test_submit.append(json_d)
    # # json_to_text(output_submit_file,test_submit)


def main(op):
    if op == 'train':
        train_df = pd.read_csv(config.train_path)
        # train_df = train_df[:1000]
        if args.mode == 1:
            # x = train_df['comment_text'].values
            # # y = train_df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
            # y = train_df['toxic'].values
            # skf = StratifiedKFold(n_splits=config.n_splits, random_state=0, shuffle=True)
            # for fold_idx, (train_idx, val_idx) in enumerate(skf.split(x, y)):
            #     train(train_df.iloc[train_idx], train_df.iloc[val_idx], fold_idx)
            # score = 0
            # score_list = []
            # for fold_idx in range(config.n_splits):
            #     score += model_score[fold_idx]
            #     score_list.append('{:.4f}'.format(model_score[fold_idx]))
            # print('val score:{}, avg val score:{:.4f}'.format(','.join(score_list), score / config.n_splits))
            pass
        else:
            train_data, val_data = train_test_split(train_df, shuffle=True, random_state=0, test_size=0.1)
            print('train:{}, val:{}'.format(train_data.shape[0], val_data.shape[0]))
            train(train_data, val_data)
    elif op == 'eval':
        pass
    elif op == 'predict':
        predict()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chinese NER')
    parser.add_argument("-o", "--operation", default='train', type=str, help="operation")
    parser.add_argument("-b", "--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("-e", "--epochs_num", default=8, type=int, help="train epochs")
    parser.add_argument("-m", "--model", default='hmm', type=str, required=True, help="choose a model: hmm")
    parser.add_argument("-mode", "--mode", default=1, type=int, help="train mode")
    args = parser.parse_args()

    config.batch_size = args.batch_size
    config.epochs_num = args.epochs_num
    model_name = args.model

    model_file = import_module('models.{}'.format(model_name))

    model_score = dict()
    main(args.operation)
