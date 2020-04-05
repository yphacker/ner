# coding=utf-8
# author=yphacker

import gc
import os
import time
import json
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from importlib import import_module
from conf import config
from utils.data_utils import MyDataset, collate_fn
from utils.metrics_utils import get_score
from utils.utils import get_entities

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


def get_inputs(batch_x, batch_y=None):
    batch_x = tuple(t.to(device) for t in batch_x)
    if batch_y is not None:
        batch_y = batch_y.to(device)
    if 'crf' in model_name:
        return dict(input_ids=batch_x[0], attention_mask=batch_x[1], input_lens=batch_x[2], labels=batch_y)
    else:
        return dict(input_ids=batch_x[0], attention_mask=batch_x[1], labels=batch_y)


def evaluate(model, val_iter):
    model.eval()
    data_len = 0
    total_loss = 0
    y_true_list = []
    y_pred_list = []
    with torch.no_grad():
        for batch_x, batch_y in tqdm(val_iter):
            batch_len = len(batch_y)
            data_len += batch_len
            inputs = get_inputs(batch_x, batch_y)
            outputs = model(**inputs)
            _loss, logits = outputs[:2]
            total_loss += _loss.item() * batch_len
            y_true_list += batch_y.cpu().data.numpy().tolist()
            if 'crf' in model_name:
                preds, _ = model.crf._obtain_labels(logits, config.id2label, inputs['input_lens'])
            else:
                preds = torch.argmax(logits, dim=2)
                preds = preds.cpu().data.numpy().tolist()
            y_pred_list += preds
    return total_loss / data_len, get_score(y_true_list, y_pred_list)


def train(train_data, val_data, fold_idx=None):
    train_dataset = MyDataset(train_data)
    val_dataset = MyDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, collate_fn=collate_fn)

    model = model_file.Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

    if fold_idx is None:
        print('start')
        model_save_path = os.path.join(config.model_path, '{}.bin'.format(model_name))
    else:
        print('start fold: {}'.format(fold_idx + 1))
        model_save_path = os.path.join(config.model_path, '{}_fold{}.bin'.format(model_name, fold_idx))

    best_val_score = 0
    last_improved_epoch = 0
    adjust_lr_num = 0
    y_true_list = []
    y_pred_list = []
    for cur_epoch in range(config.epochs_num):
        start_time = int(time.time())
        model.train()
        print('epoch:{}, step:{}'.format(cur_epoch + 1, len(train_loader)))
        cur_step = 0
        for batch_x, batch_y in tqdm(train_loader):
            inputs = get_inputs(batch_x, batch_y)
            optimizer.zero_grad()
            outputs = model(**inputs)
            train_loss, logits = outputs[:2]
            train_loss.backward()
            optimizer.step()

            cur_step += 1
            # crf比较慢，训练crf时，注释
            # y_true_list += batch_y.cpu().data.numpy().tolist()
            # if 'crf' in model_name:
            #     preds, _ = model.crf._obtain_labels(logits, config.id2label, inputs['input_lens'])
            # else:
            #     preds = torch.argmax(logits, dim=2)
            #     preds = preds.cpu().data.numpy().tolist()
            # y_pred_list += preds
            # if cur_step % config.train_print_step == 0:
            #     train_score = get_score(y_true_list, y_pred_list)
            #     msg = 'the current step: {0}/{1}, train loss: {2:>5.2}, train score: {3:>6.2%}'
            #     print(msg.format(cur_step, len(train_loader), train_loss.item(), train_score))
            #     y_true_list = []
            #     y_pred_list = []
        # 过滤前3个step, 训练太少，可能会有问题
        if cur_epoch <= 3:
            continue
        val_loss, val_score = evaluate(model, val_loader)
        if val_score >= best_val_score:
            best_val_score = val_score
            torch.save(model.state_dict(), model_save_path)
            last_improved_epoch = cur_epoch
            improved_str = '*'
        else:
            improved_str = ''
        # msg = 'the current epoch: {0}/{1}, train loss: {2:>5.2}, train acc: {3:>6.2%},  ' \
        #       'val loss: {4:>5.2}, val acc: {5:>6.2%}, {6}'
        msg = 'the current epoch: {0}/{1}, val loss: {2:>5.2}, val score: {3:>6.2%}, cost: {4}s {5}'
        end_time = int(time.time())
        print(msg.format(cur_epoch + 1, config.epochs_num, val_loss, val_score,
                         end_time - start_time, improved_str))
        if cur_epoch - last_improved_epoch >= config.patience_epoch:
            if adjust_lr_num >= model_config.adjust_lr_num:
                print("No optimization for a long time, auto stopping...")
                break
            print("No optimization for a long time, adjust lr...")
            scheduler.step()
            last_improved_epoch = cur_epoch  # 加上，不然会连续更新的
            adjust_lr_num += 1
    del model
    gc.collect()

    if fold_idx is not None:
        model_score[fold_idx] = best_val_score


def eval():
    pass


def predict():
    model = model_file.Model().to(device)
    model_save_path = os.path.join(config.model_path, '{}.bin'.format(model_name))
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    test_df = pd.read_csv(config.test_path)
    # submission = pd.read_csv(config.sample_submission_path)

    test_dataset = MyDataset(test_df, 'test')
    test_iter = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    results = []
    with torch.no_grad():
        for batch_x, _ in tqdm(test_iter):
            inputs = get_inputs(batch_x)
            outputs = model(**inputs)
            logits = outputs[0]
            preds = torch.argmax(logits, dim=2)
            for pred in preds:
                pred = pred.cpu().data.numpy()[1:-1]  # [CLS]XXXX[SEP]
                tags = [config.id2label[x] for x in pred]
                label_entities = get_entities(pred, config.id2label)
                pred_dict = dict()
                pred_dict['tag_seq'] = " ".join(tags)
                pred_dict['entities'] = label_entities
                results.append(pred_dict)
    test_text = []
    with open("../data/test.json", 'r') as fr:
        for line in fr:
            test_text.append(json.loads(line))
    submission = []
    for x, y in zip(test_text, results):
        item = dict()
        item['id'] = x['id']
        item['label'] = {}
        entities = y['entities']
        words = list(x['text'])
        if len(entities) != 0:
            for subject in entities:
                tag = subject[0]
                start = subject[1]
                end = subject[2]
                word = "".join(words[start:end + 1])
                if tag in item['label']:
                    if word in item['label'][tag]:
                        item['label'][tag][word].append([start, end])
                    else:
                        item['label'][tag][word] = [[start, end]]
                else:
                    item['label'][tag] = {}
                    item['label'][tag][word] = [[start, end]]
        submission.append(item)

    with open('submission.json', 'w') as outfile:
        for line in submission:
            line = json.dumps(line, ensure_ascii=False)
            outfile.write(line + '\n')


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
    parser.add_argument("-m", "--model", default='lstm_crf', type=str, required=True,
                        help="choose a model: lstm_crf")
    parser.add_argument("-mode", "--mode", default=1, type=int, help="train mode")
    args = parser.parse_args()

    config.batch_size = args.batch_size
    config.epochs_num = args.epochs_num
    model_name = args.model

    model_file = import_module('models.{}'.format(model_name))

    if model_name in ['bilstm_crf']:
        from conf import model_config_lstm as model_config

    model_score = dict()
    main(args.operation)
