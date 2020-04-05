# coding=utf-8
# author=yphacker

import json
import pandas as pd


def get_train_data():
    items = []
    with open('../data/train.json', 'r') as infile:
        for line in infile:
            item = json.loads(line.strip())
            print(item)
            items.append(item)

    with open('../data/dev.json', 'r') as infile:
        for line in infile:
            item = json.loads(line.strip())
            items.append(item)

    df = pd.DataFrame(items)
    print(df.head())
    df['label'] = df['label'].apply(lambda x: json.dumps(x))
    df.to_csv('../data/train.csv', index=None)

    max_len = 0
    df_len_list = []
    texts = df['text'].to_list()
    for i in range(len(texts)):
        text = texts[i]
        df_len_list.append(len(text))
        max_len = max(max_len, len(text))
    print(max_len)
    sorted_len = sorted(df_len_list)
    print(sorted_len[int(len(sorted_len) * 0.999)])


def get_test_data():
    items = []
    with open('../data/test.json', 'r') as infile:
        for line in infile:
            item = json.loads(line.strip())
            print(item)
            items.append(item)

    test = pd.DataFrame(items)
    print(test.head())
    test.to_csv('../data/test.csv', index=None)


if __name__ == '__main__':
    # get_train_data()
    get_test_data()
