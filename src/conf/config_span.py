# coding=utf-8
# author=yphacker


import os

conf_path = os.path.dirname(os.path.abspath(__file__))
work_path = os.path.dirname(os.path.dirname(conf_path))
data_path = os.path.join(work_path, "data")
# word_embedding_path = os.path.join(data_path, "glove.840B.300d.txt")
pretrain_model_path = os.path.join(data_path, "pretrain_model")
# pretrain_embedding_path = os.path.join(data_path, "pretrain_embedding.npz")
vocab_path = os.path.join(data_path, "vocab.pkl")

train_path = os.path.join(data_path, 'train.csv')
test_path = os.path.join(data_path, 'test.csv')
# sample_submission_path = os.path.join(data_path, 'sample_submission.csv')

model_path = os.path.join(data_path, "model")
submission_path = os.path.join(data_path, "submission")
for path in [model_path, submission_path]:
    if not os.path.isdir(path):
        os.makedirs(path)

# pretrain_embedding = False
# # pretrain_embedding = True
# embed_dim = 300
# num_vocab = 41530
# tokenizer = lambda x: x.split(' ')[:max_seq_len]
# padding_idx = 0

loss_type = 'ce'
max_seq_len = 52
batch_size = 32
epochs_num = 2

n_splits = 5
train_print_step = 50
patience_epoch = 3

label_list = ["O", "address", "book", "company", 'game', 'government',
              'movie', 'name', 'organization', 'position', 'scene']
num_labels = len(label_list)
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}
