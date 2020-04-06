# NER

## 数据
[CLUENER 细粒度命名实体识别](https://www.cluebenchmarks.com/introduce.html)

## score
|model|offline score|online score|note|
|:---:|:---:|:---:|:---:|
|hmm|45.80|||
|crf||||
|bilstm+crf|68.05|||
|bert|78.14||
|bert+crf|77.18|||
|bert+span|79.31|||

## script
### 训练
python main.py -m=bilstm_crf -b=200 -e=32 -mode=2
python main2.py -m=bert -b=200 -e=32 -mode=2
python main2.py -m=bert_crf -b=200 -e=32 -mode=2
python main3.py -m=bert_span -b=200 -e=32 -mode=2

### 预测
python main2.py -m=bert -o=predict
python main3.py -m=bert_span -o=predict -b=1

## 实验环境
Tesla P100
16G
cuda9  
python:3.6  
torch:1.2.0.dev20190722

## 代码目录说明
### hmm.py
hmm模型
### main.py
bilstn+crf
### main2.py
bert, bert+crf
### main3.py
bert+span

# 概念
BIO
B: 命名实体的起始 或 单个字命名实体
I: 命名实体的中间位置 或 结束位置
O：非命名实体

BIOES
B: 命名实体的起始标注
I: 命名实体的中间标注
E: 命名实体的结尾标注
O: 非命名实体
S: 单个字命名实体

## 参考文章
[1] [CLUENER2020](https://github.com/CLUEbenchmark/CLUENER2020)  
[2] [luopeixiang/named_entity_recognition](https://github.com/luopeixiang/named_entity_recognition)  

## 参考代码
[1] [Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/abs/1508.01991)  
[2] [Neural Architectures for Named Entity Recognition](https://arxiv.org/pdf/1603.01360.pdf)  
[3] [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf) 