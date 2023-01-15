# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 20:55:03 2023

@author: lihan
"""
from transformers import BertConfig
BertConfig.from_yaml_file()
from transformers import BertModel


from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import BertTokenizer
from transformers import pipeline
import tensorflow.keras.optimizers as optim
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import tensorflow as tf
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import time
from sklearn.model_selection import GridSearchCV
import logging
from gensim.models import word2vec
import gensim.downloader
import gensim
import matplotlib.patches as mpatches
import matplotlib
from sklearn.decomposition import PCA, TruncatedSVD
from nltk.tokenize import RegexpTokenizer
import nltk
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score
import jieba
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re
os.chdir(r"D:\documents\FDSM\2022-2023-1\Data Mining & Machine Learning\期末报告\dataset")


train = pd.read_csv(r'.\train\data.txt', header=None, sep=',')
test = pd.read_csv(r'.\test\test.txt', header=None, sep='\t')
train.columns = ['label', 'sentence']
test.columns = ['sentence']


train['segment'] = train['sentence'].apply(lambda w: ' '.join(jieba.cut(w)))
test['segment'] = test['sentence'].apply(lambda w: ' '.join(jieba.cut(w)))

train['label'].value_counts()

w = train.iloc[0, 1]
jieba.lcut(w)
x_train = train['segment'].iloc[:100]

vectorizer = CountVectorizer(max_df=0.9, min_df=1e-3)
tfidf_trans = TfidfTransformer()

train_vector = vectorizer.fit_transform(train['segment'])
test_vector = vectorizer.transform(test['segment'])

words = vectorizer.get_feature_names_out()
print(len(words), words[:10])
train_tfidf = tfidf_trans.fit_transform(train_vector)
test_tfidf = tfidf_trans.transform(test_vector)
X = train_tfidf.toarray()
print(X.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X, train['label'], train_size=0.8, test_size=0.2, random_state=0)

####-----------------------------------------------------------------------####
##                                Model 1: XGB                               ##
####-----------------------------------------------------------------------####

xgb = XGBClassifier()
xgb.fit(X_train, y_train)

# 预测
y_pred = xgb.predict(X_test)
print("AUC = %.4f, Accuracy = %.4g, f1-score = %.4g" % (roc_auc_score(y_test,
      y_proba), accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)))
# AUC = 0.9064, Accuracy = 0.8394, f1-score = 0.7477
print(classification_report(y_test, y_pred))
y_train_proba = xgb.predict_proba(X_train)[:, 1]
print("AUC Score (Train): %f" % roc_auc_score(y_train, y_train_proba))
y_proba = xgb.predict_proba(X_test)[:, 1]
plt.hist(y_proba)
print("AUC Score (Test): %f" % roc_auc_score(y_test, y_proba))


y_true_pred = xgb.predict(test_tfidf.toarray())
pd.DataFrame(y_true_pred).to_csv(
    r'.\test\comment_result.txt', header=False, index=0)


####-----------------------------------------------------------------------####
##                                Model 2: lr                                ##
####-----------------------------------------------------------------------####


lr = LogisticRegression()
lr.fit(X_train, y_train)

# 预测
y_pred = lr.predict(X_test)
y_proba = lr.predict_proba(X_test)[:, 1]
plt.hist(y_proba)
print("AUC = %.4f, Accuracy = %.4g, f1-score = %.4g" % (roc_auc_score(y_test,
      y_proba), accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)))
# AUC = 0.9129, Accuracy = 0.8517, f1-score = 0.7638
print(classification_report(y_test, y_pred))
y_train_proba = lr.predict_proba(X_train)[:, 1]
print("AUC Score (Train): %f" % roc_auc_score(y_train, y_train_proba))


y_true_pred = lr.predict(test_tfidf.toarray())
pd.DataFrame(y_true_pred).to_csv(
    r'.\test\comment_result_lr.txt', header=False, index=0)


####-----------------------------------------------------------------------####
##                              Model 3: LSTM                                ##
####-----------------------------------------------------------------------####

train['tokens'] = train['sentence'].apply(lambda w: ' '.join(jieba.lcut(w)))
test['tokens'] = test['sentence'].apply(lambda w: ' '.join(jieba.lcut(w)))

all_words = [word for tokens in train["tokens"] for word in tokens]
sentence_lengths = [len(tokens) for tokens in train["tokens"]]

VOCAB = sorted(list(set(all_words)))
print("%s words total, with a vocabulary size of %s" %
      (len(all_words), len(VOCAB)))
print("Max sentence length is %s" % max(sentence_lengths))

t = train[np.array(sentence_lengths) == max(sentence_lengths)]

len_plt = pd.DataFrame(
    {'sentence_lengths': sentence_lengths, 'label': train['label']})

sns.distplot(x=len_plt['sentence_lengths'][len_plt['label']
             == 0], kde=True, bins=15, label='label = 0')
sns.distplot(x=len_plt['sentence_lengths'][len_plt['label']
             == 1], kde=True, bins=15, label='label = 1')
plt.legend()
plt.xlabel('sentence_lengths')
plt.tight_layout()

pd.DataFrame(sentence_lengths).describe()
np.percentile(sentence_lengths, [10, 90, 95, 98, 99])

EMBEDDING_DIM = 25
MAX_SEQUENCE_LENGTH = max(sentence_lengths)
VOCAB_SIZE = len(VOCAB)


VALIDATION_SPLIT = .2
tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(train["tokens"].tolist())
sequences = tokenizer.texts_to_sequences(train["tokens"].tolist())
sequences_real_test = tokenizer.texts_to_sequences(test["tokens"].tolist())


word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


sequence_len = 100
nn_train = pad_sequences(sequences, maxlen=sequence_len)
nn_real_test = pad_sequences(sequences_real_test, maxlen=sequence_len)


# embedding_weights = np.zeros((len(word_index)+1, EMBEDDING_DIM))
# for word, index in word_index.items():
#     embedding_weights[index, :] = glove_model[word] if word in glove_model else np.random.rand(
#         EMBEDDING_DIM)
# print(embedding_weights.shape)


x_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
    nn_train, train["label"], train_size=0.8, test_size=0.2, random_state=1)


def lstm(sequence_len, num_words, embedding_dim,  lr=1e-4):

    embedding_layer = Embedding(num_words,
                                embedding_dim,
                                input_length=sequence_len)

    sequence_input = Input(shape=(sequence_len,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    middle = LSTM(128, activation='relu')(embedded_sequences)
    middle = Dropout(0.2)(middle)
    # middle=BatchNormalization()(middle)
    middle = Dense(128, activation='relu')(middle)
    middle = Dense(128, activation='relu')(middle)
    middle = Dropout(0.2)(middle)
    preds = Dense(1, activation='sigmoid')(middle)

    model = Model(sequence_input, preds)
    optimizer = optim.Adam(learning_rate=lr, decay=1e-6)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])

    return model


lr = 1e-3
epochs = 6
batch_size = 64
EMBEDDING_DIM = 20
tf.random.set_seed(1)
model = lstm(sequence_len, len(word_index)+1, EMBEDDING_DIM, lr=lr)
model.summary()
history = model.fit(x_train_lstm, y_train_lstm,
                    validation_split=0.2, epochs=epochs, batch_size=batch_size, shuffle=True)


def PlotTrainingProcess(history, num, epochs=epochs):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs2 = range(1, epochs+1)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121)
    ax.plot(epochs2, train_loss, 'bo', label='Training Loss')
    ax.plot(epochs2, val_loss, 'b', label='Validation Loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    ax2 = fig.add_subplot(122)
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    ax2.plot(epochs2, acc, 'bo', label='Training Acc')
    ax2.plot(epochs2, val_acc, 'b', label='Validation Acc')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    # plt.savefig(r"..\pics\lstm_"+str(num)+".png", dpi=150)
    return 0


PlotTrainingProcess(history, 'final2', epochs=epochs)


y_proba_lstm = model.predict(X_test_lstm)
y_pred_lstm = 0+(y_proba_lstm > 0.5)
plt.figure()
plt.hist(y_proba_lstm)
print("AUC = %.4f, Accuracy = %.4g, f1-score = %.4g" % (roc_auc_score(y_test_lstm,
      y_proba_lstm), accuracy_score(y_test_lstm, y_pred_lstm), f1_score(y_test_lstm, y_pred_lstm)))
# AUC = 0.9051, Accuracy = 0.838, f1-score = 0.7721
# AUC = 0.9200, Accuracy = 0.8667, f1-score = 0.8056
print(classification_report(y_test_lstm, y_pred_lstm))


y_true_pred = 0+(model.predict(nn_real_test) > 0.5)
pd.DataFrame(y_true_pred).to_csv(
    r'.\test\comment_result_lstm.txt', header=False, index=0)


####-----------------------------------------------------------------------####
##                              Model 4: BERT                                ##
####-----------------------------------------------------------------------####

X_train_forbert, X_test_forbert, y_train_forbert, y_test_forbert = train_test_split(train["sentence"].values,
                                                                                    train["label"].values, test_size=0.2, stratify=train["label"], random_state=0)

# transformers bert相关的模型使用和加载
# 分词器，词典

tokenizer_bert = BertTokenizer.from_pretrained('bert-base-chinese')
train_encoding = tokenizer_bert(
    list(X_train_forbert), truncation=True, padding=True, max_length=64)
test_encoding = tokenizer_bert(
    list(X_test_forbert), truncation=True, padding=True, max_length=64)
real_test_encoding = tokenizer_bert(
    list(test["sentence"].values), truncation=True, padding=True, max_length=64)

# 数据集读取


class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    # 读取单个样本
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = NewsDataset(train_encoding, y_train_forbert)
train_dataset.__getitem__(3)
test_dataset = NewsDataset(test_encoding, y_test_forbert)
real_test_dataset = NewsDataset(real_test_encoding, [1]*test.shape[0])

model = BertForSequenceClassification.from_pretrained(
    'bert-base-chinese', num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 单个读取到批量读取
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
real_test_loader = DataLoader(real_test_dataset, batch_size=batch_size, shuffle=False)

# 优化方法
optim = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * 1
scheduler = get_linear_schedule_with_warmup(optim,
                                            num_warmup_steps=0,  # Default value in run_glue.py
                                            num_training_steps=total_steps)


# 训练函数
def train_bert():
    model.train()
    total_train_loss = 0
    iter_num = 0
    total_iter = len(train_loader)
    for batch in train_loader:
        # 正向传播
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(
            input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        total_train_loss += loss.item()

        # 反向梯度信息
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 参数更新
        optim.step()
        scheduler.step()

        iter_num += 1
        if(iter_num % 100 == 0):
            print("epoth: %d, iter_num: %d, loss: %.4f, %.2f%%" %
                  (epoch, iter_num, loss.item(), iter_num/total_iter*100))

    print("Epoch: %d, Average training loss: %.4f" %
          (epoch, total_train_loss/len(train_loader)))

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def validation_bert():
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    for batch in test_loader:
        with torch.no_grad():
            # 正常传播
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(
                input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs[0]
        logits = outputs[1]

        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)

    avg_val_accuracy = total_eval_accuracy / len(test_loader)
    print("Accuracy: %.4f" % (avg_val_accuracy))
    print("Average testing loss: %.4f" %
          (total_eval_loss/len(test_loader)))
    print("-------------------------------")


for epoch in range(1):
    print("------------Epoch: %d ----------------" % epoch)
    train_bert()
    validation_bert()
    
# Accuracy: 0.9050
    
    
model.eval()
total_eval_accuracy = 0
total_eval_loss = 0
real_y_pred = []
for batch in real_test_loader:
    with torch.no_grad():
        # 正常传播
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(
            input_ids, attention_mask=attention_mask, labels=labels)

    loss = outputs[0]
    logits = outputs[1]

    total_eval_loss += loss.item()
    real_y_proba = logits.detach().cpu().numpy()
    real_y_pred .extend( 0+(real_y_proba[:,1]>0.5))
    

    
pd.DataFrame(real_y_pred).to_csv(
      r'.\test\comment_result_bert.txt', header=False, index=0)  
    
    
    
####-----------------------------------------------------------------------####
##                                MindSpore                                  ##
####-----------------------------------------------------------------------####
    
    
    
    
    
    
    


####-----------------------------------------------------------------------####
##                                  draft                                    ##
####-----------------------------------------------------------------------####
classifier = pipeline('sentiment-analysis')
classifier(
    'We are very happy to introduce pipeline to the transformers repository.')

x_train = ['TF-IDF 主要 思想 是', '算法 一个 重要 特点 可以 脱离 语料库 背景',
           '如果 一个 网页 被 很多 其他 网页 链接 说明 网页 重要']
# 将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
vectorizer = CountVectorizer(max_features=10)
# 计算每一个词语的TF-IDF权值
tf_idf_transformer = TfidfTransformer()
# 计算每一个词语出现的次数#将文本转换为词频并计算tf-idf;fit_transform()方法用于计算每一个词语出现的次数
X = vectorizer.fit_transform(x_train)
tf_idf = tf_idf_transformer.fit_transform(X)
# 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
x_train_weight = tf_idf.toarray()
print('输出X_train文本变量：')
print(x_train_weight)


vectorizer = CountVectorizer()  # 实例化
transformer = TfidfTransformer()
corpus = ["我 来到 中国 旅游", "中国 欢迎 你", "我 喜欢 来到 中国 天安门"]
result_list2 = transformer.fit_transform(
    vectorizer.fit_transform(corpus)).toarray().tolist()
word = vectorizer.get_feature_names()
print('词典为：')
print(word)
print('归一化后的tf-idf值为：')
for weight in result_list2:
    print(weight)
