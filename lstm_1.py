from __future__ import absolute_import
from __future__ import print_function

import pandas as pd
import numpy as np
import jieba
from gensim.corpora import Dictionary
from gensim.models import word2vec, KeyedVectors
from keras import callbacks
from keras.optimizers import Adam
from keras.preprocessing import sequence, text
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.utils import plot_model
from sklearn.model_selection import train_test_split

np.random.seed(7)
# set parameters:
MAX_SEQUENCE_LENGTH = 100  # 每条语句最大长度
EMBEDDING_DIM = 200  # 词向量空间维度
window_size = 7
batch_size = 128


def load_data():
    neg = pd.read_excel("../dataset/data/neg.xls", header=None, index=None)
    pos = pd.read_excel("../dataset/data/pos.xls", header=None, index=None)
    data = np.concatenate((pos[0], neg[0]))
    labels = np.concatenate((np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)))
    # print(data[:-3])
    # print(labels[:-3])
    return data, labels


def process_data():
    input_texts, labels = load_data()
    # texts = []
    # for doc in input_texts:
    #     seg_doc = jieba.lcut(doc.replace('\n', ''))
    #     d = ' '.join(seg_doc)
    #     texts.append(d)
    # tokenizer = text.Tokenizer()
    # tokenizer.fit_on_texts(texts)
    texts = cut_word(input_texts)
    tokenizer = tokenizer_fit()
    data = text2seq(texts)
    # text_sequences = tokenizer.texts_to_sequences(texts)
    # data = sequence.pad_sequences(text_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    input_dim = len(tokenizer.word_index) + 1
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
    return x_train, x_test, y_train, y_test, input_dim


def train():
    x_train, x_test, y_train, y_test, input_dim = process_data()
    print('Build model...')
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=200, input_length=MAX_SEQUENCE_LENGTH))
    model.add(LSTM(256, activation='relu'))
    model.add(Dropout(0.3))
    # model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    sgd = Adam(lr=0.0003)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['acc'])
    callback = callbacks.TensorBoard(log_dir='../logs', histogram_freq=0, write_graph=True, write_images=True)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=5, validation_data=(x_test, y_test),
              callbacks=[callback])
    score = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("Score:",
          score)  # [0.25330617905265534, 0.90840176860516542] Score: [0.34134976900151998, 0.91163231447617943]
    model.save("../models/lstm_1.h5")


def create_dictionaries(model=None, combined=None):
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
        w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 所有频数超过10的词语的索引
        w2vec = {word: model[word] for word in w2indx.keys()}  # 所有频数超过10的词语的词向量

        def parse_dataset(combined):
            data = []
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data

        combined = parse_dataset(combined)
        # 每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        combined = sequence.pad_sequences(combined, maxlen=MAX_SEQUENCE_LENGTH)
        return w2indx, w2vec, combined
    else:
        print('No data')


def build_w2v():
    neg = pd.read_excel("../dataset/data/neg.xls", header=None, index=None)
    pos = pd.read_excel("../dataset/data/pos.xls", header=None, index=None)
    seg_data = np.concatenate((pos[0], neg[0]))
    segs = [jieba.lcut(document.replace('\n', '')) for document in seg_data]
    model = word2vec.Word2Vec(size=EMBEDDING_DIM, window=window_size, iter=1, min_count=5)
    model.build_vocab(segs)
    model.train(segs, total_examples=model.corpus_count, epochs=model.iter)
    model.save('../models/Word2vec_model.model')


def text2seq(input_texts):
    tokenizer = tokenizer_fit()
    seq_texts = cut_word(input_texts)
    sequences = tokenizer.texts_to_sequences(seq_texts)
    return sequence.pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)


def tokenizer_fit():
    inputs, labels = load_data()
    train_texts = cut_word(inputs)
    tokenizer = text.Tokenizer()
    tokenizer.fit_on_texts(train_texts)
    return tokenizer


def cut_word(inputs):
    texts = []
    for x in inputs:
        seg_doc = jieba.lcut(x.replace('\n', ''))
        d = ' '.join(seg_doc)
        texts.append(d)
    return texts


def predict():
    input_texts = ['今天天气不错，真高兴！', '质量渣的不行，下次不来了！！', '这是我见过性价比最高的产品', '配件太少，还需要自己买',
                   '不但手机的造型美观,轻颖，而且手感良好!!',
                   '安装费太贵了，简直不能接受', '5分好评，推荐购买']
    # sentences = [jieba.lcut(document.replace('\n', '')) for document in input_texts]
    # word_model = word2vec.Word2Vec.load('./models/news_12g_baidubaike_20g_novel_90g_embedding_64.model')
    # w2indx, w2vec, texts1 = create_dictionaries(word_model, sentences)
    # print(texts1[:2])

    seq = text2seq(input_texts)

    print('加载模型')
    m = load_model("../models/lstm_1.h5")
    # plot_model(m, to_file='../models/lstm.png')
    print('开始预测')
    r = m.predict(seq, batch_size=batch_size)
    print(r)
    labels = [int(round(x[0])) for x in r]
    dic = {1: "正面", 0: "负面"}
    for x in range(len(input_texts)):
        print('{}===>{}'.format(input_texts[x], dic[labels[x]]))


if __name__ == '__main__':
    # load_data()
    # train()
    # build_w2v()
    predict()
