from gensim.models import word2vec

# 引入日志配置
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 引入数据集
raw_sentences = ["the quick brown fox jumps over the lazy dogs", "yoyoyo you go home now to sleep"]

# 切分词汇
sentences = [s.encode('utf-8').split() for s in raw_sentences]

# 构建模型
model = word2vec.Word2Vec(sentences, min_count=1)

# 进行相关性比较
model.similarity('dogs', 'you')
