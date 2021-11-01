import os, re
import json, yaml
from tqdm import tqdm
import numpy as np
import pandas as pd
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import config
from utils import get_logger

logger = None
if logger is None:
    logger = get_logger("bert-subject-extract")


mode = 0
maxlen = 128
learning_rate = 5e-5
min_learning_rate = 1e-5


config_path = os.path.join(config.chinese_bert_corpus_dir, 'bert_config.json')
checkpoint_path = os.path.join(config.chinese_bert_corpus_dir, 'bert_model.ckpt')
dict_path = os.path.join(config.chinese_bert_corpus_dir, 'vocab.txt')


token_dict = {}

with open(dict_path, 'r') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R


tokenizer = OurTokenizer(token_dict)

D = pd.read_csv(config.chinese_bert_subject_extraction_train_path, encoding='utf-8', header=None)
D = D[D[2] != u'其他']
classes = set(D[2].unique())


train_data = []
for t, c, n in zip(D[1], D[2], D[3]):
    train_data.append((t, c, n))


if not os.path.exists(config.chinese_bert_subject_extraction_random_order_filepath):
    random_order = np.arange(len(train_data)).tolist()
    np.random.shuffle(random_order)
    str_json_data = json.dumps(random_order, indent=4)
    with open(config.chinese_bert_subject_extraction_random_order_filepath, 'w') as f:
        f.write(str_json_data)
else:
    random_order = json.load(open(config.chinese_bert_subject_extraction_random_order_filepath))


dev_data = [train_data[j] for i, j in enumerate(random_order) if i % 9 == mode]
train_data = [train_data[j] for i, j in enumerate(random_order) if i % 9 != mode]
additional_chars = set()
for d in train_data + dev_data:
    additional_chars.update(re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', d[2]))

additional_chars.remove(u'，')


D = pd.read_csv(config.chinese_bert_subject_extraction_test_path, encoding='utf-8', header=None)
test_data = []
for _id, t, c in zip(D[0], D[1], D[2]):
    test_data.append((_id, t, c))


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


def list_find(list1, list2):
    """在list1中寻找子串list2，如果找到，返回第一个下标；
    如果找不到，返回-1。
    """
    n_list2 = len(list2)
    for i in range(len(list1)):
        if list1[i: i + n_list2] == list2:
            return i
    return -1


class DataGenerator(object):
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = np.arange(len(self.data)).tolist()
            np.random.shuffle(idxs)
            X1, X2, S1, S2 = [], [], [], []
            for i in idxs:
                d = self.data[i]
                text, c = d[0][:maxlen], d[1]
                text = u'___%s___%s' % (c, text)
                tokens = tokenizer.tokenize(text)
                e = d[2]
                e_tokens = tokenizer.tokenize(e)[1:-1]
                s1, s2 = np.zeros(len(tokens)), np.zeros(len(tokens))
                start = list_find(tokens, e_tokens)
                if start != -1:
                    end = start + len(e_tokens) - 1
                    s1[start] = 1
                    s2[end] = 1
                    x1, x2 = tokenizer.encode(first=text)
                    X1.append(x1)
                    X2.append(x2)
                    S1.append(s1)
                    S2.append(s2)
                    if len(X1) == self.batch_size or i == idxs[-1]:
                        X1 = seq_padding(X1)
                        X2 = seq_padding(X2)
                        S1 = seq_padding(S1)
                        S2 = seq_padding(S2)
                        yield [X1, X2, S1, S2], None
                        X1, X2, S1, S2 = [], [], [], []


bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

for l in bert_model.layers:
    l.trainable = True


x1_in = Input(shape=(None,)) # 待识别句子输入(单词编码）
x2_in = Input(shape=(None,)) # 待识别句子输入(段编码)
s1_in = Input(shape=(None,)) # 实体左边界（标签）
s2_in = Input(shape=(None,)) # 实体右边界（标签）

x1, x2, s1, s2 = x1_in, x2_in, s1_in, s2_in

# 一般来说NLP模型的输入是词ID矩阵，形状为[batch_size, seq_len]，其中我会用0作为padding的ID，而1作为UNK的ID，剩下的就随意了，然后我就用一个Lambda层生成mask矩阵：
# 这样生成的mask矩阵大小是[batch_size, seq_len, 1]，然后词ID矩阵经过Embedding层后的大小为[batch_size, seq_len, word_size]
x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x1)

x = bert_model([x1, x2])

# Dense的用法：
# 输入尺寸
# nD 张量，尺寸: (batch_size, …, input_dim)。 最常见的情况是一个尺寸为 (batch_size, input_dim) 的 2D 输入。
# 输出尺寸
# nD 张量，尺寸: (batch_size, …, units)。 例如，对于尺寸为 (batch_size, input_dim) 的 2D 输入， 输出的尺寸为 (batch_size, units)。
ps1 = Dense(1, use_bias=False)(x)
ps1 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([ps1, x_mask])
ps2 = Dense(1, use_bias=False)(x)
ps2 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([ps2, x_mask])

model = Model([x1_in, x2_in], [ps1, ps2])

train_model = Model([x1_in, x2_in, s1_in, s2_in], [ps1, ps2])

loss1 = K.mean(K.categorical_crossentropy(s1_in, ps1, from_logits=True))
#为了让ps2和ps1错开，尽量确保ps2 > ps1， ps2在s1左边值最小， s1到s2之间值中等， 大于等于s2的位置值最大，
#这样与交叉熵损失函数相对应，即 ps2值越大，则损失越小，尽量加大ps2 < ps1的惩罚，鼓励ps2 >= ps1的情况
ps2 -= (1 - K.cumsum(s1, 1)) * 1e10
loss2 = K.mean(K.categorical_crossentropy(s2_in, ps2, from_logits=True))
loss = loss1 + loss2

train_model.add_loss(loss)
train_model.compile(optimizer=Adam(learning_rate))
train_model.summary()


def soft_max(x):
    x = x - np.max(x)
    x = np.exp(x)
    return x / np.sum(x)


def extract_entity(text_in, c_in):
    if c_in not in classes:
        return 'NaN'
    text_in = u'___%s___%s' % (c_in, text_in)
    text_in = text_in[: 510]
    _tokens = tokenizer.tokenize(text_in)
    _x1, _x2 = tokenizer.encode(first=text_in)
    _x1, _x2 = np.array([_x1]), np.array([_x2])
    _ps1, _ps2 = model.predict([_x1, _x2])
    _ps1, _ps2 = soft_max(_ps1[0]), soft_max(_ps2[0])
    for i, _t in enumerate(_tokens):
        if len(_t) == 1 and re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', _t) and _t not in additional_chars:
            _ps1[i] -= 10
    start = _ps1.argmax()
    for end in range(start, len(_tokens)):
        _t = _tokens[end]
        if len(_t) == 1 and re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', _t) and _t not in additional_chars:
            break
    end = _ps2[start: end + 1].argmax() + start
    a = text_in[start - 1: end]
    return a


class Evaluate(Callback):
    def __init__(self):
        self.ACC = []
        self.best = 0.
        self.passed = 0

    def on_batch_begin(self, batch, logs=None):
        """第一个epoch用来warmup，第二个epoch把学习率降到最低
        """
        if self.passed < self.params['steps']:
            lr = (self.passed + 1.) / self.params['steps'] * learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            lr = (2 - (self.passed + 1.) / self.params['steps']) * (learning_rate - min_learning_rate)
            lr += min_learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1

    def on_epoch_end(self, epoch, logs=None):
        acc = self.evaluate()
        self.ACC.append(acc)
        if acc > self.best:
            self.best = acc
            with open(config.bert_ner_model_path[0], 'w') as outfile:
                outfile.write(yaml.dump(model.to_yaml(), default_flow_style=True))

            train_model.save_weights(config.bert_ner_model_path[1])
        logger.info('acc: %.4f, best acc: %.4f\n' % (acc, self.best))

    @staticmethod
    def evaluate():
        A = 1e-10
        F = open(config.chinese_bert_subject_extraction_dev_filepath, 'w')
        for d in tqdm(iter(dev_data)):
            R = extract_entity(d[0], d[1])
            if R == d[2]:
                A += 1
            F.write(', '.join(d + (R,)) + '\n')
        F.close()
        return A / len(dev_data)


def test(test_data):
    F = open(config.chinese_bert_subject_extraction_test_result_filepath, 'w')
    for d in tqdm(iter(test_data)):
        s = u'"%s","%s"\n' % (d[0], extract_entity(d[1], d[2]))
        s = s.encode('utf-8')
        F.write(s)
    F.close()


evaluator = Evaluate()
train_D = DataGenerator(train_data)


if __name__ == '__main__':
    train_model.fit_generator(train_D.__iter__(),
                              steps_per_epoch=len(train_D),
                              epochs=10,
                              callbacks=[evaluator]
                              )
else:
    train_model.load_weights(config.bert_ner_model_path[1])
