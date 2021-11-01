import re
import numpy as np
import jieba
from keras.models import load_model as keras_load_model
from keras.preprocessing import sequence

from tornado.log import app_log
from sentiment import word2vec_model
from sentiment.models.model import Model
from sentiment.models.model import ExternalLabel
from utils import get_stopwords


class CategoryModel(Model):
    LABEL_MAP = {
        0: '科技',
        1: '娱乐',
        2: '游戏',
        3: '时尚',
        4: '财经',
        5: '教育',
        6: '房产',
        7: '时政',
        8: '体育',
        9: '家居',
    }

    MAXLEN = 100
    _stopwords = get_stopwords()

    @staticmethod
    def load_cnn_model(path):
        try:
            if isinstance(path, list):
                path = path[0]

            model = keras_load_model(path)
        except Exception as e:
            app_log.error(e)
            model = None
        return model

    def __init__(self):
        self._word2idx, _ = word2vec_model.get()
        super(CategoryModel, self).__init__()

    def load(self, path):
        self._model = CategoryModel.load_cnn_model(path)
        return self._model

    def parse_sentence(self, sentence):
        seg_list = jieba.cut(sentence.strip())
        fenci = []
        for item in seg_list:
            if item not in CategoryModel._stopwords and re.match(r'-?\d+\.?\d*', item) is None and len(item.strip()) > 0:
                fenci.append(item)
        data = []
        new_txt = []
        for word in fenci:
            idx = self._word2idx.get(word)
            if idx:
                new_txt.append(idx)
            else:
                new_txt.append(0)

        data.append(new_txt)
        # unify the length of the sentence with the pad_sequences function of keras
        data = sequence.pad_sequences(data, maxlen=CategoryModel.MAXLEN)
        return data

    def predict(self, sentence):
        try:
            if self._model is None:
                raise Exception("{}: Model is not loaded yet".format(__class__))
            data = self.parse_sentence(sentence)
            w2v_proba = self._model.predict(data)[0]
            pred_class = np.argmax(w2v_proba)

            return CategoryModel.LABEL_MAP[pred_class]
        except Exception as e:
            app_log.error(e)
            return ExternalLabel.EXCEPTION

