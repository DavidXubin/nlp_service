import os
import re
import yaml
import jieba
from functools import partial
from itertools import product

import numpy as np
import keras.backend as K
from keras.models import model_from_yaml
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.utils import to_categorical
from keras.preprocessing import sequence
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation
from keras.layers import Bidirectional
from keras.layers import BatchNormalization

from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split

from tornado.log import app_log
from sentiment import word2vec_model
from sentiment.models.model import Model
from sentiment.models.model import InternalLabel, ExternalLabel
from utils import get_stopwords


class LSTMModel(Model):
    NEG_MID_LOSS = 3
    NEG_POS_LOSS = 5

    # the dimension of word vector
    VOCAB_DIM = 256
    # sentence length
    MAXLEN = 100
    # batch size
    BATCH_SIZE = 32
    # epoch num
    EPOCHS = 6
    # input length
    INPUT_LENGTH = 100

    _stopwords = get_stopwords()

    @staticmethod
    def w_categorical_crossentropy(y_true, y_pred, weights):
        '''
        Change loss
        This function will be keras built-in function, thus, the param y_true and y_pred will not be passed directly
        :param weights: loss weights matrix
        '''
        nb_cl = len(weights)
        final_mask = K.zeros_like(y_pred[:, 0])
        y_pred_max = K.max(y_pred, axis=1)
        y_pred_max = K.expand_dims(y_pred_max, 1)
        y_pred_max_mat = K.equal(y_pred, y_pred_max)
        for c_p, c_t in product(range(nb_cl), range(nb_cl)):
            final_mask += (K.cast(weights[c_t, c_p], K.floatx()) * K.cast(y_pred_max_mat[:, c_p], K.floatx()) * K.cast(
                y_true[:, c_t], K.floatx()))
        return K.categorical_crossentropy(y_pred, y_true) * final_mask

    @staticmethod
    def load_lstm_model(path):
        try:
            yml_path = path[0]
            weight_path = path[1]

            w_array = np.ones((3, 3))
            w_array[0, 1] = LSTMModel.NEG_MID_LOSS
            w_array[0, 2] = LSTMModel.NEG_POS_LOSS
            ncce = partial(LSTMModel.w_categorical_crossentropy, weights=w_array)
            ncce.__name__ = 'w_categorical_crossentropy'

            with open(yml_path, 'r') as f:
                yaml_string = yaml.load(f, Loader=yaml.FullLoader)
            model = model_from_yaml(yaml_string)

            app_log.info('loading weights......')
            model.load_weights(weight_path)
            model.compile(loss=ncce,
                          optimizer='adam',
                          metrics=['accuracy'])
            return model
        except Exception as e:
            app_log.error(e)
            return None

    def parse_sentence(self, sentence):
        sentence = Model.modify_content(sentence)
        seg_list = jieba.cut(sentence.strip())
        fenci = []
        for item in seg_list:
            if item not in LSTMModel._stopwords and re.match(r'-?\d+\.?\d*', item) is None and len(item.strip()) > 0:
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
        data = sequence.pad_sequences(data, maxlen=LSTMModel.MAXLEN)
        return data

    def __init__(self):
        self._word2idx, _ = word2vec_model.get()
        super(LSTMModel, self).__init__()

    def load(self, path):
        self._model = LSTMModel.load_lstm_model(path)

        return self._model

    def predict(self, sentence):
        try:
            if self._model is None:
                raise Exception("{}: Model is not loaded yet".format(__class__))
            data = self.parse_sentence(sentence)
            w2v_proba = self._model.predict_proba(data)[0]
            pred_class = np.argmax(w2v_proba)
            if pred_class != 0 and max(w2v_proba) / w2v_proba[0] < 1.3:
                pred_class = 0

            return Model._label_maps[InternalLabel(pred_class)], list(w2v_proba)

        except Exception as e:
            app_log.error(e)
            return ExternalLabel.EXCEPTION

    @staticmethod
    def segment_words(text, stoplist):
        """
        Divide the sentence and remove the disused words
        :param text: train sample array
        :param stoplist: disused words list
        :return: list of words segmented by jieba
        """
        text_list = []
        for document in text:
            seg_list = jieba.cut(document.strip())
            fenci = []
            for item in seg_list:
                if item not in stoplist and (re.match(r'-?\d+\.?\d*', item) is None) and len(item.strip()) > 0:
                    fenci.append(item)
            # if the word segmentation of the sentence is null,the label of the sentence should be deleted accordingly
            if len(fenci) > 0:
                text_list.append(fenci)
        return text_list

    @staticmethod
    def tokenizer(neg, mid, post, stoplist):
        """
        Tokenizer and generate data with label
        :param neg:
        :param mid:
        :param post:
        :param stoplist:
        :return: label and segmented data set
        """
        neg_words = LSTMModel.segment_words(neg, stoplist)
        post_words = LSTMModel.segment_words(post, stoplist)
        mid_words = LSTMModel.segment_words(mid, stoplist)
        combined = np.concatenate((post_words, mid_words, neg_words))
        # generating label and meging label data
        y = np.concatenate((np.array([2] * len(post_words), dtype=int), np.ones(len(mid_words), dtype=int),
                            np.zeros(len(neg_words), dtype=int)))
        return combined, y

    @staticmethod
    def get_data(index_dict, word_vectors, combined, y):
        """
        :param index_dict:
        :param word_vectors:
        :param combined:
        :param y:
        :return:
        """
        combined = LSTMModel.word2vec_load(combined, index_dict)

        # total number of word including the word without word vector
        n_symbols = len(index_dict) + 1
        # build word vector matrix which corresponding to the word index one by one
        embedding_weights = np.zeros((n_symbols, LSTMModel.VOCAB_DIM))
        for word, index in index_dict.items():  # 从索引为1的词语开始，对每个词语对应其词向量
            embedding_weights[index, :] = word_vectors[word]
        # partition test set and training set
        dummy_y = to_categorical(y, num_classes=3)
        # 因为样本分布不均衡，因此为每一类样本添加权重
        class_weights = class_weight.compute_class_weight('balanced', np.unique(y), y)
        x_train, x_val, y_train, y_val = train_test_split(combined, dummy_y, test_size=0.2, random_state=42)
        app_log.info(x_train.shape, y_train.shape)
        # return the input parameters needed of the lstm model
        return n_symbols, embedding_weights, x_train, y_train, x_val, y_val, class_weights

    @staticmethod
    def train_lstm(n_symbols, embedding_weights, x_train, y_train, x_val, y_val, class_weights, ncce):
        """
        :param n_symbols: total num of vocab in w2v model
        :param embedding_weights: weights to transfer index to w2v
        :param x_train:
        :param y_train:
        :param x_val:
        :param y_val:
        :param class_weights: weights for each class
        :param ncce: New cost function
        :return:
        """
        app_log.info('Defining a Simple Keras Model...')
        model = Sequential()  # or Graph or whatever
        model.add(Embedding(output_dim=LSTMModel.VOCAB_DIM,
                            input_dim=n_symbols,
                            mask_zero=True,
                            weights=[embedding_weights],
                            input_length=LSTMModel.INPUT_LENGTH))  # Adding Input Length

        model.add(Bidirectional(LSTM(units=256, return_sequences=False)))
        model.add(Dense(3))
        model.add(BatchNormalization())
        model.add(Activation('softmax'))

        app_log.info('Compiling the Model...')
        model.compile(loss=ncce,
                      optimizer='adam',
                      metrics=['accuracy'])
        app_log.info(model.summary())
        app_log.info("Train...")
        model.fit(x_train, y_train,
                  batch_size=LSTMModel.BATCH_SIZE,
                  epochs=LSTMModel.EPOCHS,
                  class_weight=class_weights,
                  verbose=1,
                  validation_data=(x_val, y_val))
        return model

    @staticmethod
    def parse_dataset(combined, index_dict):
        """
        Change character word into index number
        :param combined: character words array
        :param index_dict: word2vec model index dict {word:index}
        :return: index array
        """
        data = []
        for sentence in combined:
            new_txt = []
            for word in sentence:
                idx = index_dict.get(word)
                if idx:
                    new_txt.append(idx)
                else:
                    new_txt.append(0)
            data.append(new_txt)
        return data

    @staticmethod
    def word2vec_load(combined, index_dict):
        """
        load pre-trained word2vec model
        :param index_dict: index dict
        :param combined: train data in words
        """
        combined = LSTMModel.parse_dataset(combined, index_dict)
        combined = sequence.pad_sequences(combined, maxlen=LSTMModel.MAXLEN)

        return combined

    @staticmethod
    def build(train_data_df):
        np.random.seed(1337)

        try:
            neg = train_data_df[train_data_df.label == InternalLabel.NEGATIVE.value].content.values
            pos = train_data_df[train_data_df.label == InternalLabel.POSITIVE.value].content.values
            mid = train_data_df[train_data_df.label == InternalLabel.NEUTRAL.value].content.values

            combined, y_true = LSTMModel.tokenizer(neg, mid, pos, LSTMModel._stopwords)

            index_dict, word_vectors = word2vec_model.get()
            n_symbols, embedding_weights, x_train, y_train, x_val, y_val, class_weights = LSTMModel.get_data(index_dict,
                                                                                                             word_vectors,
                                                                                                             combined,
                                                                                                             y_true)

            w_array = np.ones((3, 3))
            w_array[0, 1] = 3
            w_array[0, 2] = 5
            ncce = partial(LSTMModel.w_categorical_crossentropy, weights=w_array)
            ncce.__name__ = 'w_categorical_crossentropy'

            model = LSTMModel.train_lstm(n_symbols, embedding_weights, x_train, y_train, x_val, y_val,
                                         class_weights, ncce)

            return model

        except Exception as e:
            app_log.error(e)
            return None

    @staticmethod
    def save(model, model_filepath):
        if model is None or model_filepath is None:
            return

        if isinstance(model_filepath, str):
            model_filepath = model_filepath.split(',')

        yml_path = model_filepath[0]
        weight_path = model_filepath[1]

        # save the trained lstm model
        p = yml_path.rfind('.')
        if p > 0:
            yml_path = yml_path[: p]

        p = weight_path.rfind('.')
        if p > 0:
            weight_path = weight_path[: p]

        new_yml_path = yml_path + '_updated.yml'
        new_weight_path = weight_path + '_updated.h5'

        if os.path.exists(new_yml_path):
            os.remove(new_yml_path)

        if os.path.exists(new_weight_path):
            os.remove(new_weight_path)

        yaml_string = model.to_yaml()
        with open(new_yml_path, 'w') as outfile:
            outfile.write(yaml.dump(yaml_string, default_flow_style=True))

        model.save_weights(new_weight_path)
