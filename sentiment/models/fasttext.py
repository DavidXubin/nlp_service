import os
import fasttext
import datetime as dt
from tornado.log import app_log
from sentiment.models.model import Model
from sentiment.models.model import InternalLabel, ExternalLabel
from sentiment.models.preprocess import PreprocessLabel


class FastTextModel(Model):

    _epoch = 25
    _word_ngrams = 2
    _train_data_filepath = os.path.join(os.getcwd(), "resources/fasttext_train.txt")

    def load(self, path):
        try:
            if isinstance(path, list):
                path = path[0]

            self._model = fasttext.load_model(path)
        except Exception as e:
            app_log.error(e)
            self._model = None
        return self._model

    def predict(self, text, keywords=None, products=None):
        try:
            if text is None or len(text) == 0:
                raise Exception("{}: Input text is empty".format(__class__))

            if self._model is None:
                raise Exception("{}: Model is not loaded yet".format(__class__))

            text = Model.extract(text, keywords)

            result = Model.preprocess(text, products)
            if result == PreprocessLabel.ERROR:
                raise Exception("Preprocess error !")

            if result == PreprocessLabel.DROP:
                return ExternalLabel.NONSENSE, []

            result = self._model.predict(text, k=-1)
            proba = list(result[1])
            result = InternalLabel(int(result[0][0].replace('__label__', '')))

            return Model._label_maps[result], proba

        except Exception as e:
            app_log.error(e)
            return ExternalLabel.EXCEPTION

    @staticmethod
    def make_data_file(x_data, y_data):

        file = open(FastTextModel._train_data_filepath, "w")

        buffer_size = 500
        buffer = ''

        for i, x in enumerate(x_data):
            buffer = buffer + x + "\t__label__" + str(y_data[i]) + "\n"
            if i > 0 and i % buffer_size == 0:
                file.write(buffer)
                buffer = ''

        if len(buffer) > 0:
            file.write(buffer)

        file.close()

    @staticmethod
    def build(train_data_df):

        try:
            x_train, y_train = train_data_df.content.values, train_data_df.label.values

            FastTextModel.make_data_file(x_train, y_train)

            model = fasttext.train_supervised(FastTextModel._train_data_filepath, label_prefix="__label__",
                                              epoch=FastTextModel._epoch, wordNgrams=FastTextModel._word_ngrams)
            return model
        except Exception as e:
            app_log.error(e)
            return None

    @staticmethod
    def save(model, model_filepath):
        if model:
            if isinstance(model_filepath, list):
                model_filepath = model_filepath[0]

            p = model_filepath.rfind('.')
            if p > 0:
                model_filepath = model_filepath[: p]

            new_model_filepath = model_filepath + '_updated.bin'
            if os.path.exists(new_model_filepath):
                os.remove(new_model_filepath)

            model.save_model(new_model_filepath)
