from tornado.log import app_log
from keras.preprocessing import sequence
from keras.models import load_model as keras_load_model
from bert import tokenization
from config import vocab_path
from sentiment.models.model import Model
from sentiment.models.model import BinaryLabel, ExternalLabel


class WoAiWoJiaModel(Model):

    MAXLEN = 100
    _tokenizer = tokenization.FullTokenizer(vocab_path)

    def load(self, path):
        try:
            if isinstance(path, list):
                path = path[0]

            self._model = keras_load_model(path)
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

            text_token = [self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(text))]
            text_matrix = sequence.pad_sequences(text_token, maxlen=WoAiWoJiaModel.MAXLEN)
            pred_class = self._model.predict_classes(text_matrix)

            return BinaryLabel(pred_class[0][0])
        except Exception as e:
            app_log.error(e)
            return ExternalLabel.EXCEPTION
