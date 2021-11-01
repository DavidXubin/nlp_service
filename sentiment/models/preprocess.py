import os
from enum import Enum
import config
from tornado.log import app_log
from sentiment.models.model import Model, BinaryLabel, ExternalLabel
from sentiment.models.wo_ai_wo_jia import WoAiWoJiaModel


class PreprocessLabel(Enum):
    ERROR = 0
    PASS = 1
    DROP = 2


_preprocess_handlers = {"我爱我家": (WoAiWoJiaModel, config.preprocess_model_path)}


def preprocess(text, products=None):

    if products is None or len(products) == 0:
        return PreprocessLabel.PASS
    if isinstance(products, str):
        products = products.split(',')

    if len(products) > 1 or products[0].find(config.preprocess_keyword.strip()) < 0:
        return PreprocessLabel.PASS

    try:
        handler = _preprocess_handlers.get(config.preprocess_keyword)
        if handler is None:
            raise Exception("Cannot find handler for {}".format(config.preprocess_keyword))

        model = handler[0]()

        if model.load(handler[1]) is None:
            return PreprocessLabel.ERROR

        result = model.predict(text)
        if result == BinaryLabel.TRUE:
            result = PreprocessLabel.PASS
        elif result == BinaryLabel.FALSE:
            result = PreprocessLabel.DROP
        elif result == ExternalLabel.EXCEPTION:
            result = PreprocessLabel.ERROR

        return result
    except Exception as e:
        app_log.error(e)
        return PreprocessLabel.ERROR


Model.preprocess = preprocess

