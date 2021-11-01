import re
import datetime
from tornado.log import app_log

import config
from sentiment import celery_app, city_dict
from sentiment.models.model import Model
from sentiment.models.model import ExternalLabel
from sentiment.models.preprocess import PreprocessLabel
from sentiment.models.categorization import CategoryModel
from sentiment.models.fasttext import FastTextModel
from sentiment.models.lstm import LSTMModel
from sentiment.models.textcnn import TextCNNModel
from utils import generate_uid, get_region
from redis_db import RedisDBWrapper
#from rw_flock import RWFlocker

_model_factory = {
    "Fasttext": (FastTextModel, config.fasttext_model_path),
    "LSTM": (LSTMModel, config.lstm_model_path),
    "TEXTCNN": (TextCNNModel, config.textcnn_model_path),
}

_redis = RedisDBWrapper()
_other_region = "其他"


@celery_app.task
def predict(original_text, keywords=None, products=None):
    cop = re.compile(re.compile(u'[^0-9a-zA-Z\u4e00-\u9fa5.，,。？“”]+', re.UNICODE))
    original_text = cop.sub(" ", original_text)
    uid = generate_uid(original_text)

    try:
        if original_text is None or len(original_text) == 0:
            # No Input
            raise Exception("Input text is empty")
        text = Model.extract(original_text, keywords)

        if text is None:
            # No keywords
            return uid, ExternalLabel.NONSENSE.value, _other_region

        result = Model.preprocess(text, products)
        if result == PreprocessLabel.ERROR:
            raise Exception("Preprocess error !")

        if result == PreprocessLabel.DROP:
            # 不是我爱我家房产的文本
            return uid, ExternalLabel.NONSENSE.value, _other_region

        # inter-process RW lock harms performance
        # RWFlocker.lock(RWFlocker.READ)
        """
        get region here
        """
        region = get_region(original_text, city_dict)
        app_log.info(region)

        # 添加预测类型的功能，如果预测主题在config.nonsense_topic里面，则直接返回-10
        category_model = CategoryModel()
        category_model.load(config.category_model_path)
        category = category_model.predict(text)
        del category_model

        app_log.info("Category is {}".format(category))

        if category in config.nonsense_topic:
            return uid, ExternalLabel.NONSENSE.value, _other_region

        model_class = _model_factory.get(config.preferred_model)
        if model_class is None:
            raise Exception("{} does not exist".format(config.preferred_model))

        _model = model_class[0]()
        _model.load(model_class[1])

        label, proba = _model.predict(text)

        del _model
        # RWFlocker.unlock()

        if label == ExternalLabel.NONSENSE or label == ExternalLabel.EXCEPTION:
            return uid, label.value, region

        if keywords and isinstance(keywords, str):
            keywords = keywords.split(',')

        if products and isinstance(products, str):
            products = products.split(',')

        data = {
            "type": config.MessageType.PREDICTED.value,
            "uid": uid,
            "keywords": keywords,
            "products": products,
            "text": original_text,
            "label": label.value,
            "proba": str(proba).lstrip('[').rstrip(']'),
            "datetime": datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        }

        if _redis.push_data(data, config.predict_table_key) < 0:
            app_log.error("Fail to push data to redis queue: {}".format(config.predict_table_key))

        return uid, label.value, region
    except Exception as e:
        app_log.error(e)
        return uid, ExternalLabel.EXCEPTION.value, _other_region
    # finally:
    #    RWFlocker.unlock()


@celery_app.task
def extract_region(text, subject):
    try:
        cop = re.compile(re.compile(u'[^0-9a-zA-Z\u4e00-\u9fa5.，,。？“”]+', re.UNICODE))
        text = cop.sub(" ", text)
        uid = generate_uid(text)

        region = get_region(text, city_dict)
        app_log.info(region)

        subject = subject.strip()
        if len(subject) == 0:
            return uid, region, False

        _, relationship, _ = extract_company_risks(text, subject)
        related = False

        if relationship == config.RiskRelation.RELATED_WITH_RISK.value:
            related = True

        return uid, region, related
    except Exception as e:
        app_log.error("Fail to extract region and relation: {}".format(e))
        return "-1", _other_region, False


@celery_app.task
def extract_company_risks(text, subject):
    try:
        cop = re.compile(re.compile(u'[^0-9a-zA-Z\u4e00-\u9fa5.，,。？“”]+', re.UNICODE))
        text = cop.sub(" ", text)
        uid = generate_uid(text)

        subject = subject.strip()
        if len(subject) == 0:
            return uid, config.RiskRelation.UNRELATED.value, "", ""

        data = {
            "type": config.MessageType.PREDICTED.value,
            "uid": uid,
            "content": text,
            "subject": subject,
            "datetime": datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        }

        if _redis.push_data(data, config.ner_predict_req_table_key) < 0:
            app_log.error("Fail to push data to redis queue: {}".format(config.ner_predict_req_table_key))

        message = _redis.pop_data(uid, True)
        app_log.info(message)
        relationship = message["relationship"]

        label = ExternalLabel.NEUTRAL.value
        if relationship == config.RiskRelation.RELATED_WITH_RISK.value:
            label = ExternalLabel.THIRD_PARTY_NEGATIVE.value

        data = {
            "type": config.MessageType.PREDICTED.value,
            "uid": uid,
            "keywords":  "开发商-relation:" + relationship,
            "products": subject,
            "text": text,
            "label": label,
            "proba": "",
            "datetime": datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        }

        if _redis.push_data(data, config.predict_table_key) < 0:
            app_log.error("Fail to push data to redis queue: {}".format(config.predict_table_key))

        return uid, relationship, message["risks"]
    except Exception as e:
        app_log.error("Fail to extract region and relation: {}".format(e))
        return "-1", config.RiskRelation.UNRELATED.value, []


@celery_app.task
def extract_text_summary(text, algorithm="seq2seq"):
    try:
        uid = generate_uid(text + "extract_text_summary")

        data = {
            "type": config.MessageType.PREDICTED.value,
            "uid": uid,
            "algorithm": algorithm,
            "content": text,
            "datetime": datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        }

        if _redis.push_data(data, config.text_summary_req_table_key) < 0:
            app_log.error("Fail to push data to redis queue: {}".format(config.text_summary_req_table_key))

        message = _redis.pop_data(uid, True)
        app_log.info(message)

        return uid, message["status"], message["summary"]
    except Exception as e:
        app_log.error("Fail to extract summary: {}".format(e))
        return "-1", "error", str(e)
