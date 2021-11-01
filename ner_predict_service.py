import os
import re
import json
import yaml
import config
import datetime
import numpy as np
import tensorflow as tf
from chinese_ner import utils
from utils import get_logger, get_risk_keywords
from redis_db import RedisDBWrapper
from keras_bert import get_custom_objects
from keras.models import model_from_yaml
from keras_bert import load_trained_model_from_checkpoint, Tokenizer

logger = get_logger("ner-predict-service")

_redis = RedisDBWrapper()

prev_text_context_len = 30

post_text_context_len = 150

sub_company_context_len = 20

sub_company_words = ["子公司", "旗下", "属下"]

st_pattern = re.compile("ST[\u4E00-\u9FA5]+")

additional_chars = {' ', '&', "'", '(', ')', '-', '.', '?', '—', '“', '”', '（', '）', 'Ａ', 'Ｂ'}

gpus = tf.config.experimental.list_physical_devices('GPU')

tf.config.experimental.set_visible_devices(devices=gpus[1], device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[1], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1000)])


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


def softmax(x):
    x = x - np.max(x)
    x = np.exp(x)
    return x / np.sum(x)


def load_ner_model():

    try:
        with open(config.bert_ner_model_path[0], 'r') as f:
            yaml_string = yaml.load(f, Loader=yaml.FullLoader)

        model = model_from_yaml(yaml_string, custom_objects=get_custom_objects())

        model.load_weights(config.bert_ner_model_path[1])

        return model
    except Exception as e:
        logger.error("Fail to load bert model: {}".format(e))
        return None


def init_tokenizer():

    dict_path = os.path.join(config.chinese_bert_corpus_dir, "vocab.txt")

    token_dict = {}

    with open(dict_path, 'r') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)

    return OurTokenizer(token_dict)


def predict(text_in, tokenizer, model):

    _tokens = tokenizer.tokenize(text_in)
    _x1, _x2 = tokenizer.encode(first=text_in)
    _x1, _x2 = np.array([_x1]), np.array([_x2])
    _ps1, _ps2 = model.predict([_x1, _x2])
    _ps1, _ps2 = softmax(_ps1[0]), softmax(_ps2[0])
    for i, _t in enumerate(_tokens):
        if len(_t) == 1 and re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', _t) and _t not in additional_chars:
            _ps1[i] -= 10
    start = _ps1.argmax()
    for end in range(start, len(_tokens)):
        _t = _tokens[end]
        if len(_t) == 1 and re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', _t) and _t not in additional_chars:
            break
    end = _ps2[start: end + 1].argmax() + start

    return text_in[start - 1: end], start - 1


def check_sub_company(text, company_prefix, sub_company, start_pos, lower_bound, upper_bound):

    prev_context_len = sub_company_context_len if start_pos - sub_company_context_len > lower_bound \
        else start_pos - lower_bound

    last_context_len = sub_company_context_len if start_pos + sub_company_context_len < upper_bound \
        else upper_bound - start_pos

    context_text = text[start_pos - prev_context_len: start_pos]
    context_text += text[start_pos: start_pos + last_context_len]

    for word in sub_company_words:
        if context_text.find(word) < 0:
            continue

        if sub_company.find(company_prefix) >= 0:
            if context_text.count(company_prefix) > 1:
                return True
        else:
            if context_text.find(company_prefix) >= 0 and context_text.find(sub_company) >= 0:
                return True

    return False


def has_bad_credit(text):
    freq = text.count("失信")

    freq_detail = text.count("失信被执行人") + text.count("失信责任主体") + text.count("失信惩戒对象")

    if freq == freq_detail or freq == freq_detail + 1:

        if text.count("不是失信") > 0 or text.count("非失信") > 0:
            return False

    return True


def handle_special_keywords(context_text, keyword, risk_type, company_prefix):

    if risk_type["dimension"] in ["高管涉案风险"] and context_text.find(keyword) >= 0:
        keyword_cnt = context_text.count(keyword)
        neg_cnt = context_text.count("未")
        if keyword_cnt > neg_cnt or neg_cnt == context_text.count("未来"):
            return True

    if keyword == "清算":
        if context_text.count("清算") != context_text.count("清算所"):
            return True

    if keyword == "下调":
        if context_text.count("下调") != context_text.count("下调价格") + context_text.count("价格下调"):
            return True

    if keyword == "调低":
        if context_text.count("调低") != context_text.count("调低价格") + context_text.count("价格调低"):
            return True

    if keyword == "延期":
        if context_text.count("延期") != context_text.count("自动延期"):
            return True

    if keyword == "失信" and context_text.find(keyword) >= 0:
        if has_bad_credit(context_text):
            return True

    if keyword == "自救" and context_text.find(keyword) >= 0:
        if context_text.find("演练") < 0 and context_text.find("演习") < 0 \
                and context_text.find("救援") < 0 and context_text.find("安全") < 0:
            return True

    if keyword == "ST":
        st_p = context_text.find("ST")
        if st_p >= 0:
            matched = st_pattern.findall(context_text[st_p:])
            if len(matched) > 0 and "_".join(matched).find(company_prefix) > 0:
                return True

    return False


def check_risk(text, company, risk_type, tokenizer, model):
    total_len = len(text)

    location, company_prefix = utils.extract_prefix(company)
    p = text.find(company_prefix)

    risk_type = json.loads(risk_type)
    risk_keywords = risk_type["fields"]

    related_risk_keywords = set()

    while p >= 0:

        prev_context_len = prev_text_context_len if p > prev_text_context_len else p

        last_context_len = post_text_context_len if p + post_text_context_len < total_len else total_len - p

        context_text = text[p - prev_context_len: p]
        context_text += text[p: p + last_context_len]

        found = False
        risk_keyword = ""

        for keyword in risk_keywords:
            if keyword in ["清算", "延期", "失信", "自救", "ST", "下调", "调低"] or risk_type["dimension"] in ["高管涉案风险"]:
                if handle_special_keywords(context_text, keyword, risk_type, company_prefix):
                    found = True
                    risk_keyword = keyword
                    break
            else:
                if context_text.find(keyword) >= 0:
                    found = True
                    risk_keyword = keyword
                    break

        if not found:
            p = text.find(company_prefix, p + len(company_prefix))
            continue

        text_in = u'___%s___%s' % (risk_type["dimension"], context_text)

        entity = None

        try:
            text_in = text_in.replace("\n", "\\n")
            entity, entity_start_pos = predict(text_in, tokenizer, model)
            #logger.info("Bert model find the entity of {}".format(entity))
        except Exception as e:
            logger.error("Bert model predict error: {}".format(e))

        if entity is None:
            p = text.find(company_prefix, p + len(company_prefix))
            continue

        if len(entity) > 3:
            _, entity_prefix = utils.extract_prefix(entity)
        else:
            entity_prefix = entity

        if company.find(location + entity_prefix) >= 0:
            if entity_start_pos < context_text.find(risk_keyword):
                logger.info("Company[{}] find the entity of {}, keyword = {}".format(company, entity, risk_keyword))
                related_risk_keywords.add(risk_keyword)

            if risk_keyword in ["下调", "调低", "ST", "自救"] or total_len - entity_start_pos < 30:
                logger.info("Company[{}] find the entity of {}, keyword = {}".format(company, entity, risk_keyword))
                related_risk_keywords.add(risk_keyword)

        entity_start_pos -= len(u'___%s___' % (risk_type["dimension"]))
        lower_bound = p - prev_context_len
        upper_bound = p + last_context_len

        if company.find(location + entity_prefix) < 0 and \
                check_sub_company(text, company_prefix, entity, entity_start_pos, lower_bound, upper_bound):
            related_risk_keywords.add(risk_keyword)

        p = text.find(company_prefix, p + len(company_prefix))

    return risk_type["dimension"], related_risk_keywords


def check_all_risks(text, company, tokenizer, model):

    risk_types = get_risk_keywords()

    related_risks = {}
    for risk in risk_types:

        risk_dimension, related_risk_keywords = check_risk(text, company, risk, tokenizer, model)

        if len(related_risk_keywords) > 0:
            related_risks[risk_dimension] = related_risk_keywords

    return related_risks


def _execute():

    logger.info("start ner model daemon")

    model = load_ner_model()
    tokenizer = init_tokenizer()

    while True:
        try:
            message = _redis.pop_data(config.ner_predict_req_table_key, True)
            #logger.info(message)

            if 'type' not in message:
                raise Exception("Data miss type field")

            if message['type'] != config.MessageType.PREDICTED.value:
                raise Exception("Unrecognized type field")

            related_risks = {}
            relationship = config.RiskRelation.UNRELATED.value

            related = utils.check_related_company_v2(message['content'], message["subject"].strip())
            if related:
                related_risks = check_all_risks(message['content'], message["subject"], tokenizer, model)

                if len(related_risks) > 0:
                    relationship = config.RiskRelation.RELATED_WITH_RISK.value
                else:
                    relationship = config.RiskRelation.RELATED_WITHOUT_RISK.value

            logger.info("Text: {} is {} with {}".format(message['uid'], relationship, message["subject"]))

            risks = []
            for dimension, risk_keywords in related_risks.items():
                risks.append({"risk_dimension": dimension, "risk_keywords": ",".join(risk_keywords)})

            data = {
                "type": config.MessageType.PREDICTED.value,
                "uid": message['uid'],
                "relationship": relationship,
                "risks": risks,
                "datetime": datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
            }

            if _redis.push_data(data, message['uid']) < 0:
                logger.error("Fail to push data to redis queue: {}".format(message['uid']))

        except Exception as e:
            logger.error(e)

            data = {
                "type": config.MessageType.PREDICTED.value,
                "uid": message['uid'],
                "relationship": config.RiskRelation.UNRELATED.value,
                "risks": [],
                "datetime": datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
            }

            if _redis.push_data(data, message['uid']) < 0:
                logger.error("Fail to push data to redis queue: {}".format(message['uid']))


if __name__ == "__main__":
    _execute()

