import os
import json
import celery
import hashlib
import datetime
import re
import logging
import jieba
import hashlib
import pandas as pd
from tornado.log import app_log

from config import stopwords_path, keywords_path, city_dict_path, company_risk_keywords_file, topic_local_dict_path


def get_logger(log_name):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler("logs/" + log_name + ".log", mode='a')
    handler.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(name)s[%(process)d] - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    ch.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)
    logger.addHandler(ch)
    return logger


def generate_uid(content, block_size=64 * 1024):

    half_len = int(len(content) / 2)
    time = str(datetime.datetime.now())
    content = time + content[: half_len] + time + content[half_len:] + time

    total_len = len(content)
    sha1 = hashlib.sha1()
    start = 0

    while start < total_len:
        end = start + block_size
        if end > total_len:
            end = total_len

        data = content[start: end].encode("utf8")
        start = end
        sha1.update(data)

    return sha1.hexdigest()


_celery_sentiment_app = None


def get_celery_sentiment_app(config):
    global _celery_sentiment_app
    if _celery_sentiment_app:
        return _celery_sentiment_app

    _celery_sentiment_app = celery.Celery(config_source=config.CelerySentimentConfig)

    return _celery_sentiment_app


_celery_topic_app = None


def get_celery_topic_app(config):
    global _celery_topic_app
    if _celery_topic_app:
        return _celery_topic_app

    _celery_topic_app = celery.Celery(config_source=config.CeleryTopicConfig)

    return _celery_topic_app


def get_words(filepath):

    words = set()
    if not os.path.exists(filepath):
        return words

    with open(filepath) as file:
        for line in file:
            words.add(line.strip())
    return words


_stop_words = None


def get_stopwords():
    global _stop_words

    if _stop_words is None:
        _stop_words = get_words(stopwords_path)

    return _stop_words


_init_keywords = None


def get_keywords(extra_keywords=None):
    global _init_keywords

    if _init_keywords is None:
        _init_keywords = get_words(keywords_path)
    if extra_keywords is None:
        return _init_keywords

    if isinstance(extra_keywords, str):
        extra_keywords = extra_keywords.split(',')
        extra_keywords = [x.strip() for x in extra_keywords]

    return _init_keywords.union(set(extra_keywords))


def create_lock_file(filepath):
    if not os.path.exists(filepath):
        os.makedirs(filepath[0: filepath.rfind("/")])
        fd = open(filepath, 'w')
        fd.close()


_city_dict = None


def get_city_dict():
    global _city_dict
    if _city_dict is not None:
        return _city_dict
    with open(city_dict_path, 'r', encoding="utf-8") as fr:
        _city_dict = json.load(fr)
    return _city_dict


def get_region(text, city_dict):
    region_count = dict()
    key_list = list()
    for key in city_dict.keys():
        if key in text:
            key_list.append(key)
            region = city_dict.get(key)
            if region not in region_count:
                region_count[region] = text.count(key)
            else:
                region_count[region] += text.count(key)
    if len(region_count) == 0:
        return "其他"
    app_log.info(key_list)
    max_region = max(region_count, key=lambda k: region_count[k])
    return max_region


_topic_local_dict = None


def load_local_dict():
    global _topic_local_dict

    if _topic_local_dict is None:
        with open(topic_local_dict_path, 'r', encoding='utf-8') as f:
            _topic_local_dict = f.readlines()
            #jieba.add_word(word)

        _topic_local_dict = [word.strip() for word in _topic_local_dict]

    return _topic_local_dict


def segmentation(text, stopwords):
    """
    Chinese segmentation for text
    :param text: list of text
    :param stopwords: stop words list
    :return:
    """
    seg_corpus = []
    for doc in text:
        if pd.isnull(doc):
            continue
        seg_list = jieba.cut(doc.strip(), HMM=False)
        seg_words = []
        for item in seg_list:
            if item not in stopwords and re.match(r'-?\d+\.?\d*', item) is None and len(item.strip()) > 0:
                seg_words.append(item)
        seg_words = ' '.join(seg_words)
        seg_corpus.append(seg_words)
    return seg_corpus


_risk_keywords = None


def has_company_risk_keywords(text):

    global _risk_keywords
    if _risk_keywords is None:
        if not os.path.exists(company_risk_keywords_file):
            return False, None

        with open(company_risk_keywords_file, 'r') as f:
            _risk_keywords = f.readlines()

    for keywords in _risk_keywords:
        keywords = json.loads(keywords)

        for _, words in keywords.items():
            if isinstance(words, str):
                continue

            for word in words:
                if text.find(word) >= 0:
                    return True, word

    return False, None


def get_risk_keywords():
    global _risk_keywords
    if _risk_keywords is None:
        if not os.path.exists(company_risk_keywords_file):
            return None

        with open(company_risk_keywords_file, 'r') as f:
            _risk_keywords = f.readlines()

    return _risk_keywords


def insert_dataframe_into_db(df_data, spark, table_name):
    if df_data is None or spark is None or table_name is None:
        return
    # Tranform Pandas.Dataframe data to Spark.Dataframe
    data_spark = spark.createDataFrame(df_data)

    # Spark.Dataframe claims a temp view
    data_spark.createOrReplaceTempView("TEMP_TABLE")

    # Data available in sql again
    sql = "insert into TABLE " + table_name
    sql += " select * from TEMP_TABLE"

    # Test temp view
    spark.sql(sql)


def is_chinese_text(content):

    chinese_char_num = 0

    for ch in content:
        if u'\u4e00' <= ch <= u'\u9fff':
            chinese_char_num += 1

        if chinese_char_num >= 100:
            return True

    return False


def make_sha1(x):
    sha = hashlib.sha1()

    sha.update(x.encode('utf-8'))
    return sha.hexdigest()



