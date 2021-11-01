import os
import re
import datetime
import json
import requests
import threading
import pandas as pd
import numpy as np
from enum import Enum
from requests.auth import HTTPBasicAuth
from multiprocessing import Process
from datetime import timedelta
import config
from pyspark_manager import PySparkMgr
from redis_db import RedisDBWrapper
from rw_flock import RWFlocker
from topic_analysis import logger
import topic_analysis.LDAModel as LDAModel
import topic_analysis.TFIDFModel as TFIDFModel
from chinese_ner.utils import check_related_company_v2
from utils import insert_dataframe_into_db, has_company_risk_keywords, get_stopwords, is_chinese_text, make_sha1
from topic_analysis.run import get_real_estate_business_csv_path, get_real_estate_company_csv_path
import dedup_with_tfifd
from get_text_summay import extract_business_topic_texts_summaries


class TopicType(Enum):
    COMPANY = "company"
    BUSINESS = "business"


_redis = RedisDBWrapper()


def set_topic_last_process_day(day, company_type):
    _redis.get_handler().set(company_type + config.topic_last_process_date_key, day)


def get_topic_last_process_day(company_type):
    return _redis.get_handler().get(company_type + config.topic_last_process_date_key)


def _get_crawler_data_cnt(company_type, start_date, end_date):
    params = {
        "start_date": start_date, "end_date": end_date, "company": company_type,
        "page_num": config.TopicRequestConfig.default_page_num, "page_size": config.TopicRequestConfig.default_page_size
    }

    try:
        auth = HTTPBasicAuth(config.TopicRequestConfig.username, config.TopicRequestConfig.password)
        r = requests.get(config.TopicRequestConfig.request_url, params=params, auth=auth,
                         timeout=config.TopicRequestConfig.timeout)
        r.raise_for_status()

        response = json.loads(r.text)
        if response["status"] != config.TopicRequestConfig.status_ok or \
                response["message"] != config.TopicRequestConfig.success_message:
            raise Exception("Fail to get {} data size: status={}, message={}".
                            format(company_type, response["status"], response["message"]))

        total_count = response["count"]

        logger.info("Get crawler data, total size is {}".format(total_count))

        return total_count
    except Exception as e:
        logger.error(e)
        return -1


def _fetch_internal(company_type, start_date, end_date, start_page=config.TopicRequestConfig.default_page_num,
                    total_count=-1):
    params = {
        "start_date": start_date, "end_date": end_date, "company": company_type,
        "page_num": start_page, "page_size": config.TopicRequestConfig.default_page_size
    }

    auth = HTTPBasicAuth(config.TopicRequestConfig.username, config.TopicRequestConfig.password)

    df = pd.DataFrame([],  columns=["crawler_id", "sentiment_id", "keywords", "products", "content",
                                    "company_type", "pub_date", "start_date", "end_date"])
    index = 0
    processed_count = 0

    try:
        while True:
            r = requests.get(config.TopicRequestConfig.request_url, params=params, auth=auth,
                             timeout=config.TopicRequestConfig.timeout)
            r.raise_for_status()

            response = json.loads(r.text)
            if response["status"] != config.TopicRequestConfig.status_ok or \
                    response["message"] != config.TopicRequestConfig.success_message:
                raise Exception("Fail to fetch {} data: status={}, message={}".
                                format(company_type, response["status"], response["message"]))

            if total_count < 0:
                total_count = response["count"]

            for data in response["data"]:
                products = data["products"]
                if isinstance(products, list):
                    products = ','.join(products)

                keywords = data["keywords"]
                if isinstance(keywords, list):
                    keywords = ','.join(keywords)

                df.loc[index] = {
                    "crawler_id": data["id"],
                    "sentiment_id": data["mood_id"] if data["mood_id"] is not None else "null",
                    "keywords": keywords,
                    "products": products,
                    "content": data["content"],
                    "company_type": company_type,
                    "pub_date": data["pub_date"],
                    "start_date": start_date,
                    "end_date": end_date
                }

                index += 1

            processed_count += len(response["data"])

            logger.info("Thread[{}]: processed_count = {}, total_count={}".format(threading.currentThread().ident,
                                                                                  processed_count, total_count))

            if processed_count >= total_count:
                break

            params["page_num"] += 1

        tmp_file = os.path.join(config.topic_data_tmp_path, company_type + "_" + str(start_page) + ".csv")
        df.to_csv(tmp_file, sep=',', header=True, index=False)

        logger.info("result_df[{}] shape is {}".format(start_page, df.shape))

    except Exception as e:
        logger.error(e)


def _fetch(company_type, start_date, end_date, start_page=config.TopicRequestConfig.default_page_num, total_count=-1):
    _fetch_internal(company_type, start_date, end_date, start_page, total_count)

    tmp_file = os.path.join(config.topic_data_tmp_path, company_type + "_" + str(start_page) + ".csv")
    if not os.path.exists(tmp_file):
        return None

    return pd.read_csv(tmp_file,  sep=',')


def _concurrent_fetch_data(company_type, start_date, end_date):

    if not os.path.exists(config.topic_data_tmp_path):
        os.makedirs(config.topic_data_tmp_path)

    data_cnt = _get_crawler_data_cnt(company_type, start_date, end_date)
    if data_cnt < 0:
        return

    page_cnt = int(data_cnt / config.TopicRequestConfig.default_page_size)
    if page_cnt % config.TopicRequestConfig.default_page_size > 0:
        page_cnt += 1

    if page_cnt < config.TopicRequestConfig.max_threads:
        concurrent_number = 1
        step = 0
        step_data_range = data_cnt
    else:
        concurrent_number = config.TopicRequestConfig.max_threads
        step = int(page_cnt / config.TopicRequestConfig.max_threads)
        step_data_range = step * config.TopicRequestConfig.default_page_size

    df_paths = []
    _threads = []
    for i in range(concurrent_number):
        start_idx = i * step + 1

        if i == concurrent_number - 1:
            step_data_range = data_cnt - (start_idx - 1) * config.TopicRequestConfig.default_page_size

        logger.info("Thread[{}]: start={}, range={}".format(i, start_idx, step_data_range))

        t = threading.Thread(target=_fetch_internal, args=(company_type, start_date, end_date,
                                                           start_idx, step_data_range, ))
        t.start()
        _threads.append(t)
        df_paths.append(os.path.join(config.topic_data_tmp_path, company_type + "_" + str(start_idx) + ".csv"))

    for t in _threads:
        t.join()

    all_df = None
    for path in df_paths:
        if not os.path.exists(path):
            logger.error("{} does not exists !".format(path))
            continue

        df = pd.read_csv(path,  sep=',')
        if all_df is None:
            all_df = df
        else:
            all_df = pd.concat([all_df, df], axis=0)

        os.remove(path)

    logger.info("result_df[ALL] shape is {}".format(all_df.shape))

    return all_df


def _fetch_internal_v2(company_type, start_date, end_date, global_start_date, global_end_date):
    params = {
        "start_date": start_date, "end_date": end_date, "company": company_type,
        "page_num": config.TopicRequestConfig.default_page_num, "page_size": config.TopicRequestConfig.default_page_size
    }

    auth = HTTPBasicAuth(config.TopicRequestConfig.username, config.TopicRequestConfig.password)

    df = pd.DataFrame([],  columns=["crawler_id", "sentiment_id", "keywords", "products", "content",
                                    "company_type", "pub_date", "start_date", "end_date"])
    index = 0

    try:
        while True:
            r = requests.get(config.TopicRequestConfig.request_url, params=params, auth=auth,
                             timeout=config.TopicRequestConfig.timeout)
            r.raise_for_status()

            response = json.loads(r.text)
            if response["status"] != config.TopicRequestConfig.status_ok or \
                    response["message"] != config.TopicRequestConfig.success_message:
                raise Exception("Fail to fetch {} data: status={}, message={}".
                                format(company_type, response["status"], response["message"]))

            page_count = response["count"]

            for data in response["data"]:
                products = data["products"]
                if isinstance(products, list):
                    products = ','.join(products)

                keywords = data["keywords"]
                if isinstance(keywords, list):
                    keywords = ','.join(keywords)

                df.loc[index] = {
                    "crawler_id": data["id"],
                    "sentiment_id": data["mood_id"] if data["mood_id"] is not None else "null",
                    "keywords": keywords,
                    "products": products,
                    "content": data["content"],
                    "company_type": company_type,
                    "pub_date": data["pub_date"],
                    "start_date": global_start_date,
                    "end_date": global_end_date
                }

                index += 1

            logger.info("Thread[{}]: page index = {}".format(threading.currentThread().ident, params["page_num"]))

            if page_count < params["page_size"]:
                break

            params["page_num"] += 1

        tmp_file = os.path.join(config.topic_data_tmp_path, company_type + "_" + start_date + ".csv")
        df.to_csv(tmp_file, sep=',', header=True, index=False)

        logger.info("result_df[{}] shape is {}".format(start_date, df.shape))

    except Exception as e:
        logger.error(e)


def _concurrent_fetch_data_v2(company_type, start_date, end_date):

    if not os.path.exists(config.topic_data_tmp_path):
        os.makedirs(config.topic_data_tmp_path)

    date_range = pd.date_range(start=start_date, end=end_date, freq='W').tolist()
    if pd.Timestamp(start_date) < date_range[0]:
        date_range.insert(0, pd.Timestamp(start_date))

    if pd.Timestamp(end_date) > date_range[-1]:
        date_range.append(pd.Timestamp(end_date))

    date_range = [pd.to_datetime(x) for x in date_range]
    logger.info(date_range)

    df_paths = []
    _threads = []
    for i, date in enumerate(date_range):
        if i == len(date_range) - 1:
            break

        start = date_range[i]
        if i > 0:
            start = start + timedelta(days=1)
        end = date_range[i + 1]

        start = start.strftime("%Y-%m-%d")
        end = end.strftime("%Y-%m-%d")

        logger.info("Thread[{}]: start date={}, end date={}".format(i, start, end))

        t = threading.Thread(target=_fetch_internal_v2, args=(company_type, start, end, start_date, end_date, ))
        t.start()

        _threads.append(t)
        df_paths.append(os.path.join(config.topic_data_tmp_path, company_type + "_" + start + ".csv"))

    for t in _threads:
        t.join()

    all_df = None
    for path in df_paths:
        if not os.path.exists(path):
            logger.error("{} does not exists !".format(path))
            continue

        df = pd.read_csv(path,  sep=',')
        if all_df is None:
            all_df = df
        else:
            all_df = pd.concat([all_df, df], axis=0)

        os.remove(path)

    logger.info("result_df[ALL] shape is {}".format(all_df.shape))

    return all_df


def concurrent_fetch_data_v3(start_date, end_date):
    sc = None

    try:
        spark_manager = PySparkMgr(config.spark_config, "topic-analysis-pyspark-get-data")
        spark, sc = spark_manager.start(config.spark_user)
        if spark is None:
            raise Exception("Fail to launch spark session")
        logger.info(spark)

        sql = "select * from " + config.dw_crawler_data_table + " where pub_date >='" + start_date + \
              "' and pub_date <= '" + end_date + "'"
        df = spark.sql(sql).toPandas()
        df.rename(columns={"_id": "crawler_id", "mood_id": "sentiment_id",
                           "rb_parse": "content", "company_target": "products",
                           "crawl_word": "keywords"}, inplace=True)

        df = df[(df["pub_date"] >= start_date) & (df["pub_date"] <= end_date)]

        df.to_csv(config.dw_crawler_data_file_path, sep=',', header=True, index=False)

    except Exception as e:
        logger.error(e)
        return None
    finally:
        if sc is not None:
            sc.stop()


def fetch_market_data(company_type, start_date, end_date):

    df = _fetch(company_type, start_date, end_date)
    if df is None:
        logger.error("Fail to retrieve data of {}".format(company_type))
        return

    df.to_csv(config.topic_local_market_data_path, sep=',', header=True, index=False)


def fetch_public_relation_data(company_type, start_date, end_date):

    df = _fetch(company_type, start_date, end_date)
    if df is None:
        logger.error("Fail to retrieve data of {}".format(company_type))
        return

    df.to_csv(config.topic_local_public_relation_data_path, sep=',', header=True, index=False)


def fetch_sentiment_data():
    sc = None
    try:
        spark_manager = PySparkMgr(config.spark_config, "topic-analysis-pyspark-get-sentiment-data")
        spark, sc = spark_manager.start(config.spark_user)
        if spark is None:
            raise Exception("Fail to launch spark session")
        logger.info(spark)

        sentiment_df = _get_sentiment_labels_data(spark)

        if sentiment_df is not None:
            sentiment_df.to_csv(config.sentiment_data_file_path, sep=',', header=True, index=False)
    except Exception as e:
        logger.error("Fail to fetch sentiment data: {}".format(e))
    finally:
        if sc is not None:
            sc.stop()


def fetch_hot_company_data(start_date, end_date):
    if not os.path.exists(config.dw_crawler_data_file_path):
        logger.error("Fail to retrieve crawler data: {} does not exist".format(config.dw_crawler_data_file_path))
        return

    data_df = pd.read_csv(config.dw_crawler_data_file_path, sep=',')

    data_df = data_df[~data_df.products.str.contains(config.CompanyType.REAL_ESTATE.value)]

    hot_company_df = pd.read_csv(config.hot_company_keyword_config_path, sep=',')

    hot_companies = np.unique(hot_company_df["company"].values).tolist()

    selected = []

    for idx in data_df.index:

        products = data_df.loc[idx, "products"]
        if products.find("[") >= 0 and products.find("]") > 0:
            products = json.loads(products)
        else:
            products = re.split("[,，]", products)

        hits = np.array([hot_companies.count(x) for x in products]).sum()
        if hits == 0:
            continue

        selected.append(idx)

    data_df = data_df.loc[selected, :]
    data_df["start_date"] = start_date
    data_df["end_date"] = end_date

    data_df.drop_duplicates(["crawler_id"], inplace=True)

    data_df.index = np.arange(data_df.shape[0])

    tmp_idx = 0
    tmp_df = pd.DataFrame([], columns=data_df.columns)

    for idx in data_df.index:

        companies = data_df.loc[idx, "company"]
        if companies.find("[") >= 0 and companies.find("]") > 0:
            companies = json.loads(companies)
        else:
            companies = re.split("[,，]", companies)

        found = False

        for i, company in enumerate(companies):
            if company not in hot_companies:
                continue

            if not found:
                data_df.loc[idx, "company"] = company
                data_df.loc[idx, "products"] = company

                found = True
            else:
                tmp_df.loc[tmp_idx] = json.loads(data_df.loc[idx, :].to_json())
                tmp_df.loc[tmp_idx, "company"] = company
                tmp_df.loc[tmp_idx, "products"] = company

                tmp_idx += 1

    data_df = pd.concat([data_df, tmp_df])
    data_df.index = np.arange(data_df.shape[0])

    data_df.to_csv(config.hot_company_keyword_data_path, sep=',', header=True, index=False)


def fetch_oversea_topic_data(start_date, end_date):
    if not os.path.exists(config.dw_crawler_data_file_path):
        logger.error("Fail to retrieve crawler data: {} does not exist".format(config.dw_crawler_data_file_path))
        return

    data_df = pd.read_csv(config.dw_crawler_data_file_path, sep=',')

    data_df = data_df[~data_df.products.str.contains(config.CompanyType.REAL_ESTATE.value)]

    oversea_topic_df = pd.read_csv(config.oversea_topic_config_path, sep=',')

    oversea_topics = np.unique(oversea_topic_df["topic_name"].values).tolist()

    selected = []

    for idx in data_df.index:

        products = data_df.loc[idx, "products"]
        if products.find("[") >= 0 and products.find("]") > 0:
            products = json.loads(products)
        else:
            products = re.split("[,，]", products)

        hits = np.array([oversea_topics.count(x) for x in products]).sum()
        if hits == 0:
            continue

        selected.append(idx)

    data_df = data_df.loc[selected, :]
    data_df["start_date"] = start_date
    data_df["end_date"] = end_date

    data_df = data_df[~pd.isnull(data_df.content)]
    data_df = data_df[data_df.content.str.strip().str.len() > 0]

    data_df.content = data_df.content.astype("str")

    data_df.crawler_id = np.where(~pd.isnull(data_df.crawler_id), data_df.crawler_id,
                                  data_df.content.apply(lambda x: make_sha1(x.strip())))

    data_df.crawler_id = data_df.crawler_id.astype("str")

    data_df.crawler_id = np.where(data_df.crawler_id.str.strip().str.len() > 0, data_df.crawler_id,
                                  data_df.content.apply(lambda x: make_sha1(x.strip())))

    data_df["is_chinese"] = data_df.content.apply(lambda x: is_chinese_text(x.strip()))
    data_df = data_df[data_df.is_chinese]

    if data_df.shape[0] == 0:
        logger.info("Oversea topic has not chinese articles from {} to {}".format(start_date, end_date))
        return

    data_df.drop(["is_chinese"], axis=1, inplace=True)

    data_df.drop_duplicates(["crawler_id"], inplace=True)

    data_df.index = np.arange(data_df.shape[0])

    tmp_idx = 0
    tmp_df = pd.DataFrame([], columns=data_df.columns)

    for idx in data_df.index:

        companies = data_df.loc[idx, "company"]
        if companies.find("[") >= 0 and companies.find("]") > 0:
            companies = json.loads(companies)
        else:
            companies = re.split("[,，]", companies)

        found = False

        for i, company in enumerate(companies):
            if company not in oversea_topics:
                continue

            if not found:
                data_df.loc[idx, "company"] = company
                data_df.loc[idx, "products"] = company

                found = True
            else:
                tmp_df.loc[tmp_idx] = json.loads(data_df.loc[idx, :].to_json())
                tmp_df.loc[tmp_idx, "company"] = company
                tmp_df.loc[tmp_idx, "products"] = company

                tmp_idx += 1

    data_df = pd.concat([data_df, tmp_df], ignore_index=True)
    data_df.index = np.arange(data_df.shape[0])

    data_df.to_csv(config.oversea_topic_data_path, sep=',', header=True, index=False)


def fetch_real_estate_company_data(start_date, end_date):

    if not os.path.exists(config.dw_crawler_data_file_path):
        logger.error("Fail to retrieve real estate company data: {} does not exist".format(config.dw_crawler_data_file_path))
        return

    data_df = pd.read_csv(config.dw_crawler_data_file_path, sep=',')

    selected = []
    for idx in data_df.index:
        company = data_df.loc[idx, "company"]
        if pd.isnull(company):
            continue

        if company.find(config.CompanyType.REAL_ESTATE.value) < 0:
            continue

        selected.append(idx)

    data_df = data_df.loc[selected, :]
    data_df["start_date"] = start_date
    data_df["end_date"] = end_date

    data_df.to_csv(config.topic_local_real_estate_data_path, sep=',', header=True, index=False)


def fetch_ke_research_data(start_date, end_date):

    sc = None

    try:
        spark_manager = PySparkMgr(config.spark_config, "topic-analysis-pyspark-get-ke-research-data")
        spark, sc = spark_manager.start(config.spark_user)
        if spark is None:
            raise Exception("Fail to launch spark session")
        logger.info(spark)

        max_date = spark.sql("select max(pt) as max_date from spark_dw.dw_ke_bkjf_content_policy_news_da").toPandas()
        max_date = max_date.loc[0, "max_date"]

        sql = """select news_id as crawler_id, news_content as content, 'ke_research' as products, null as keywords, null as sentiment_id, """ \
              """ publish_time as pub_date, 'ke_research' as company, '{}' as start_date, '{}' as end_date from {} where date(publish_time) >='{}' and """ \
              """ date(publish_time) <= '{}' and pt = '{}' and policy_type in ('41','31','32','33','34','35','36','37','38','51','52','55')""". \
            format(start_date, end_date, "spark_dw.dw_ke_bkjf_content_policy_news_da", start_date, end_date, max_date)

        news_df = spark.sql(sql).toPandas()

        company_df = pd.read_csv(config.topic_local_real_estate_data_path, sep=",")

        all_data_df = pd.concat([company_df, news_df])

        all_data_df.to_csv(config.topic_local_real_estate_data_path, sep=',', header=True, index=False)
    except Exception as e:
        logger.error(e)
        return None
    finally:
        if sc is not None:
            sc.stop()


def _analyze_market_topic(spark, sc, today):
    try:
        data_path = config.topic_local_market_data_path

        if not os.path.exists(data_path):
            raise Exception("{} does not exist".format(data_path))

        text_df = pd.read_csv(data_path, sep=',')
        text_df.fillna(value="0", inplace=True)

        text_df = dedup_with_tfifd.dedup(spark, sc, text_df, get_stopwords())

        insert_dataframe_into_db(text_df, spark, config.topic_data_table)
        logger.info("inserted new data into topic data table")

        result_df, top_n_per_topic_df = LDAModel.process(text_df, today)
        if result_df is None or top_n_per_topic_df is None:
            raise Exception("LDAModel failed when analyzing market data")

        insert_dataframe_into_db(top_n_per_topic_df, spark, config.top_n_per_topic_result_table)
        insert_dataframe_into_db(result_df, spark, config.topic_result_table)
        logger.info("inserted topic analysis result into topic result table")

        top_n_per_topic_df = extract_business_topic_texts_summaries(text_df, top_n_per_topic_df, result_df)

        RWFlocker.lock(RWFlocker.WRITE)

        result_df.to_csv(config.topic_local_market_result_path, sep=',', header=True, index=False)
        top_n_per_topic_df.to_csv(config.topic_local_market_top_n_per_topic_path, sep=',', header=True, index=False)
    except Exception as e:
        logger.error("Market topic failed: {}".format(e))
    finally:
        RWFlocker.unlock()


def _analyze_public_relation_topic(spark, sc, today):
    try:
        data_path = config.topic_local_public_relation_data_path
        if not os.path.exists(data_path):
            raise Exception("{} does not exist".format(data_path))

        text_df = pd.read_csv(data_path, sep=',')
        text_df.fillna(value="0", inplace=True)

        logger.info("before debup")
        logger.info(text_df.columns)
        logger.info(text_df.shape)

        text_df = dedup_with_tfifd.dedup(spark, sc, text_df, get_stopwords())

        logger.info("after debup")
        logger.info(text_df.columns)
        logger.info(text_df.shape)

        insert_dataframe_into_db(text_df, spark, config.topic_data_table)
        logger.info("inserted new data into topic data table")

        result_df = TFIDFModel.process(text_df, today)
        if result_df is None:
            raise Exception("TFIDFModel failed when analyzing public relation data")

        insert_dataframe_into_db(result_df, spark, config.topic_result_table)
        logger.info("inserted topic analysis result into topic result table")

        result_df.rename(columns={"crawler_id": "sentiment_id_list"}, inplace=True)

        result_df = extract_business_topic_texts_summaries(text_df, result_df, result_df)

        result_df.rename(columns={"sentiment_id_list": "crawler_id"}, inplace=True)

        RWFlocker.lock(RWFlocker.WRITE)

        result_df.to_csv(config.topic_local_public_relation_result_path, sep=',', header=True, index=False)
    except Exception as e:
        logger.error("Public relation topic failed: {}".format(e))
    finally:
        RWFlocker.unlock()


def _analyze_real_estate_company_topic(company_name, company_df, today):
    company_keywords = None
    neg_indices = []

    for idx in company_df.index:
        products = company_df.loc[idx, "products"]
        if pd.isnull(products):
            continue

        p = products.find(company_name)
        if p < 0:
            continue

        keywords = company_df.loc[idx, "keywords"]
        if pd.isnull(keywords):
            continue

        if company_keywords is None:
            company_keywords = keywords

        company_df.loc[idx, "products"] = company_name.strip()

        neg_indices.append(idx)

    logger.info("{}: negative text number is {}".format(company_name, len(neg_indices)))

    if len(neg_indices) == 0:
        return None, None

    company_df = company_df.loc[neg_indices, :]

    result_df, top_n_per_topic_df = LDAModel.process(company_df, today, True, False)
    if result_df is None or top_n_per_topic_df is None:
        logger.error("LDAModel failed when analyzing real estate data")
        return None, None

    top_n_per_topic_df["products"] = company_name.strip()
    top_n_per_topic_df["keywords"] = company_keywords.strip()

    return result_df, top_n_per_topic_df


def _analyze_all_real_estate_company_topic(company_df, time_range_key, time_range_value, batch_uid):
    try:
        end_date = datetime.datetime.now()
        start_date = end_date + datetime.timedelta(days=-time_range_value)
        start_date = start_date.strftime("%Y-%m-%d")
        end_date = end_date.strftime("%Y-%m-%d")
        logger.info("Extract real estate company topic data from {} to {}".format(start_date, end_date))

        company_df = company_df[(company_df["pub_date"] >= start_date) & (company_df["pub_date"] <= end_date)]

        with open(config.real_estate_company_path) as f:
            companies = f.readlines()

        companies = [company.strip() for company in companies]
        logger.info("Real estate company number: {}".format(len(companies)))

        merged_result_df = pd.DataFrame([])
        merged_top_n_per_topic_df = pd.DataFrame([])

        for company in companies:
            result_df, top_n_per_topic_df = _analyze_real_estate_company_topic(company, company_df, end_date)
            if result_df is None or top_n_per_topic_df is None:
                continue

            merged_result_df = pd.concat([merged_result_df, result_df])
            merged_top_n_per_topic_df = pd.concat([merged_top_n_per_topic_df, top_n_per_topic_df])
            logger.info("{} negative topics is extracted".format(company))

        RWFlocker.lock(RWFlocker.WRITE)

        result_path, top_n_per_topic_path = get_real_estate_company_csv_path(time_range_key, batch_uid)

        merged_result_df.to_csv(result_path, sep=',', header=True, index=False)
        merged_top_n_per_topic_df.to_csv(top_n_per_topic_path, sep=',', header=True, index=False)

        RWFlocker.unlock()

        logger.info("All negative topics of real estate companies are extracted")

    except Exception as e:
        logger.error(e)
    finally:
        RWFlocker.unlock()


def _analyze_real_estate_business_topic(company_df, time_range_key, time_range_value, batch_uid):
    try:
        end_date = datetime.datetime.now()
        start_date = end_date + datetime.timedelta(days=-time_range_value)
        start_date = start_date.strftime("%Y-%m-%d")
        end_date = end_date.strftime("%Y-%m-%d")
        logger.info("Extract real estate business topic data from {} to {}".format(start_date, end_date))

        company_df = company_df[(company_df["pub_date"] >= start_date) & (company_df["pub_date"] <= end_date)]

        result_df, top_n_per_topic_df = LDAModel.process(company_df, end_date, False, False)
        if result_df is None:
            logger.error("LDAModel failed when analyzing all real estate data: result df is empty")
            return

        if top_n_per_topic_df is None:
            logger.error("LDAModel failed when analyzing all real estate data: top N df is empty")
            return

        RWFlocker.lock(RWFlocker.WRITE)

        result_path, top_n_per_topic_path = get_real_estate_business_csv_path(time_range_key, batch_uid)

        result_df.to_csv(result_path, sep=',', header=True, index=False)
        top_n_per_topic_df.to_csv(top_n_per_topic_path, sep=',', header=True, index=False)

        RWFlocker.unlock()

        logger.info("All negative topics of real estate business are extracted")

    except Exception as e:
        logger.error(e)
    finally:
        RWFlocker.unlock()


def _get_sentiment_labels_data(spark):
    try:
        sql = "select p.uid, p.label, c.label as new_label " \
              "from " + config.predict_table + " as p left join " + config.calibrated_table + " as c on p.uid = c.uid"

        sentiment_df = spark.sql(sql).toPandas()
        for idx in sentiment_df.index:
            if not pd.isnull(sentiment_df.loc[idx, "new_label"]):
                sentiment_df.loc[idx, 'label'] = sentiment_df.loc[idx, 'new_label']

        sentiment_df.drop(['new_label'], axis=1, inplace=True)
        sentiment_df['label'] = sentiment_df['label'].astype(int)

        logger.info("sentiment df shape: {}".format(sentiment_df.shape))

        return sentiment_df
    except Exception as e:
        logger.error(e)
        return None


def _preprocess_real_estate_company_data(start_date, end_date, batch_companies=[], batch_uid=None):

    try:
        data_path = config.topic_local_real_estate_data_path
        if not os.path.exists(data_path):
            raise Exception("{} does not exist".format(data_path))

        df = pd.read_csv(data_path, sep=',')
        logger.info("Real estate company df shape: {}".format(df.shape))

        data_path = config.sentiment_data_file_path
        if not os.path.exists(data_path):
            raise Exception("{} does not exist".format(data_path))

        sentiment_df = pd.read_csv(data_path, sep=',')
        logger.info("Sentiment df shape: {}".format(sentiment_df.shape))

        logger.info("Start to preprocess real estate company topic data from {} to {}".format(start_date, end_date))

        if len(batch_companies) > 0:
            company_df = pd.DataFrame([], columns=["crawler_id", "content", "products", "keywords", "sentiment_id",
                                                   "pub_date", "company", "start_date", "end_date"])
            for company in batch_companies:
                tmp_df = df[df.products.str.contains(company)]
                if tmp_df.shape[0] == 0:
                    continue

                company_df = pd.concat([company_df, tmp_df])
        else:
            company_df = df

        if company_df.shape[0] == 0:
            return

        company_df = company_df[(company_df["pub_date"] >= start_date) & (company_df["pub_date"] <= end_date)]

        neg_indices = []
        for idx in company_df.index:
            content = company_df.loc[idx, "content"]
            if pd.isnull(content) or len(content.strip()) == 0:
                continue

            products = company_df.loc[idx, "products"]
            if pd.isnull(products):
                continue

            if products.find("ke_research") >= 0:
                neg_indices.append(idx)
                continue

            if products.find("[") >= 0 and products.find("]") > 0:
                products = json.loads(products)
            else:
                products = re.split("[,，]", products)

            company_names = [product for product in products if product.strip() != config.CompanyType.REAL_ESTATE.value]
            if len(company_names) > 0:
                related = check_related_company_v2(content, company_names[0].strip())

                if not related:
                    logger.info("{}: {} has unrelated content".format(company_names[0], company_df.loc[idx, "crawler_id"]))
                    continue

            uid = company_df.loc[idx, "sentiment_id"]

            if not pd.isnull(uid) and uid != "error" and uid != "-1":
                sentiment = sentiment_df[sentiment_df["uid"] == uid]
                if sentiment.values.shape[0] > 0:
                    label = sentiment.label.values[0]
                    if label != -1 and label != -200:
                        continue
                else:
                    has_risk, _ = has_company_risk_keywords(content)
                    if not has_risk:
                        continue
            else:
                has_risk, _ = has_company_risk_keywords(content)
                if not has_risk:
                    continue

            neg_indices.append(idx)

        logger.info("Negative text number is {} between {} and {}".format(len(neg_indices), start_date, end_date))

        if len(neg_indices) == 0:
            return

        company_df = company_df.loc[neg_indices, :]

        if batch_uid is None:
            tmp_path = os.path.join(config.topic_data_tmp_path, \
                                    config.CompanyType.REAL_ESTATE.value + "_" + start_date + ".csv")
        else:
            tmp_path = os.path.join(config.topic_data_tmp_path, \
                                    config.CompanyType.REAL_ESTATE.value + "_" + start_date + "_" + batch_uid + ".csv")

        company_df.to_csv(tmp_path, sep=',', header=True, index=False)

        logger.info("Preprocessed real estate company df shape is {} between {} and {}".format(company_df.shape, start_date, end_date))
    except Exception as e:
        logger.error(e)


def preprocess_real_estate_company_data(batch_companies, batch_uid, time_range=config.GetResultTimeRange.ONE_MONTH.value):
    if not os.path.exists(config.topic_data_tmp_path):
        os.makedirs(config.topic_data_tmp_path)

    end_date = datetime.datetime.now()
    start_date = end_date + \
                 datetime.timedelta(
                     days=-config.get_result_time_range[time_range])

    start_date = start_date.strftime("%Y-%m-%d")
    end_date = end_date.strftime("%Y-%m-%d")
    logger.info("start date is {}".format(start_date))
    logger.info("end date is {}".format(end_date))

    date_range = pd.date_range(start=start_date, end=end_date, freq='W').tolist()
    if pd.Timestamp(start_date) < date_range[0]:
        date_range.insert(0, pd.Timestamp(start_date))

    if pd.Timestamp(end_date) > date_range[-1]:
        date_range.append(pd.Timestamp(end_date))

    date_range = [pd.to_datetime(x) for x in date_range]
    logger.info(date_range)

    df_paths = []
    _processes = []
    for i, date in enumerate(date_range):
        if i == len(date_range) - 1:
            break

        start = date_range[i]
        if i > 0:
            start = start + timedelta(days=1)
        end = date_range[i + 1]

        start = start.strftime("%Y-%m-%d")
        end = end.strftime("%Y-%m-%d")

        logger.info("Process[{}]: start date={}, end date={}".format(i, start, end))

        p = Process(target=_preprocess_real_estate_company_data, args=(start, end, batch_companies, batch_uid, ))
        p.start()
        _processes.append(p)

        if batch_uid is None:
            tmp_path = os.path.join(config.topic_data_tmp_path, \
                                    config.CompanyType.REAL_ESTATE.value + "_" + start + ".csv")
        else:
            tmp_path = os.path.join(config.topic_data_tmp_path, \
                                    config.CompanyType.REAL_ESTATE.value + "_" + start + "_" + batch_uid + ".csv")

        df_paths.append(tmp_path)

    for p in _processes:
        p.join()

    all_df = None
    for path in df_paths:
        if not os.path.exists(path):
            logger.error("{} does not exists !".format(path))
            continue

        df = pd.read_csv(path,  sep=',')
        if all_df is None:
            all_df = df
        else:
            all_df = pd.concat([all_df, df], axis=0)

        os.remove(path)

    if all_df is not None and all_df.shape[0] > 0:
        if batch_uid is None:
            all_df.to_csv(config.preprocessed_real_estate_data_path, sep=',', header=True, index=False)
        else:
            p = config.preprocessed_real_estate_data_path.rfind(".")
            all_df.to_csv(config.preprocessed_real_estate_data_path[:p] + "_" + batch_uid + ".csv",
                          sep=',', header=True, index=False)


def analyze_topic(_spark_config, spark_user, message):
    try:
        spark_config = _spark_config.copy()
        spark_config.update({"spark.py.files": [os.path.join(config.cwd_path, "dedup_with_tfifd.py")]})

        spark_manager = PySparkMgr(spark_config, "topic-analysis-pyspark")
        spark, sc = spark_manager.start(spark_user)
        if spark is None:
            raise Exception("Fail to launch spark session")
        logger.info(spark)

        company_type = message['type']
        logger.info("start to analyze topic for {}".format(company_type))

        today = datetime.datetime.now().strftime('%Y-%m-%d')

        if company_type == config.MessageType.MARKET_TOPIC.value:
            _analyze_market_topic(spark, sc, today)
        elif company_type == config.MessageType.PUBLIC_RELATION_TOPIC.value:
            _analyze_public_relation_topic(spark, sc, today)

    except Exception as e:
        logger.error(e)
    finally:
        RWFlocker.unlock()


def analyze_real_estate_company_topic(time_range, topic_type, batch_uid=None):
    try:
        if batch_uid is None:
            data_path = config.preprocessed_real_estate_data_path
        else:
            p = config.preprocessed_real_estate_data_path.rfind(".")
            data_path = config.preprocessed_real_estate_data_path[:p] + "_" + batch_uid + ".csv"

        if not os.path.exists(data_path):
            raise Exception("Extracting real estate company topic error: no preproccessed company data")

        company_df = pd.read_csv(data_path, sep=',')
        logger.info("Topic type is {}".format(topic_type))

        if topic_type == TopicType.COMPANY.value:
            _analyze_all_real_estate_company_topic(company_df, time_range, config.get_result_time_range[time_range])
        elif topic_type == TopicType.BUSINESS.value:
            _analyze_real_estate_business_topic(company_df, time_range, config.get_result_time_range[time_range], batch_uid)

    except Exception as e:
        logger.error(e)


def _execute():
    logger.info("start topic analysis daemon")

    batch_uid = None
    data = {}

    while True:
        try:
            message = _redis.pop_data(config.topic_message_key, True)
            logger.info(message)

            if 'type' not in message:
                raise Exception("Data miss type field")

            if message['type'] not in [config.MessageType.MARKET_TOPIC.value,
                                       config.MessageType.PUBLIC_RELATION_TOPIC.value,
                                       config.MessageType.REAL_ESTATE_COMPANY_TOPIC.value]:
                raise Exception("Unrecognized type field")

            end_date = datetime.datetime.now()
            start_date = end_date + \
                         datetime.timedelta(
                             days=-config.get_result_time_range[config.GetResultTimeRange.ONE_WEEK.value])

            start_date = start_date.strftime("%Y-%m-%d")
            end_date = end_date.strftime("%Y-%m-%d")

            if "fetch_data" in message:
                if message['type'] == config.MessageType.MARKET_TOPIC.value:
                    fetch_market_data(config.CompanyType.MARKET.value, start_date, end_date)
                    logger.info("Market data fetched, prepared for analysis")

                elif message['type'] == config.MessageType.PUBLIC_RELATION_TOPIC.value:
                    fetch_public_relation_data(config.CompanyType.PUBLIC_RELATION.value, start_date, end_date)
                    logger.info("Public relation data fetched, prepared for analysis")

                elif message['type'] == config.MessageType.REAL_ESTATE_COMPANY_TOPIC.value:

                    end_date = datetime.datetime.now()
                    start_date = end_date + \
                                 datetime.timedelta(
                                     days=-config.get_result_time_range[config.GetResultTimeRange.ONE_MONTH.value])

                    start_date = start_date.strftime("%Y-%m-%d")
                    end_date = end_date.strftime("%Y-%m-%d")

                    p = Process(target=concurrent_fetch_data_v3, args=(start_date, end_date,))
                    p.start()
                    p.join()
                    logger.info("Real estate company data fetched, prepared for analysis")

                    fetch_real_estate_company_data(start_date, end_date)
                    p = Process(target=fetch_sentiment_data, args=())
                    p.start()
                    p.join()

            topic_processes = []

            if message['type'] in [config.MessageType.MARKET_TOPIC.value,
                                   config.MessageType.PUBLIC_RELATION_TOPIC.value]:
                p = Process(target=analyze_topic,
                            args=(config.spark_config, config.spark_user, message,))
                p.start()
                topic_processes.append(p)
            else:

                if "batch_uid" in message:
                    batch_uid = message["batch_uid"]

                if "batch_companies" in message:
                    batch_companies = message["batch_companies"]
                    logger.info("batch_companies: {}".format(batch_companies))

                if "batch_time_range" in message:
                    batch_time_range = message["batch_time_range"]

                preprocess_real_estate_company_data(batch_companies, batch_uid, batch_time_range)

                if batch_uid is None:
                    for time_range in config.get_result_time_range.keys():
                        for topic_type in TopicType:
                            p = Process(target=analyze_real_estate_company_topic,
                                        args=(time_range, topic_type.value, ))
                            p.start()
                            topic_processes.append(p)
                else:
                    p = Process(target=analyze_real_estate_company_topic,
                                args=(batch_time_range, TopicType.COMPANY.value, batch_uid, ))
                    p.start()
                    topic_processes.append(p)

            for p in topic_processes:
                p.join()

            set_topic_last_process_day(datetime.datetime.now().strftime('%Y-%m-%d'), message['type'])

            if batch_uid is not None:
                data["status"] = "success"
                data["datetime"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

                if _redis.push_data(data, batch_uid) < 0:
                    logger.error("Fail to push data to redis queue: {}".format(batch_uid))

                data.clear()
                batch_uid = None

        except Exception as e:
            logger.error(e)

            if batch_uid is not None:
                data["status"] = "failed"
                data["datetime"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

                if _redis.push_data(data, batch_uid) < 0:
                    logger.error("Fail to push data to redis queue: {}".format(batch_uid))

                data.clear()
                batch_uid = None


if __name__ == "__main__":
    _execute()
