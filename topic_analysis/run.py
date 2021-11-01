import os
import datetime
import pandas as pd
from tornado.log import app_log
from topic_analysis import celery_app
from utils import generate_uid
from redis_db import RedisDBWrapper
from rw_flock import RWFlocker
from config import CompanyType, MessageType, TopicMessageType, topic_message_key, spark_topic_message_key, dedup_req_key
from config import topic_local_market_data_path
from config import topic_local_market_result_path, topic_local_market_top_n_per_topic_path
from config import topic_local_public_relation_result_path, duplicated_doc_path
from config import topic_local_real_estate_result_path, topic_local_real_estate_top_n_per_topic_path
from config import GetResultTimeRange, DedupMethod

_redis = RedisDBWrapper()


def get_real_estate_business_csv_path(time_range, batch_uid=None):
    p = topic_local_real_estate_result_path.rfind(".")
    if p < 0:
        raise Exception("topic_local_real_estate_result_path is not correct")

    if batch_uid is None:
        result_path = topic_local_real_estate_result_path[:p] + "_business_" + \
                      time_range + topic_local_real_estate_result_path[p:]
    else:
        result_path = topic_local_real_estate_result_path[:p] + "_business_" + \
                      time_range + "_" + batch_uid + topic_local_real_estate_result_path[p:]

    p = topic_local_real_estate_top_n_per_topic_path.rfind(".")
    if p < 0:
        raise Exception("topic_local_real_estate_top_n_per_topic_path is not correct")

    if batch_uid is None:
        top_n_per_topic_path = topic_local_real_estate_top_n_per_topic_path[:p] + "_business_" + \
                               time_range + topic_local_real_estate_top_n_per_topic_path[p:]
    else:
        top_n_per_topic_path = topic_local_real_estate_top_n_per_topic_path[:p] + "_business_" + \
                               time_range + "_" + batch_uid + topic_local_real_estate_top_n_per_topic_path[p:]

    return result_path, top_n_per_topic_path


def get_real_estate_company_csv_path(time_range, batch_uid=None):
    p = topic_local_real_estate_result_path.rfind(".")
    if p < 0:
        raise Exception("topic_local_real_estate_result_path is not correct")

    if batch_uid is None:
        result_path = topic_local_real_estate_result_path[:p] + "_" + time_range + topic_local_real_estate_result_path[p:]
    else:
        result_path = topic_local_real_estate_result_path[:p] + "_" + time_range + "_" + batch_uid + \
                      topic_local_real_estate_result_path[p:]

    p = topic_local_real_estate_top_n_per_topic_path.rfind(".")
    if p < 0:
        raise Exception("topic_local_real_estate_top_n_per_topic_path is not correct")

    if batch_uid is None:
        top_n_per_topic_path = topic_local_real_estate_top_n_per_topic_path[:p] + "_" + time_range + \
                               topic_local_real_estate_top_n_per_topic_path[p:]
    else:
        top_n_per_topic_path = topic_local_real_estate_top_n_per_topic_path[:p] + "_" + time_range + "_" + batch_uid + \
                               topic_local_real_estate_top_n_per_topic_path[p:]

    return result_path, top_n_per_topic_path


def trigger_topic_analysis(message_type):
    if message_type not in [MessageType.MARKET_TOPIC.value,
                            MessageType.PUBLIC_RELATION_TOPIC.value,
                            MessageType.REAL_ESTATE_COMPANY_TOPIC.value]:

        app_log.error("No such message as {}".format(message_type))
        return

    message = {
        "type": message_type,
        "topic": TopicMessageType.ANALYZE_DATA.value,
        "fetch_data": True,
        "datetime": datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    }

    message_key = topic_message_key
    if message_type == MessageType.REAL_ESTATE_COMPANY_TOPIC.value:
        message_key = spark_topic_message_key

    if _redis.push_data(message, message_key) < 0:
        app_log.error("Fail to push data to redis queue: {}".format(message_key))


@celery_app.task
def trigger_dedup():

    message = {
        "type": MessageType.DEDUP_DOC.value,
        "fetch_data": True,
        "method": DedupMethod.TFIDF.value,
        "datetime": datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    }

    if _redis.push_data(message, dedup_req_key) < 0:
        app_log.error("Fail to push data to redis queue: {}".format(dedup_req_key))

    app_log.info("Prepare to dedup")


@celery_app.task
def fetch_data():
    app_log.info("Fetch texts from crawler")

    try:
        tmp_folder = topic_local_market_data_path[0: topic_local_market_data_path.rfind("/")]
        if not os.path.exists(tmp_folder):
            os.makedirs(tmp_folder)

        trigger_topic_analysis(MessageType.MARKET_TOPIC.value)
        app_log.info("Prepare to fetch market data")

        trigger_topic_analysis(MessageType.PUBLIC_RELATION_TOPIC.value)
        app_log.info("Prepare to fetch public relation data")

        trigger_topic_analysis(MessageType.REAL_ESTATE_COMPANY_TOPIC.value)
        app_log.info("Prepare to fetch real estate company data")

    except Exception as e:
        app_log.error(e)


def extract_topic_results(result_df, top_n_per_topic_df):

    all_topics = {"result": []}
    if result_df.values.shape[0] == 0:
        return all_topics

    topic_stats = {"topic_stats": []}
    total_cnt = result_df.count().crawler_id
    grouped = result_df.groupby("topic_id")

    for name, group in grouped:
        item = {"topic_id": name, "topic_words": group.topic_words.values[0].split(','),
                "doc_ids": group.crawler_id.values.tolist(), "top_n_doc_ids": [], "summary": ""}

        if "summary" in group.columns:
            item["summary"] = group.summary.values[0]

        all_topics["result"].append(item)

        topic_stats["topic_stats"].append({"topic": name, "ratio": len(group.crawler_id.values.tolist()) / total_cnt})

    if top_n_per_topic_df is not None and top_n_per_topic_df.values.shape[0] > 0:

        top_n_per_topic_df.topic_id = top_n_per_topic_df.topic_id.astype("int")

        for i, item in enumerate(all_topics["result"]):
            topic_id = int(item['topic_id'])

            doc_ids = top_n_per_topic_df[top_n_per_topic_df.topic_id == topic_id].sentiment_id_list.values.tolist()
            if len(doc_ids) == 0 or len(doc_ids) == 1 and doc_ids[0].strip() == '':
                continue

            all_topics['result'][i]['top_n_doc_ids'].extend(doc_ids)

            summary = top_n_per_topic_df[top_n_per_topic_df.topic_id == topic_id].summary.values[0]
            all_topics['result'][i]['summary'] = summary

    return {**topic_stats, **all_topics}


@celery_app.task
def fetch_results(company_type, time_range):
    app_log.info("Fetch topic analysis results")

    resp_header = {"status": "OK", "message": ""}

    try:
        if time_range not in [GetResultTimeRange.ONE_WEEK.value, GetResultTimeRange.HALF_MONTH.value,
                              GetResultTimeRange.ONE_MONTH.value, GetResultTimeRange.ONE_DAY.value,
                              GetResultTimeRange.THREE_DAYS.value]:
            raise Exception("Fetching topic time range is incorrect")

        company_type = company_type.strip()

        if company_type == CompanyType.MARKET.value:
            result_path = topic_local_market_result_path
            top_n_per_topic_path = topic_local_market_top_n_per_topic_path
        elif company_type == CompanyType.PUBLIC_RELATION.value:
            result_path = topic_local_public_relation_result_path
            top_n_per_topic_path = None
        else:
            if company_type == CompanyType.REAL_ESTATE.value:
                result_path, top_n_per_topic_path = get_real_estate_business_csv_path(time_range)
            else:\
                result_path, top_n_per_topic_path = get_real_estate_company_csv_path(time_range)

        app_log.info("Result file is {}".format(result_path))

        if not os.path.exists(result_path):
            raise Exception("{} does not exist".format(result_path))

        RWFlocker.lock(RWFlocker.READ)

        result_df = pd.read_csv(result_path,  sep=',')
        app_log.info("result_df shape is {}".format(result_df.shape))

        top_n_per_topic_df = None

        if top_n_per_topic_path is not None:
            if not os.path.exists(top_n_per_topic_path):
                raise Exception("{} does not exist".format(top_n_per_topic_path))

            top_n_per_topic_df = pd.read_csv(top_n_per_topic_path,  sep=',')
            app_log.info("top_n_per_topic_df shape is {}".format(top_n_per_topic_df.shape))

        RWFlocker.unlock()

        if company_type not in [CompanyType.MARKET.value, CompanyType.PUBLIC_RELATION.value,
                                CompanyType.REAL_ESTATE.value]:
            result_df = result_df[result_df["products"].str.contains(company_type, na=False)]
            top_n_per_topic_df = top_n_per_topic_df[top_n_per_topic_df["products"].str.contains(company_type,  na=False)]

        all_topics = extract_topic_results(result_df, top_n_per_topic_df)
        return {**resp_header, **all_topics}

    except Exception as e:
        app_log.error(e)

        resp_header["status"] = "ERROR"
        resp_header["message"] = str(e)
        all_topics = {"result": []}
        return {**resp_header, **all_topics}
    finally:
        RWFlocker.unlock()


@celery_app.task
def fetch_batch_results(companies, time_range, batch_method="business"):
    app_log.info("Fetch topic analysis results")

    resp_header = {"status": "OK", "message": ""}

    try:
        if time_range not in [GetResultTimeRange.ONE_WEEK.value, GetResultTimeRange.HALF_MONTH.value,
                              GetResultTimeRange.ONE_MONTH.value, GetResultTimeRange.ONE_DAY.value,
                              GetResultTimeRange.THREE_DAYS.value]:
            raise Exception("Fetching topic time range is incorrect")

        batch_uid = generate_uid(companies)

        if isinstance(companies, str):
            companies = companies.split(",")

        message = {
            "type": MessageType.REAL_ESTATE_COMPANY_TOPIC.value,
            "topic": TopicMessageType.ANALYZE_DATA.value,
            "batch_companies": companies,
            "batch_uid": batch_uid,
            "batch_time_range": time_range,
            "batch_method": batch_method,
            "datetime": datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        }

        if _redis.push_data(message, spark_topic_message_key) < 0:
            app_log.error("Fail to push data to redis queue: {}".format(spark_topic_message_key))

        message = _redis.pop_data(batch_uid, True)
        app_log.info(message)

        if message["batch_method"] == "business":
            result_path, top_n_per_topic_path = get_real_estate_business_csv_path(time_range, batch_uid)
        else:
            result_path, top_n_per_topic_path = get_real_estate_company_csv_path(time_range, batch_uid)

        if not os.path.exists(result_path):
            raise Exception("{} does not exist".format(result_path))

        RWFlocker.lock(RWFlocker.READ)

        result_df = pd.read_csv(result_path,  sep=',')
        app_log.info("result_df shape is {}".format(result_df.shape))

        top_n_per_topic_df = None

        if top_n_per_topic_path is not None:
            if not os.path.exists(top_n_per_topic_path):
                raise Exception("{} does not exist".format(top_n_per_topic_path))

            top_n_per_topic_df = pd.read_csv(top_n_per_topic_path,  sep=',')
            app_log.info("top_n_per_topic_df shape is {}".format(top_n_per_topic_df.shape))

        RWFlocker.unlock()

        all_topics = {}

        if message["batch_method"] == "business":
            all_topics = extract_topic_results(result_df, top_n_per_topic_df)
        else:
            for company in companies:
                company_result_df = result_df[result_df["products"].str.contains(company)]
                company_top_n_per_topic_df = top_n_per_topic_df[top_n_per_topic_df["products"].str.contains(company)]

                company_topics = extract_topic_results(company_result_df, company_top_n_per_topic_df)

                all_topics[company] = company_topics

        return {**resp_header, **all_topics}

    except Exception as e:
        app_log.error(e)

        resp_header["status"] = "ERROR"
        resp_header["message"] = str(e)
        all_topics = {"result": []}
        return {**resp_header, **all_topics}
    finally:
        RWFlocker.unlock()


@celery_app.task
def fetch_dedup_results(time_range):
    app_log.info("Fetch duplicated doc results")

    resp_header = {"status": "OK", "message": ""}

    try:
        if time_range not in [GetResultTimeRange.ONE_WEEK.value, GetResultTimeRange.HALF_MONTH.value,
                              GetResultTimeRange.ONE_DAY.value, GetResultTimeRange.THREE_DAYS.value,
                              GetResultTimeRange.ONE_MONTH.value]:
            raise Exception("Fetching dedup time range is incorrect")

        if time_range in [GetResultTimeRange.HALF_MONTH.value, GetResultTimeRange.ONE_MONTH.value]:
            time_range = GetResultTimeRange.ONE_WEEK.value

        result_path = duplicated_doc_path + time_range

        app_log.info("Result file is {}".format(result_path))

        if not os.path.exists(result_path):
            raise Exception("{} does not exist".format(result_path))

        RWFlocker.lock(RWFlocker.READ)

        with open(result_path, "r") as f:
            duplicated_doc_ids = f.read()

        result_data = {
            "result": duplicated_doc_ids.split(",")
        }

        return {**resp_header, **result_data}
    except Exception as e:
        app_log.error(e)

        resp_header["status"] = "ERROR"
        resp_header["message"] = str(e)
        result_data = {"result": []}
        return {**resp_header, **result_data}
    finally:
        RWFlocker.unlock()
