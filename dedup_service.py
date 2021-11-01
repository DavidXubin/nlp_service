import os
import datetime
import numpy as np
import pandas as pd
from multiprocessing import Process
from utils import get_logger, get_stopwords
from redis_db import RedisDBWrapper
from pyspark_manager import PySparkMgr
from dedup_with_tfifd import dedup_get_duplication_maxtrix
#from dedup_with_bert import dedup as bert_dedup
import config

logger = get_logger("dedup-service")

_redis = RedisDBWrapper()

dedup_spark_config = {
    "spark.driver.memory": "30g",
    "spark.executor.memory": "20g",
    "spark.driver.maxResultSize": "10g",
    "spark.rpc.message.maxSize": "2000",
    "spark.executor.cores": 3,
    "spark.executor.instances": 4,
    "spark.py.files": [os.path.join(config.cwd_path, "dedup_with_tfifd.py")]
}


def get_duplication_by_tfidf(news_df, time_range):
    spark_manager = PySparkMgr(dedup_spark_config, "dedup-pyspark-" + time_range)
    spark, sc = spark_manager.start(config.spark_user, False)
    if spark is None:
        raise Exception("Fail to launch spark session")
    logger.info(spark)

    dup_matrix = dedup_get_duplication_maxtrix(spark, sc, news_df, get_stopwords())

    dup_doc_list = list(set([x[0] for x in dup_matrix if x[1]]))

    logger.info("Duplicated data size in {}: {}".format(time_range, len(dup_doc_list)))

    return dup_doc_list


def get_duplication_by_bert(news_df, time_range):
    #_, dup_docs = bert_dedup(news_df)

    #return dup_docs
    return []


def dedup(time_range, method):
    t1 = datetime.datetime.now()

    try:
        if not os.path.exists(config.raw_data_path):
            raise Exception("Extracting data error: no data downloaded")

        data_df = pd.read_csv(config.raw_data_path, sep=',')
        data_df.index = np.arange(data_df.shape[0])

        end_date = datetime.datetime.now()
        start_date = end_date + datetime.timedelta(days=-config.get_result_time_range[time_range])
        start_date = start_date.strftime("%Y-%m-%d")
        end_date = end_date.strftime("%Y-%m-%d")
        logger.info("Extract data from {} to {}".format(start_date, end_date))

        data_df = data_df[(~pd.isnull(data_df["crawler_id"])) & (data_df["crawler_id"].str.strip().str.len() > 0)]
        data_df.drop_duplicates(["crawler_id"], inplace=True)
        data_df = data_df[(data_df["pub_date"] >= start_date) & (data_df["pub_date"] <= end_date)]
        data_df = data_df[(~pd.isnull(data_df["content"])) & (data_df["content"].str.strip().str.len() > 0)]

        logger.info("Before dedup data shape in {}: {}".format(time_range, data_df.shape))

        tmp_folder = config.duplicated_doc_path[0: config.duplicated_doc_path.rfind("/")]
        if not os.path.exists(tmp_folder):
            os.makedirs(tmp_folder)

        dup_doc_path = config.duplicated_doc_path + time_range

        if method == config.DedupMethod.BERT:
            dup_doc = get_duplication_by_bert(data_df, time_range)
        else:
            dup_doc = get_duplication_by_tfidf(data_df, time_range)

        with open(dup_doc_path, "w") as f:
            f.write(",".join(dup_doc))

    except Exception as e:
        logger.error(e)
    finally:
        t2 = datetime.datetime.now()

        elapsed = (t2 - t1).seconds // 60
        logger.info("dedup in {} takes {} minutes".format(time_range, elapsed))


def fetch_data(start_date, end_date):

    try:
        spark_manager = PySparkMgr(dedup_spark_config, "dedup-pyspark-get-data")
        spark, sc = spark_manager.start(config.spark_user)
        if spark is None:
            raise Exception("Fail to launch spark session")
        logger.info(spark)

        sql = "select _id as crawler_id, rb_parse as content, pub_date from " + config.dw_crawler_data_table + \
              " where pub_date >='" + start_date + "' and pub_date <= '" + end_date + "'"

        logger.info(sql)

        data_df = spark.sql(sql).toPandas()

        max_date = spark.sql("select max(pt) as max_date from spark_dw.dw_ke_bkjf_content_policy_news_da").toPandas()
        max_date = max_date.loc[0, "max_date"]

        sql = """select news_id as crawler_id, news_content as content, """ \
              """ publish_time as pub_date from {} where date(publish_time) >='{}' and """ \
              """ date(publish_time) <= '{}' and pt = '{}' and policy_type in ('41','31','32','33','34','35','36','37','38','51','52','55')""". \
            format("spark_dw.dw_ke_bkjf_content_policy_news_da", start_date, end_date, max_date)

        logger.info(sql)

        data_df = pd.concat([data_df, spark.sql(sql).toPandas()])

        if data_df.shape[0] > 0:
            tmp_folder = config.duplicated_doc_path[0: config.duplicated_doc_path.rfind("/")]
            if not os.path.exists(tmp_folder):
                os.makedirs(tmp_folder)

            data_df.to_csv(config.raw_data_path, sep=',', header=True, index=False)
    except Exception as e:
        logger.error("Fetching data for dedup failed: {}".format(e))


def _execute():

    logger.info("start deduplication service daemon")

    while True:
        try:
            message = _redis.pop_data(config.dedup_req_key, True)
            logger.info(message)

            if 'type' not in message:
                raise Exception("Data miss type field")

            if message['type'] != config.MessageType.DEDUP_DOC.value:
                raise Exception("Unrecognized type field")

            method = config.DedupMethod.TFIDF.value
            if "method" in message:
                method = message["method"]

            if "fetch_data" in message:
                end_date = datetime.datetime.now()
                start_date = end_date + \
                             datetime.timedelta(
                                 days=-config.get_result_time_range[config.GetResultTimeRange.ONE_WEEK.value])

                start_date = start_date.strftime("%Y-%m-%d")
                end_date = end_date.strftime("%Y-%m-%d")

                p = Process(target=fetch_data, args=(start_date, end_date, ))
                p.start()
                p.join()
                logger.info("Raw data fetched, prepared for dedup")

            t1 = datetime.datetime.now()

            dup_processes = []
            for time_range in config.get_result_time_range.keys():
                if time_range in [config.GetResultTimeRange.HALF_MONTH.value, config.GetResultTimeRange.ONE_MONTH.value]:
                    continue

                p = Process(target=dedup, args=(time_range, method, ))
                p.start()
                dup_processes.append(p)

            for p in dup_processes:
                p.join()

            t2 = datetime.datetime.now()

            elapsed = (t2 - t1).seconds // 60
            logger.info("dedup takes {} minutes".format(elapsed))
        except Exception as e:
            logger.error(e)


if __name__ == "__main__":
    _execute()
