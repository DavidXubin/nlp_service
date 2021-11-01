import os
import re
import sys
import json
import pandas as pd
from multiprocessing import Process
import config
from text2vec import Similarity
from utils import get_logger
from pyspark_manager import PySparkMgr

SIMILARITY_THRESHOLD = 0.997

CONCURRENCY = 5


logger = None
if logger is None:
    logger = get_logger("dedup-company-data")


dedup_spark_config = {
    "spark.driver.memory": "10g",
    "spark.executor.memory": "5g",
    "spark.driver.maxResultSize": "5g",
    "spark.executor.cores": 3,
    "spark.executor.instances": 2
}


def get_company_df(company_df, spark, sc):

    company_df.index = [x for x in range(company_df.shape[0])]

    spark_company_df = spark.createDataFrame(company_df)

    doc_matrix = {}
    max_id = company_df.shape[0]

    for idx in company_df.index:

        if idx == max_id - 1:
            break

        crawler_id = company_df.loc[idx, "crawler_id"]

        doc_matrix[crawler_id] = company_df.iloc[idx + 1: max_id].content.values

    return spark_company_df, sc.broadcast(doc_matrix)


def check_dup_doc(iterator, doc_matrix):

    doc_matrix = doc_matrix.value

    sim = Similarity(w2v_path="/data/software/nlp_w2v/light_Tencent_AILab_ChineseEmbedding.bin",
                     w2v_kwargs={'binary': True}, sequence_length='auto')

    for doc in iterator:

        if doc["crawler_id"] not in doc_matrix:
            yield (doc["crawler_id"], False)
            continue

        doc_list = doc_matrix[doc["crawler_id"]]

        crawler_id = doc["crawler_id"]

        found_dup = False
        for content in doc_list:
            s = sim.get_score(doc["content"], content)

            if s > SIMILARITY_THRESHOLD:
                found_dup = True
                break

        yield (crawler_id, found_dup)


def cal_dup_rate(dup_maxtric):

    dup_cnt = sum([1 for x in dup_maxtric if x[1]])

    return dup_cnt / len(dup_maxtric) if len(dup_maxtric) != 0 else 0


def dedup_companies(filepath, start_date, company_range, partition_id):

    try:

        spark_manager = PySparkMgr(dedup_spark_config, "dedup-analysis-pyspark")
        spark, sc = spark_manager.start(config.spark_user)
        if spark is None:
            raise Exception("Fail to launch spark session")
        logger.info(spark)

        all_df = pd.read_csv(filepath, sep=',')

        company_dup_rate = {}

        dedup_company_id = []

        for company in company_range:
            logger.info("Check {}".format(company))

            company_df = all_df[(all_df.company.str.contains(company)) & (all_df.pub_date >= start_date)]

            spark_company_df, sc_doc_matrix = get_company_df(company_df, spark, sc)

            dup_maxtric = spark_company_df.rdd.mapPartitions(lambda x: check_dup_doc(x, sc_doc_matrix)).collect()

            company_dup_rate[company] = cal_dup_rate(dup_maxtric)

            dedup_company_id.extend([x[0] for x in dup_maxtric if not x[1]])

        dup_rate_file = os.path.join(config.topic_data_tmp_path, "dup_rate_" + str(partition_id) + "_" + start_date + ".txt")
        with open(dup_rate_file, 'w') as f:
            for k, v in company_dup_rate.items():
                f.write(k + ":" + str(v))
                f.write("\n")

        dedup_file = os.path.join(config.topic_data_tmp_path, "dedup_" + str(partition_id) + "_" + start_date + ".txt")
        with open(dedup_file, 'w') as f:
            f.write("\n".join(dedup_company_id))

    except Exception as e:
        logger.error("Company partition[{}] has error: {}".format(partition_id, e))


def _execute(filepath, start_date):

    if not os.path.exists(config.topic_data_tmp_path):
        os.makedirs(config.topic_data_tmp_path)

    if not os.path.exists(filepath):
        raise Exception("{} does not exist".format(filepath))

    all_df = pd.read_csv(filepath, sep=',')
    logger.info("Real estate company df shape: {}".format(all_df.shape))

    all_df = all_df[all_df.pub_date >= start_date]

    companies = all_df.company.values.tolist()
    company_set = set()

    for company in companies:
        if company.find("[") >= 0 and company.find("]") > 0:
            items = json.loads(company)
        else:
            items = re.split("[,，]", company)

        company_name = [item for item in items if item.strip() != config.CompanyType.REAL_ESTATE.value][0]
        company_set.add(company_name)

    partition_cnt = int(len(company_set) / CONCURRENCY)
    companies = list(company_set)

    _processes = []
    for i in range(CONCURRENCY):
        if i < CONCURRENCY - 1:
            company_range = companies[i * partition_cnt: (i + 1) * partition_cnt]
        else:
            company_range = companies[i * partition_cnt: len(companies)]

        p = Process(target=dedup_companies, args=(filepath, start_date, company_range, i, ))
        p.start()
        _processes.append(p)

    for p in _processes:
        p.join()

    total_dup_rate = []
    total_dedup_id = []

    for i in range(CONCURRENCY):

        dup_rate_file = os.path.join(config.topic_data_tmp_path, "dup_rate_" + str(i) + "_" + start_date + ".txt")
        with open(dup_rate_file, 'r') as f:
            dup_rate = f.readlines()

        total_dup_rate.extend(dup_rate)

        dedup_file = os.path.join(config.topic_data_tmp_path, "dedup_" + str(i) + "_" + start_date + ".txt")
        with open(dedup_file, 'r') as f:
            dedup_id = f.readlines()

        total_dedup_id.extend(dedup_id)

    with open(os.path.join(config.topic_data_tmp_path, "dup_rate_" + start_date + ".txt"), 'w') as f:
        f.write("\n".join(total_dup_rate))

    with open(os.path.join(config.topic_data_tmp_path, "dedup_" + start_date + ".txt"), 'w') as f:
        f.write("\n".join(total_dedup_id))


if __name__ == "__main__":
    _execute(sys.argv[1], sys.argv[2])
