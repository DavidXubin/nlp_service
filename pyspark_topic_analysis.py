import os
import re
import json
import jieba
import datetime
import heapq
import config
import pandas as pd
import numpy as np
from enum import Enum
from multiprocessing import Process
from pyspark.sql.functions import pandas_udf
from pyspark.sql.functions import PandasUDFType
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from sklearn.feature_extraction.text import CountVectorizer as sklearnCountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from pyspark.sql import Row
from pyspark.ml.feature import CountVectorizer, HashingTF, IDF
from pyspark.ml.clustering import LDA
from utils import get_stopwords, load_local_dict
from pyspark_manager import PySparkMgr
from rw_flock import RWFlocker
from redis_db import RedisDBWrapper
from topic_analysis.run import get_real_estate_company_csv_path, get_real_estate_business_csv_path
from topic_analysis_service import fetch_real_estate_company_data, concurrent_fetch_data_v3, fetch_sentiment_data
from topic_analysis_service import preprocess_real_estate_company_data, fetch_ke_research_data
from topic_analysis_service import fetch_hot_company_data, fetch_oversea_topic_data
from utils import get_logger
from get_text_summay import extract_business_topic_texts_summaries, extract_company_topic_texts_summaries
import dedup_with_tfifd

logger = None
if logger is None:
    logger = get_logger("spark-topic-analysis-service")


class TopicType(Enum):
    COMPANY = "company"
    BUSINESS = "business"


topic_spark_config = {
    "spark.driver.memory": "10g",
    "spark.executor.memory": "5g",
    "spark.driver.maxResultSize": "5g",
    "spark.executor.cores": 1,
    "spark.executor.instances": 4,
    "spark.py.files": [os.path.join(config.cwd_path, "dedup_with_tfifd.py")]
}

stopwords_list = None
local_dict_words = None

top_n_texts_per_topic = 3

_redis = RedisDBWrapper()


topic_df_schema = StructType([
    StructField("crawler_id", StringType()),
    StructField("products", StringType()),
    StructField("keywords", StringType()),
    StructField("sentiment_id", StringType()),
    StructField("company", StringType()),
    StructField("topic_id", StringType()),
    StructField("topic_words", StringType()),
    StructField("word_counts", IntegerType()),
    StructField("datetime", StringType()),
])


top_topic_schema = StructType([
    StructField("topic_id", StringType()),
    StructField("sentiment_id_list", StringType()),
    StructField("datetime", StringType()),
    StructField("products", StringType()),
    StructField("keywords", StringType())
])


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
            if item not in stopwords and re.match(r'-?\d+\.?\d*', item) is None and len(item.strip()) > 1:
                seg_words.append(item)
        seg_words = ' '.join(seg_words)
        seg_corpus.append(seg_words)
    return seg_corpus


def build_count_vector(seg_corpus):
    """
    build word-frequency matrix
    :param seg_corpus: segmented corpus
    :return: count vector, words list
    """
    try:
        vectorizer = sklearnCountVectorizer(min_df=10,
                                            max_df=0.90,
                                            max_features=500
                                            )
        cv = vectorizer.fit_transform(seg_corpus)
        cv_words = np.array(vectorizer.get_feature_names())
    except Exception as e:
        print(e)
        vectorizer = sklearnCountVectorizer(max_features=500)

        cv = vectorizer.fit_transform(seg_corpus)
        cv_words = np.array(vectorizer.get_feature_names())

    return cv, cv_words


def lda_fit_cluster(cv, n_cluster, text_df, cv_words):
    """
    build lda model and get topic words
    :param cv: count vector
    :param n_cluster: best number of lda cluster
    :param text_df: dataframe
    :param cv_words: word list
    :return: pandas dataframe
    """
    lda_model = LatentDirichletAllocation(n_components=n_cluster,
                                          learning_method='batch',
                                          max_iter=1000,
                                          random_state=42,
                                          evaluate_every=200)
    lda_type = lda_model.fit_transform(cv)
    text_df["topic_id"] = np.argmax(lda_type, axis=1)
    topic_words_dict = {}
    for i in range(len(lda_model.components_)):
        topic_words_dict[i] = [str(x) for x in cv_words[np.argsort(lda_model.components_[i])[-10:]]]
    text_df["topic_words"] = text_df.topic_id.apply(lambda x: topic_words_dict[x])
    return text_df


def find_top_n(text, topic_words):
    return sum([text.count(x) for x in topic_words])


def select_top_n_docs(text_df, n_cluster):
    """
    select typical documents of each list of keywords
    :param text_df: pandas dataframe
    :param n_cluster: number of cluster centers
    :param get_word_count: whether to count word frequency
    :return: type name and uid
    """
    try:
        sorted_data = text_df.sort_values('word_counts', ascending=False)
        sorted_data.topic_id = sorted_data.topic_id.apply(lambda x: int(x))
        max_topic_id = sorted_data.topic_id.max()

        rst_dict = dict()
        for i in range(max_topic_id + 1):
            topic_data = sorted_data[sorted_data.topic_id == i]

            if topic_data.shape[0] == 0:
                continue

            if topic_data.shape[0] >= top_n_texts_per_topic:
                rst_dict[i] = topic_data.head(top_n_texts_per_topic).crawler_id.values
            else:
                rst_dict[i] = topic_data.crawler_id.values

        return rst_dict
    except Exception as e:
        print(e)
        return None


def find_cluster_center(cv, start=1, end=10):
    """
    Search for best cluster center, use elbow point
    :param start: start number of cluster centers to be try.
    :param end: end number of cluster centers to be try.
    :param cv: count vector of words.
    :return:
    """
    if end < start:
        raise Exception("end number should bigger than start number,"
                        " find end={} < than start={}".format(end, start))
    n_topics = range(start, end + 1)
    try_times = len(n_topics)
    perplexity_list = [1.0] * try_times
    lda_models = []
    for idx, n_topic in enumerate(n_topics):
        lda = LatentDirichletAllocation(n_components=n_topic,
                                        learning_method='batch',
                                        max_iter=200,
                                        random_state=42,
                                        evaluate_every=200)
        lda.fit(cv)
        perplexity_list[idx] = lda.perplexity(cv)
        lda_models.append(lda)
    if try_times == 1:
        return 1, lda_models[0]
    if try_times == 2:
        min_index = np.argmin(perplexity_list)
        return min_index, perplexity_list[min_index]
    scope_list = np.diff(perplexity_list)
    #growth_rate_list = scope_list / perplexity_list[:-1]

    scope_index = cal_n_cluster(scope_list)
    best_cluster = n_topics[scope_index]
    return best_cluster


def cal_n_cluster(scope_list):
    """
    计算合适的聚类中心数
    :param scope_diff: 斜率的一阶差分，即复杂度的二阶差分
    :return 返回最合适聚类中心个数的index，例如np.arange(2, 10)[rst]
    """
    options = np.argwhere(scope_list>=0).reshape(-1, ).tolist() # 复杂度升高的聚类中心数
    if len(options) == 0:
        # 如果复杂度一直降低，没有波动，那么返回最后一个聚类中心个数，index = -1
        return -1
    for i in options:
        # 对每个差分为正的点
        if i < len(scope_list) - 1:
            # 如果不是差分数组中最后一个点
            if scope_list[i+1] < 0:
                # 只有当该点为尖峰点时(该点以及其后一个点复杂度都上升时则不为尖峰点)
                if scope_list[i] > abs(scope_list[i+1]):
                    return i
            else:
                # 如果这个点后面一个点也是复杂度升高，那么直接返回这个点
                return i
    # 如果遍历后没有合适点，则选择options的最后一个点
    return options[-1]


def execute_lda(text_df, today, stopwords_list, local_dict_words):

    try:
        print("data shape is {}".format(text_df.shape))
        for word in local_dict_words:
            jieba.add_word(word)

        text_df = text_df[~pd.isnull(text_df["content"])]
        seg_texts = segmentation(text_df.content.values, stopwords_list)
        print("Doc word segments done")
        count_vector, cv_words = build_count_vector(seg_texts)
        print("Doc count vectors built")

        #n_topic = 1
        #if 1 < text_df.shape[0] <= 10:
        #    n_topic = 2
        #elif text_df.shape[0] > 10:
        #    remainder = text_df.shape[0] % 5
        #    if remainder > 0:
        #        remainder = 1

        #    n_topic = min(int(text_df.shape[0] / 5) + remainder, 10)

        n_topic = find_cluster_center(count_vector)

        print("Got {} topics".format(n_topic))

        text_df = lda_fit_cluster(count_vector, int(n_topic), text_df, cv_words)

        text_df.drop(columns=['pub_date', 'start_date', 'end_date'], inplace=True)

        print("Topic result schema: {}".format(text_df.columns.values))
        text_df.topic_id = text_df.topic_id.apply(lambda x: str(x))
        text_df.topic_words = text_df.topic_words.apply(lambda x: ','.join(x))

        text_df['word_counts'] = text_df.apply(lambda x: find_top_n(x.content, x.topic_words), axis=1)

        text_df.drop(columns=['content'], inplace=True)

        text_df['datetime'] = today

        return text_df
    except Exception as e:
        print(e)
        return pd.DataFrame([], columns=["crawler_id", "products", "keywords", "sentiment_id", "company", "topic_id",
                                         "topic_words", "word_counts", "datetime"])


@pandas_udf(topic_df_schema, functionType=PandasUDFType.GROUPED_MAP)
def _spark_analyze_real_estate_company_topic(company_df):

    merged_result_df = pd.DataFrame([], columns=["crawler_id", "products", "keywords", "sentiment_id", "company",
                                                 "topic_id", "topic_words", "word_counts", "datetime"])

    try:
        today = datetime.datetime.now().strftime('%Y-%m-%d')

        company_df = company_df[~pd.isnull(company_df.company)]
        companies = company_df.company.unique().tolist()
        companies = [company for company in companies if len(company.strip()) > 0]

        clean_companies = []
        for company in companies:
            if company.find("[") >= 0 and company.find("]") > 0:
                items = json.loads(company)
            else:
                items = re.split("[,，]", company)

            clean_companies.extend([item for item in items if item.strip() != "开发商"])

        for company in clean_companies:
            result_df = execute_lda(company_df[company_df["company"].str.contains(company)],
                                    today, stopwords_list.value, local_dict_words.value)

            result_df.keywords = result_df.keywords.astype("str")

            merged_result_df = pd.concat([merged_result_df, result_df])
            print("{} negative topics is extracted".format(company))

    except Exception as e:
        print("Fail to extract company topic: {}".format(e))

    return merged_result_df


@pandas_udf(top_topic_schema, functionType=PandasUDFType.GROUPED_MAP)
def get_top_topics(company_df):

    merged_result_df = pd.DataFrame([], columns=['topic_id', 'sentiment_id_list', 'datetime', 'products', 'keywords'])

    try:
        today = datetime.datetime.now().strftime('%Y-%m-%d')

        company_df = company_df[~pd.isnull(company_df.company)]
        companies = company_df.company.unique().tolist()
        companies = [company for company in companies if len(company.strip()) > 0]

        clean_companies = []
        for company in companies:
            if company.find("[") >= 0 and company.find("]") > 0:
                items = json.loads(company)
            else:
                items = re.split("[,，]", company)

            clean_companies.extend([item for item in items if item.strip() != "开发商"])

        for company in clean_companies:
            single_company_df = company_df[company_df["company"].str.contains(company)]

            n_topics = len(single_company_df.topic_id.value_counts().index)

            top_n_per_topic = select_top_n_docs(single_company_df, n_topics)
            if top_n_per_topic is None:
                continue

            top_n_per_topic_df = pd.DataFrame([], columns=['topic_id', 'sentiment_id_list', 'datetime'])
            print("Top N per topic result schema: {}".format(top_n_per_topic_df.columns.values))
            index = 0
            for k, v in top_n_per_topic.items():
                top_n_per_topic_df.loc[index] = {'topic_id': str(k), 'sentiment_id_list': ','.join(v), 'datetime': today}
                index += 1

            top_n_per_topic_df["products"] = single_company_df.products.values[0]
            top_n_per_topic_df["keywords"] = single_company_df.keywords.values[0]

            top_n_per_topic_df["products"] = top_n_per_topic_df["products"].astype("str")
            top_n_per_topic_df["keywords"] = top_n_per_topic_df["keywords"].astype("str")

            merged_result_df = pd.concat([merged_result_df, top_n_per_topic_df])
            print("{} top topics is extracted".format(company))
    except Exception as e:
        print("Fail to extract company N top topic: {}".format(e))

    return merged_result_df


def _spark_analyze_all_real_estate_company_topic(company_df, time_range_key, time_range_value, batch_uid):
    global stopwords_list, local_dict_words

    try:
        end_date = datetime.datetime.now()
        start_date = end_date + datetime.timedelta(days=-time_range_value)
        start_date = start_date.strftime("%Y-%m-%d")
        end_date = end_date.strftime("%Y-%m-%d")
        logger.info("Extract real estate company topic data from {} to {}".format(start_date, end_date))

        company_df = company_df[(company_df["pub_date"] >= start_date) & (company_df["pub_date"] <= end_date)]

        company_df = company_df[(~pd.isnull(company_df["content"])) & (company_df["content"].str.strip().str.len() > 0)]

        spark_manager = PySparkMgr(topic_spark_config, "topic-analysis-pyspark")
        spark, sc = spark_manager.start(config.spark_user, False)
        if spark is None:
            raise Exception("Fail to launch spark session")
        logger.info(spark)

        stopwords_list = sc.broadcast(get_stopwords())

        local_dict_words = sc.broadcast(load_local_dict())

        company_df = dedup_with_tfifd.dedup(spark, sc, company_df, get_stopwords())

        company_df.sentiment_id = company_df.sentiment_id.astype("str")

        spark_company_df = spark.createDataFrame(company_df)

        topic_df = spark_company_df.groupBy('company').apply(_spark_analyze_real_estate_company_topic).toPandas()

        spark_topic_df = spark.createDataFrame(topic_df)

        top_topic_df = spark_topic_df.groupBy('company').apply(get_top_topics).toPandas()

        result_path, top_n_per_topic_path = get_real_estate_company_csv_path(time_range_key, batch_uid)

        topic_df = topic_df[["crawler_id", "products", "keywords", "sentiment_id", "company",
                             "topic_id", "topic_words", "datetime"]]

        top_topic_df = extract_company_topic_texts_summaries(company_df, top_topic_df, topic_df, time_range_key)

        logger.info(top_topic_df[["topic_id", "summary"]].head())

        RWFlocker.lock(RWFlocker.WRITE)

        topic_df.to_csv(result_path, sep=',', header=True, index=False)

        top_topic_df.to_csv(top_n_per_topic_path, sep=',', header=True, index=False)

        RWFlocker.unlock()

        logger.info("All negative topics of real estate companies are extracted")

    except Exception as e:
        logger.error("Company topic error: {}".format(e))
    finally:
        RWFlocker.unlock()


def spark_segmentation(texts, stopwords):
    """
    Chinese segmentation for text
    :param text: list of text
    :param stopwords: stop words list
    :return:
    """
    stopwords = stopwords.value

    for doc in texts:
        if pd.isnull(doc[0]):
            continue
        seg_list = jieba.cut(doc[0].strip(), HMM=False)
        seg_words = []
        for item in seg_list:
            if item not in stopwords and re.match(r'-?\d+\.?\d*', item) is None and len(item.strip()) > 1:
                seg_words.append(item)

        yield (Row(id=doc[1], raw_corpus=seg_words))


def get_doc_topic(iterator):

    for doc in iterator:

        topic_probes = []

        for p in doc["topicDistribution"]:
            topic_probes.append(p)

        topic_probes = np.array(topic_probes)

        yield (doc["id"], np.argmax(topic_probes))


def get_top_n_doc(topic_id, iterator):

    docs = [x for x in iterator]

    doc_counts = np.array([doc[1] for doc in docs])
    top_indices = heapq.nlargest(3, range(len(doc_counts)), doc_counts.take)

    return int(topic_id), ",".join([docs[i][0] for i in top_indices]), datetime.datetime.now().strftime('%Y-%m-%d')


def get_doc_topic_word_count(iterator, topic_words):

    topic_words_dict = {item[0]: item[1] for item in topic_words.value}

    for doc in iterator:
        yield (doc[2], (doc[0], sum([doc[1].count(word) for word in topic_words_dict[doc[2]]])))


def align_doc_topic_words(iterator, topic_words):

    topic_words_dict = dict(topic_words.value)

    today = datetime.datetime.now().strftime('%Y-%m-%d')

    for doc in iterator:
        yield (doc[1][0], int(doc[0]), ",".join(topic_words_dict[doc[0]]), today)


def _spark_analyze_real_estate_business_topic(company_df, time_range_key, time_range_value, batch_uid):
    global stopwords_list

    try:
        end_date = datetime.datetime.now()
        start_date = end_date + datetime.timedelta(days=-time_range_value)
        start_date = start_date.strftime("%Y-%m-%d")
        end_date = end_date.strftime("%Y-%m-%d")
        logger.info("Extract real estate company topic data from {} to {}".format(start_date, end_date))

        company_df = company_df[(company_df["pub_date"] >= start_date) & (company_df["pub_date"] <= end_date)]
        company_df = company_df[company_df.products.str.contains(config.CompanyType.REAL_ESTATE.value)]
        company_df = company_df[(~pd.isnull(company_df["content"])) & (company_df["content"].str.strip().str.len() > 0)]

        company_df.keywords = company_df.keywords.astype("str")
        company_df.sentiment_id = company_df.sentiment_id.astype("str")

        logger.info(company_df.shape)

        spark_manager = PySparkMgr(topic_spark_config, "topic-analysis-pyspark")
        spark, sc = spark_manager.start(config.spark_user, False)
        if spark is None:
            raise Exception("Fail to launch spark session")
        logger.info(spark)

        company_df = dedup_with_tfifd.dedup(spark, sc, company_df, get_stopwords())

        spark_company_df = spark.createDataFrame(company_df)

        if stopwords_list is None:
            stopwords_list = sc.broadcast(get_stopwords())

        seg_corps = sc.parallelize(company_df.content.values).zipWithIndex(). \
            mapPartitions(lambda x: spark_segmentation(x, stopwords_list))

        seg_corps_df = spark.createDataFrame(seg_corps)

        count_vec = CountVectorizer(inputCol="raw_corpus", outputCol="tf_features", vocabSize=500)

        count_vec_model = count_vec.fit(seg_corps_df)
        seg_corps_df = count_vec_model.transform(seg_corps_df)

        idf = IDF(inputCol="tf_features", outputCol="features")
        seg_corps_df = idf.fit(seg_corps_df).transform(seg_corps_df)
        seg_corps_df = seg_corps_df.drop('tf_features')

        n_topic = 1
        if 1 < company_df.shape[0] <= 10:
            n_topic = 2
        elif company_df.shape[0] > 10:
            remainder = company_df.shape[0] % 5
            if remainder > 0:
                remainder = 1

            n_topic = min(int(company_df.shape[0] / 5) + remainder, 10)

        lda = LDA(k=int(n_topic), featuresCol="features", seed=0)
        model = lda.fit(seg_corps_df)

        transformed = model.transform(seg_corps_df)

        vocab = count_vec_model.vocabulary

        topic_words = model.describeTopics().rdd \
            .map(lambda row: (row['topic'], [vocab[idx] for idx in row['termIndices']])).collect()

        logger.info("topic words = {}".format(topic_words))

        topic_words = sc.broadcast(topic_words)

        doc_topic_id = transformed.rdd.mapPartitions(lambda rows: get_doc_topic(rows))
        # result RDD: (crawler_id, content, topic_id)
        doc_topic_id = spark_company_df.rdd.map(lambda row: (row["crawler_id"], row['content'])).zipWithIndex(). \
            map(lambda x: (x[1], x[0])).join(doc_topic_id).map(lambda x: x[1]).map(lambda x: (x[0][0], x[0][1], x[1]))

        # result RDD: (topic_id, (crawler_id, word_count))
        cached_doc_topic = doc_topic_id.mapPartitions(lambda x: get_doc_topic_word_count(x, topic_words)).cache()
        doc_topic = cached_doc_topic.mapPartitions(lambda x: align_doc_topic_words(x, topic_words))

        logger.info(doc_topic.take(2))

        structFields = [StructField("crawler_id", StringType(), True),
                        StructField("topic_id", IntegerType(), True),
                        StructField("topic_words", StringType(), True),
                        StructField("datetime", StringType(), True)]

        doc_topic_df = spark.createDataFrame(doc_topic, StructType(structFields))
        doc_topic_df = spark_company_df.select("crawler_id", "sentiment_id", "company"). \
            join(doc_topic_df, ["crawler_id"]).toPandas()
        doc_topic_df.index = np.arange(doc_topic_df.shape[0])

        doc_top_topics = cached_doc_topic.groupByKey().map(lambda x: get_top_n_doc(x[0], x[1])).collect()
        doc_top_topic_df = spark.createDataFrame(doc_top_topics, ["topic_id", "sentiment_id_list", "datetime"]).toPandas()
        doc_top_topic_df.index = np.arange(doc_top_topic_df.shape[0])

        doc_top_topic_df = extract_business_topic_texts_summaries(company_df, doc_top_topic_df, doc_topic_df, time_range_key)
        logger.info(doc_top_topic_df[["topic_id", "summary"]].head())

        RWFlocker.lock(RWFlocker.WRITE)

        result_path, top_n_per_topic_path = get_real_estate_business_csv_path(time_range_key, batch_uid)

        doc_topic_df.to_csv(result_path, sep=',', header=True, index=False)
        doc_top_topic_df.to_csv(top_n_per_topic_path, sep=',', header=True, index=False)

    except Exception as e:
        logger.error("Business topic error: {}".format(e))
    finally:
        RWFlocker.unlock()


def spark_analyze_real_estate_company_topic(time_range, topic_type, batch_uid=None):
    try:
        if batch_uid is None:
            data_path = config.preprocessed_real_estate_data_path
        else:
            p = config.preprocessed_real_estate_data_path.rfind(".")
            data_path = config.preprocessed_real_estate_data_path[:p] + "_" + batch_uid + ".csv"

        if not os.path.exists(data_path):
            raise Exception("Extracting real estate company topic error: no preproccessed company data")

        company_df = pd.read_csv(data_path, sep=',')
        company_df.index = np.arange(company_df.shape[0])
        logger.info("Topic type is {}".format(topic_type))

        if topic_type == TopicType.COMPANY.value:
            _spark_analyze_all_real_estate_company_topic(company_df, time_range, config.get_result_time_range[time_range],
                                                         batch_uid)
        elif topic_type == TopicType.BUSINESS.value:
            _spark_analyze_real_estate_business_topic(company_df, time_range, config.get_result_time_range[time_range],
                                                      batch_uid)

    except Exception as e:
        logger.error(e)


def _execute():
    logger.info("start topic analysis daemon")

    data = {}
    batch_uid = None

    while True:
        try:
            message = _redis.pop_data(config.spark_topic_message_key, True)
            logger.info(message)

            if 'type' not in message:
                raise Exception("Data miss type field")

            if message['type'] not in [config.MessageType.REAL_ESTATE_COMPANY_TOPIC.value,
                                       config.MessageType.HOT_COMPANY_TOPIC]:
                raise Exception("Unrecognized type field")

            if "fetch_data" in message:
                end_date = datetime.datetime.now()
                start_date = end_date + \
                             datetime.timedelta(
                                 days=-config.get_result_time_range[config.GetResultTimeRange.ONE_MONTH.value])

                start_date = start_date.strftime("%Y-%m-%d")
                end_date = end_date.strftime("%Y-%m-%d")

                p = Process(target=concurrent_fetch_data_v3, args=(start_date, end_date,))
                p.start()
                p.join()
                logger.info("All crawler data fetched, prepared for analysis")

                fetch_real_estate_company_data(start_date, end_date)
                p = Process(target=fetch_sentiment_data, args=())
                p.start()
                p.join()

                p = Process(target=fetch_ke_research_data, args=(start_date, end_date,))
                p.start()
                p.join()
                logger.info("Real estate company data fetched, prepared for analysis")

                fetch_hot_company_data(start_date, end_date)
                logger.info("Hot company data fetched, prepared for analysis")

                fetch_oversea_topic_data(start_date, end_date)
                logger.info("oversea topic data fetched, prepared for analysis")

            if "batch_uid" in message:
                batch_uid = message["batch_uid"]

            batch_companies = []
            if "batch_companies" in message:
                batch_companies = message["batch_companies"]
                logger.info("batch_companies: {}".format(batch_companies))

            batch_time_range = config.GetResultTimeRange.ONE_MONTH.value
            if "batch_time_range" in message:
                batch_time_range = message["batch_time_range"]

            if batch_uid is not None and batch_time_range in [config.GetResultTimeRange.ONE_DAY.value,
                                                              config.GetResultTimeRange.THREE_DAYS.value]:
                preprocess_real_estate_company_data(batch_companies, batch_uid, config.GetResultTimeRange.ONE_WEEK.value)
            else:
                preprocess_real_estate_company_data(batch_companies, batch_uid, batch_time_range)

                preprocessed_df = None

                if os.path.exists(config.hot_company_keyword_data_path):
                    hot_company_df = pd.read_csv(config.hot_company_keyword_data_path, sep=',')
                    preprocessed_df = pd.read_csv(config.preprocessed_real_estate_data_path, sep=',')
                    preprocessed_df = pd.concat([preprocessed_df, hot_company_df], ignore_index=True)

                if os.path.exists(config.oversea_topic_data_path):
                    oversea_data_df = pd.read_csv(config.oversea_topic_data_path, sep=',')

                    if preprocessed_df is None:
                        preprocessed_df = pd.read_csv(config.preprocessed_real_estate_data_path, sep=',')
                    preprocessed_df = pd.concat([preprocessed_df, oversea_data_df], ignore_index=True)

                if preprocessed_df is not None:
                    preprocessed_df.to_csv(config.preprocessed_real_estate_data_path, sep=',', header=True, index=False)

            topic_processes = []

            if batch_uid is None:
                for time_range in config.get_result_time_range.keys():
                    for topic_type in TopicType:
                        p = Process(target=spark_analyze_real_estate_company_topic,
                                    args=(time_range, topic_type.value, ))
                        p.start()
                        topic_processes.append(p)

            else:
                topic_type = TopicType.COMPANY.value
                if "batch_method" in message and message["batch_method"] == "business":
                    topic_type = TopicType.BUSINESS.value

                p = Process(target=spark_analyze_real_estate_company_topic,
                            args=(batch_time_range, topic_type, batch_uid, ))
                p.start()
                topic_processes.append(p)

            for p in topic_processes:
                p.join()

            data["status"] = "success"
            if "batch_method" in message:
                data["batch_method"] = message["batch_method"]
            data["datetime"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

            if batch_uid is not None:
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
