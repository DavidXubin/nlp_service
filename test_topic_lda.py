import os
import re
import json
import jieba
import datetime
import heapq
import random
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
import config
from config import spark_user, preprocessed_real_estate_data_path, get_result_time_range
from topic_analysis.run import get_real_estate_company_csv_path, get_real_estate_business_csv_path
from topic_analysis_service import fetch_company_data, preprocess_real_estate_company_data
from utils import get_logger

logger = None
if logger is None:
    logger = get_logger("spark-topic-analysis-service-test")


class TopicType(Enum):
    COMPANY = "company"
    BUSINESS = "business"


topic_spark_config = {
    "spark.driver.memory": "10g",
    "spark.executor.memory": "5g",
    "spark.driver.maxResultSize": "5g",
    "spark.executor.cores": 3,
    "spark.executor.instances": 4
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

cwd_path = os.getcwd()

def get_real_estate_company_test_csv_path():

    result_path = os.path.join(cwd_path, "topic_result.csv")
    top_n_per_topic_path = os.path.join(cwd_path, "topic_result_top_n.csv")

    return result_path, top_n_per_topic_path


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

        n_topic = 1
        if 1 < text_df.shape[0] <= 10:
            n_topic = 2
        elif text_df.shape[0] > 10:
            remainder = text_df.shape[0] % 5
            if remainder > 0:
                remainder = 1

            n_topic = min(int(text_df.shape[0] / 5) + remainder, 10)

        print("Got {} topics".format(n_topic))

        text_df = lda_fit_cluster(count_vector, n_topic, text_df, cv_words)

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

    today = datetime.datetime.now().strftime('%Y-%m-%d')

    companies = company_df.company.value_counts().index.tolist()

    clean_companies = []
    for company in companies:
        if company.find("[") >= 0 and company.find("]") > 0:
            items = json.loads(company)
        else:
            items = re.split("[,，]", company)

        clean_companies.extend([item for item in items if item.strip() != "开发商"])

    merged_result_df = pd.DataFrame([], columns=["crawler_id", "products", "keywords", "sentiment_id", "company",
                                                 "topic_id", "topic_words", "word_counts", "datetime"])

    for company in clean_companies:
        result_df = execute_lda(company_df[company_df["company"].str.contains(company)],
                                today, stopwords_list.value, local_dict_words.value)
        if result_df is None:
            continue

        merged_result_df = pd.concat([merged_result_df, result_df])
        print("{} negative topics is extracted".format(company))

    return merged_result_df


@pandas_udf(top_topic_schema, functionType=PandasUDFType.GROUPED_MAP)
def get_top_topics(company_df):

    today = datetime.datetime.now().strftime('%Y-%m-%d')

    companies = company_df.company.value_counts().index.tolist()

    clean_companies = []
    for company in companies:
        if company.find("[") >= 0 and company.find("]") > 0:
            items = json.loads(company)
        else:
            items = re.split("[,，]", company)

        clean_companies.extend([item for item in items if item.strip() != "开发商"])

    merged_result_df = pd.DataFrame([], columns=['topic_id', 'sentiment_id_list', 'datetime', 'products', 'keywords'])

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

        merged_result_df = pd.concat([merged_result_df, top_n_per_topic_df])
        print("{} top topics is extracted".format(company))

    return merged_result_df


def _spark_analyze_all_real_estate_company_topic(company_df, time_range_key=None, time_range_value=None):
    global stopwords_list, local_dict_words

    try:
        #end_date = datetime.datetime.now()
        #start_date = end_date + datetime.timedelta(days=-time_range_value)
        #start_date = start_date.strftime("%Y-%m-%d")
        #end_date = end_date.strftime("%Y-%m-%d")
        #logger.info("Extract real estate company topic data from {} to {}".format(start_date, end_date))

        #company_df = company_df[(company_df["pub_date"] >= start_date) & (company_df["pub_date"] <= end_date)]

        company_df = company_df[~pd.isnull(company_df["content"])]

        spark_manager = PySparkMgr(topic_spark_config, "topic-analysis-pyspark-test")
        spark, sc = spark_manager.start(spark_user, True)
        if spark is None:
            raise Exception("Fail to launch spark session")
        logger.info(spark)

        if stopwords_list is None:
            stopwords_list = sc.broadcast(get_stopwords())

        if local_dict_words is None:
            local_dict_words = sc.broadcast(load_local_dict())

        spark_company_df = spark.createDataFrame(company_df)

        topic_df = spark_company_df.groupBy('company').apply(_spark_analyze_real_estate_company_topic).toPandas()

        spark_topic_df = spark.createDataFrame(topic_df)

        top_topic_df = spark_topic_df.groupBy('company').apply(get_top_topics).toPandas()

        RWFlocker.lock(RWFlocker.WRITE)

        result_path, top_n_per_topic_path = get_real_estate_company_test_csv_path()

        topic_df[["crawler_id", "products", "keywords", "sentiment_id", "company",
                  "topic_id", "topic_words", "datetime"]].to_csv(result_path, sep=',', header=True, index=False)

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


def _spark_analyze_real_estate_business_topic(company_df, time_range_key=None, time_range_value=None, batch_uid=None):
    global stopwords_list

    try:
        #end_date = datetime.datetime.now()
        #start_date = end_date + datetime.timedelta(days=-time_range_value)
        #start_date = start_date.strftime("%Y-%m-%d")
        #end_date = end_date.strftime("%Y-%m-%d")
        #logger.info("Extract real estate company topic data from {} to {}".format(start_date, end_date))

        #company_df = company_df[(company_df["pub_date"] >= start_date) & (company_df["pub_date"] <= end_date)]

        company_df = company_df[~pd.isnull(company_df["content"])]

        logger.info(company_df.shape)

        spark_manager = PySparkMgr(topic_spark_config, "topic-analysis-pyspark-test")
        spark, sc = spark_manager.start(spark_user, True)
        if spark is None:
            raise Exception("Fail to launch spark session")
        logger.info(spark)

        spark_company_df = spark.createDataFrame(company_df)

        if stopwords_list is None:
            stopwords_list = sc.broadcast(get_stopwords())

        seg_corps = sc.parallelize(company_df.content.values).zipWithIndex(). \
            mapPartitions(lambda x: spark_segmentation(x, stopwords_list))

        seg_corps_df = spark.createDataFrame(seg_corps)

        count_vec = CountVectorizer(inputCol="raw_corpus", outputCol="tf_features", vocabSize=500, minDF=10)

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

        lda = LDA(k=n_topic, featuresCol="features", seed=0)
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
            map(lambda x: (x[1], x[0])). join(doc_topic_id).map(lambda x: x[1]).map(lambda x: (x[0][0], x[0][1], x[1]))

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

        doc_top_topics = cached_doc_topic.groupByKey().map(lambda x: get_top_n_doc(x[0], x[1])).collect()
        doc_top_topic_df = spark.createDataFrame(doc_top_topics, ["topic_id", "sentiment_id_list", "datetime"]).toPandas()

        RWFlocker.lock(RWFlocker.WRITE)

        #result_path, top_n_per_topic_path = get_real_estate_business_csv_path(time_range_key, batch_uid)
        result_path, top_n_per_topic_path = get_real_estate_company_test_csv_path()

        doc_topic_df.to_csv(result_path, sep=',', header=True, index=False)
        doc_top_topic_df.to_csv(top_n_per_topic_path, sep=',', header=True, index=False)

        RWFlocker.unlock()

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
        logger.info("Topic type is {}".format(topic_type))

        if topic_type == TopicType.COMPANY.value:
            _spark_analyze_all_real_estate_company_topic(company_df, time_range, config.get_result_time_range[time_range])
        elif topic_type == TopicType.BUSINESS.value:
            _spark_analyze_real_estate_business_topic(company_df, time_range, config.get_result_time_range[time_range],
                                                      batch_uid)

    except Exception as e:
        logger.error(e)


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


company_df = pd.read_csv(config.preprocessed_real_estate_data_path, sep=',')

total_num = np.arange(company_df.shape[0])

random.shuffle(total_num)

company_df = company_df.loc[total_num, :]

test_batch_list = [50, 100, 200, 300, 400]

result_df = pd.DataFrame([], columns=["topic_id", "topic_words", "text_id_list", "test_batch_size"])

idx = 0

for batch in test_batch_list:
    _spark_analyze_real_estate_business_topic(company_df[:batch])

    topic_path, _ = get_real_estate_company_test_csv_path()

    topic_df = pd.read_csv(topic_path, sep=",")

    for topic_id, df in topic_df.groupby("topic_id"):

        result_df.loc[idx] = {"topic_id": str(topic_id),
                              "topic_words": ",".join(df.topic_words.values),
                              "text_id_list": ",".join(df.crawler_id.values),
                              "test_batch_size": batch}
        idx += 1


spark_manager = PySparkMgr(topic_spark_config, "topic-analysis-pyspark-test")
spark, sc = spark_manager.start(spark_user)

insert_dataframe_into_db(result_df, spark, "infra_ml.lda_topic_test")

test_topic_df = spark.sql("select * from infra_ml.lda_topic_test").toPandas()

all_text_ids = []

for data in test_topic_df.text_id_list.values:
    all_text_ids.extend(data.split(","))



