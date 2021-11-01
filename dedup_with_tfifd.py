import re
import jieba
import numpy as np
import pandas as pd
from pyspark.sql import Row
from scipy.spatial.distance import cosine
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import HashingTF, IDF, Tokenizer


TFIDF_SIMILARITY_THRESHOLD = 0.94


def get_company_comp_matrix(spark, sc, news_df, company_name):
    if company_name is not None:
        company_df = news_df[news_df.company.str.contains(company_name)]
    else:
        company_df = news_df.copy()

    company_df.index = np.arange(company_df.shape[0])

    company_df = company_df[["crawler_id", "content"]]

    spark_company_df = spark.createDataFrame(company_df)

    doc_matrix = {}
    max_id = company_df.shape[0]

    for idx in company_df.index:

        if idx == max_id - 1:
            break

        crawler_id = company_df.loc[idx, "crawler_id"]

        doc_matrix[crawler_id] = company_df.iloc[idx + 1: max_id]["crawler_id"].values

    return spark_company_df, sc.broadcast(doc_matrix)


def spark_segmentation_for_dedup(texts, stopwords):
    """
    Chinese segmentation for text
    :param text: list of text
    :param stopwords: stop words list
    :return:
    """
    stopwords = stopwords.value

    for doc in texts:
        if pd.isnull(doc["crawler_id"]):
            continue

        seg_list = jieba.cut(doc["content"].strip(), HMM=False)
        seg_words = []
        for item in seg_list:
            if item not in stopwords and re.match(r'-?\d+\.?\d*', item) is None and len(item.strip()) > 1:
                seg_words.append(item)

        yield (Row(id=doc["crawler_id"], raw_corpus=seg_words))


def check_dup_doc_tfidf(iterator, doc_matrix, tfidf_dict):
    doc_matrix = doc_matrix.value

    tfidf_dict = tfidf_dict.value

    for doc in iterator:

        if doc["crawler_id"] not in doc_matrix:
            yield (doc["crawler_id"], False)
            continue

        doc_list = doc_matrix[doc["crawler_id"]]

        doc_vec = tfidf_dict[doc["crawler_id"]]

        found_dup = False
        for crawler_id in doc_list:
            s = 1 - cosine(doc_vec, tfidf_dict[crawler_id])

            if not pd.isnull(s) and s > TFIDF_SIMILARITY_THRESHOLD:
                found_dup = True
                break

        yield (doc["crawler_id"], found_dup)


def toDenseArray(data):
    return data.toArray().tolist()


def dedup_get_duplication_maxtrix(spark, sc, news_df, stop_words, company_name=None):

    news_df.drop_duplicates(["crawler_id"], inplace=True)

    news_df["text_len"] = news_df.content.apply(lambda x: len(x))

    news_df.sort_values(by=["text_len"], inplace=True)

    news_df.index = np.arange(news_df.shape[0])

    _ = news_df.pop("text_len")

    sc_stopwords = sc.broadcast(stop_words)

    spark_news_df = spark.createDataFrame(news_df[["crawler_id", "content"]])

    seg_corps = spark_news_df.rdd.mapPartitions(lambda x: spark_segmentation_for_dedup(x, sc_stopwords))

    seg_corps_df = spark.createDataFrame(seg_corps)

    count_vec = CountVectorizer(inputCol="raw_corpus", outputCol="tf_features", vocabSize=512, minDF=1)

    seg_corps_df = count_vec.fit(seg_corps_df).transform(seg_corps_df)

    idf = IDF(inputCol="tf_features", outputCol="features")

    seg_corps_df = idf.fit(seg_corps_df).transform(seg_corps_df)

    denseArrayUDF = F.udf(toDenseArray, ArrayType(FloatType()))

    seg_corps_df = seg_corps_df.withColumn('tf_idfs', denseArrayUDF("features"))

    tfidf_df = seg_corps_df.select("id", "tf_idfs").toPandas()

    tfidf_df_dict = {}

    for idx in tfidf_df.index:
        tfidf_df_dict[tfidf_df.loc[idx, "id"]] = tfidf_df.loc[idx, "tf_idfs"]

    sc_tfidf_dict = sc.broadcast(tfidf_df_dict)

    spark_company_df, sc_doc_matrix = get_company_comp_matrix(spark, sc, news_df, company_name)

    dup_maxtrix = spark_company_df.rdd.mapPartitions(
        lambda x: check_dup_doc_tfidf(x, sc_doc_matrix, sc_tfidf_dict)).collect()

    return dup_maxtrix


def dedup(spark, sc, news_df, stop_words, company_name=None):

    dup_maxtrix = dedup_get_duplication_maxtrix(spark, sc, news_df, stop_words, company_name)

    dedup_id = [x[0] for x in dup_maxtrix if not x[1]]

    dedup_news_df = news_df[news_df.crawler_id.isin(dedup_id)]

    dedup_news_df.index = np.arange(dedup_news_df.shape[0])

    return dedup_news_df
