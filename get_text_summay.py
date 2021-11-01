import config
import datetime
import numpy as np
import pandas as pd
from utils import generate_uid
from redis_db import RedisDBWrapper


_redis = RedisDBWrapper()


def extract_summary(content, uid, algorithm, topic_words):
    message = {
        "type": config.MessageType.PREDICTED.value,
        "uid":  uid,
        "content": content,
        "algorithm": algorithm,
        "datetime": datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    }

    _redis.push_data(message, config.text_summary_req_table_key)

    result = _redis.pop_data(uid, True)

    if result["status"] != "ok":
        return topic_words
    else:
        return result["summary"]


def extract_business_topic_texts_summaries(company_df, doc_top_topic_df, doc_topic_df, time_range_key=""):
    topic_texts_df = pd.DataFrame([], columns=["topic_id", "content", "uid", "algorithm"])

    for idx in doc_top_topic_df.index:

        topic_id = doc_top_topic_df.loc[idx, "topic_id"]

        sentiment_id_list = doc_top_topic_df.loc[idx, "sentiment_id_list"].split(",")

        texts = ""

        for sentiment_id in sentiment_id_list:

            text = company_df[company_df.crawler_id == sentiment_id].content.values[0]

            texts = texts + "\n" + text

        algorithm = "mixed"
        if len(sentiment_id_list) == 1:
            algorithm = "seq2seq"

        topic_texts_df.loc[idx] = {"topic_id": topic_id,  "content": texts,
                                   "uid": generate_uid(time_range_key + texts + time_range_key),
                                   "algorithm": algorithm}

    topic_words_df = doc_topic_df.groupby("topic_id")["topic_words"].value_counts().to_frame(). \
        drop(["topic_words"], axis=1).reset_index()

    topic_texts_df = pd.merge(topic_texts_df, topic_words_df, on="topic_id")

    topic_texts_df.drop_duplicates(["uid"], inplace=True)

    topic_texts_df["summary"] = topic_texts_df.apply(lambda x: extract_summary(x["content"], x["uid"],
                                                                               x["algorithm"], x["topic_words"]), axis=1)

    doc_top_topic_df = pd.merge(doc_top_topic_df, topic_texts_df[["topic_id", "summary"]], on="topic_id")

    doc_top_topic_df.index = np.arange(doc_top_topic_df.shape[0])

    return doc_top_topic_df


def extract_company_topic_texts_summaries(company_df, doc_top_topic_df, doc_topic_df, time_range_key):

    topic_texts_df = pd.DataFrame([], columns=["topic_id", "content", "uid", "algorithm", "products"])

    index = 0

    for company, df in doc_top_topic_df.groupby("products"):

        for idx in df.index:

            topic_id = df.loc[idx, "topic_id"]

            sentiment_id_list = df.loc[idx, "sentiment_id_list"].split(",")

            texts = ""

            for sentiment_id in sentiment_id_list:

                text = company_df[company_df.crawler_id == sentiment_id].content.values[0]

                texts = texts + "\n" + text

            algorithm = "mixed"
            if len(sentiment_id_list) == 1:
                algorithm = "seq2seq"

            topic_texts_df.loc[index] = {"topic_id": topic_id, "content": texts,
                                         "uid": generate_uid(time_range_key + texts + time_range_key),
                                         "algorithm": algorithm, "products": company}

            index += 1

    topic_texts_df.sort_values(by=["products", "topic_id"], inplace=True)

    topic_words_df = doc_topic_df.groupby(["products", "topic_id"])["topic_words"].value_counts().to_frame(). \
        drop(["topic_words"], axis=1).reset_index()

    topic_words_df.sort_values(by=["products", "topic_id"], inplace=True)

    topic_words_df.index = np.arange(topic_words_df.shape[0])

    topic_texts_df = pd.concat([topic_texts_df, topic_words_df[["topic_words"]]], axis=1)

    topic_texts_df["summary"] = topic_texts_df.apply(lambda x: extract_summary(x["content"], x["uid"],
                                                                               x["algorithm"], x["topic_words"]), axis=1)

    doc_top_topic_df.sort_values(by=["products", "topic_id"], inplace=True)

    doc_top_topic_df.index = np.arange(doc_top_topic_df.shape[0])

    return pd.concat([doc_top_topic_df, topic_texts_df[["summary"]]], axis=1)
