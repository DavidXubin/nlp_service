import jieba
import jieba.analyse
import numpy as np
import pandas as pd
from topic_analysis import logger as app_log
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import config
from utils import load_local_dict, get_stopwords, segmentation

_MIN_TOPIC_NUM = 1
_MAX_TOPIC_NUM = 10


def build_count_vector(seg_corpus):
    """
    build word-frequency matrix
    :param seg_corpus: segmented corpus
    :return: count vector, words list
    """

    try:
        vectorizer = CountVectorizer(min_df=10,
                                     max_df=0.90,
                                     max_features=500
                                     )
        cv = vectorizer.fit_transform(seg_corpus)
        cv_words = np.array(vectorizer.get_feature_names())
    except Exception as e:
        app_log.error(e)
        vectorizer = CountVectorizer(max_features=500)

        cv = vectorizer.fit_transform(seg_corpus)
        cv_words = np.array(vectorizer.get_feature_names())

    return cv, cv_words


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


def find_cluster_center(cv, start=_MIN_TOPIC_NUM, end=_MAX_TOPIC_NUM):
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


def select_top_n_docs(text_df, n_cluster, get_word_count=True):
    """
    select typical documents of each list of keywords
    :param text_df: pandas dataframe
    :param n_cluster: number of cluster centers
    :param get_word_count: whether to count word frequency
    :return: type name and uid
    """
    if get_word_count:
        text_df['word_counts'] = text_df.apply(lambda x: find_top_n(x.content, x.topic_words), axis=1)

    sorted_data = text_df.sort_values('word_counts', ascending=False)
    rst_dict = dict()
    for i in range(n_cluster):
        topic_data = sorted_data[sorted_data.topic_id == i]

        if topic_data.shape[0] >= config.top_n_texts_per_topic:
            rst_dict[i] = topic_data.head(config.top_n_texts_per_topic).crawler_id.values
        else:
            rst_dict[i] = topic_data.crawler_id.values

    return rst_dict


def process(text_df, today, keep_keywords_products=False, add_keyword_to_jieba=True):
    try:
        app_log.info("data shape is {}".format(text_df.shape))
        local_dict_words = load_local_dict()

        for word in local_dict_words:
            jieba.add_word(word)

        if add_keyword_to_jieba:
            for key in set(text_df.keywords.values):
                if key is not None:
                    jieba.add_word(key)

        stopwords_list = get_stopwords()
        text_df = text_df[~pd.isnull(text_df["content"])]
        seg_texts = segmentation(text_df.content.values, stopwords_list)
        app_log.info("Doc word segments done")
        count_vector, cv_words = build_count_vector(seg_texts)
        app_log.info("Doc count vectors built")
        n_topic = find_cluster_center(count_vector, start=_MIN_TOPIC_NUM, end=_MAX_TOPIC_NUM)
        app_log.info("Got {} topics".format(n_topic))

        text_df = lda_fit_cluster(count_vector, n_topic, text_df, cv_words)
        top_n_per_topic = select_top_n_docs(text_df, n_topic)
        app_log.info(top_n_per_topic)

        if keep_keywords_products:
            text_df.drop(columns=['content', 'pub_date', 'word_counts', 'start_date', 'end_date'], inplace=True)
        else:
            text_df.drop(columns=['content', 'keywords', 'products', 'pub_date', 'word_counts', 'start_date', 'end_date'],
                         inplace=True)
        text_df['datetime'] = today
        app_log.info("Topic result schema: {}".format(text_df.columns.values))
        text_df.topic_id = text_df.topic_id.apply(lambda x: str(x))
        text_df.topic_words = text_df.topic_words.apply(lambda x: ','.join(x))

        top_n_per_topic_df = pd.DataFrame([], columns=['topic_id', 'sentiment_id_list', 'datetime'])
        app_log.info("Top N per topic result schema: {}".format(top_n_per_topic_df.columns.values))
        index = 0
        for k, v in top_n_per_topic.items():
            top_n_per_topic_df.loc[index] = {'topic_id': str(k), 'sentiment_id_list': ','.join(v), 'datetime': today}
            index += 1

        text_df.fillna(value="0", inplace=True)

        return text_df, top_n_per_topic_df
    except Exception as e:
        app_log.error(e)
        return None, None

