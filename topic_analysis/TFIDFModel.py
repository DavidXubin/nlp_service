import re
import jieba
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import config
from topic_analysis import logger as app_log
from utils import load_local_dict, get_stopwords, segmentation


def build_tfidf_vector(seg_corpus):
    """
    build word-frequency matrix
    :param seg_corpus: segmented corpus
    :return: count vector, words list
    """
    vectorizer = CountVectorizer(
                                 max_features=500
                                 )
    cv = vectorizer.fit_transform(seg_corpus)
    cv_words = np.array(vectorizer.get_feature_names())
    transformer = TfidfTransformer()
    tf_idf = transformer.fit_transform(cv)
    weight = tf_idf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    return weight, cv_words


def get_keywords(text_df, stopwords):
    """
    build keywords list for each company
    :param text_df:
    :param stopwords:
    :return: dict, {k:v} = {company: keywords str}
    """
    rst_dict = dict()
    rep_info = re.compile("“|”|\"|\'")
    companies = set([rep_info.sub('', x.strip()) for x in set(text_df.keywords.values)])

    for company in companies:
        company_data = text_df[text_df.keywords == company]
        seg_corpus = segmentation(company_data.content.values, stopwords)
        weight, cv_words = build_tfidf_vector(seg_corpus)
        total_weight = (weight / weight.sum(axis=1)[:, np.newaxis]).sum(axis=0)
        keywords = cv_words[np.argsort(total_weight)[-10:]]
        rst_dict[company] = ','.join(keywords)
    return rst_dict


def process(text_df, today):
    try:
        result_df = text_df.copy()

        local_dict_words = load_local_dict()
        for word in local_dict_words:
            jieba.add_word(word)

        stopwords_list = get_stopwords()
        company_dict = get_keywords(result_df, stopwords_list)

        result_df.drop(columns=['content', 'products', 'pub_date', 'start_date', 'end_date'], inplace=True)

        result_df.rename(columns={'keywords': 'topic_id'}, inplace=True)
        rep_info = re.compile("“|”|\"|\'")
        result_df.topic_id = result_df.topic_id.apply(lambda x: rep_info.sub('', x.strip()))

        result_df['datetime'] = today
        result_df["topic_words"] = result_df.topic_id.apply(lambda x: company_dict[x])

        app_log.info("Topic result schema: {}".format(result_df.columns.values))
        return result_df
    except Exception as e:
        app_log.error(e)
        return None
