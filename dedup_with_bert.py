import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from bert4keras.models import build_transformer_model
# from bert4keras.tokenizers import Tokenizer, load_vocab
import bert4keras.tokenizers as bert_tokenizers
from config import KerasBertWWMModel

BERT_SIMILARITY_THRESHOLD = 0.95


def build_tokenizer(dict_path):
    token_dict, keep_tokens = bert_tokenizers.load_vocab(
        dict_path=dict_path,
        simplified=True,
        startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]']
    )

    tokenizer = bert_tokenizers.Tokenizer(token_dict, do_lower_case=True)

    return tokenizer, keep_tokens


# 加载 bert 模型
def build_model(config_path, checkpoint_path, keep_tokens):
    bert_model = build_transformer_model(config_path, checkpoint_path, keep_tokens=keep_tokens)
    return bert_model


# 生成mask矩阵
def generate_mask(sen_list, max_len):
    len_list = [len(i) if len(i) <= max_len else max_len for i in sen_list]
    array_mask = np.array([np.hstack((np.ones(j), np.zeros(max_len - j))) for j in len_list])
    return np.expand_dims(array_mask, axis=2)


def extract_emb_feature(model, tokenizer, sentences, max_len, mask_if=False):
    mask = generate_mask(sentences, max_len)
    token_ids_list = []
    segment_ids_list = []
    for sen in sentences:
        token_ids, segment_ids = tokenizer.encode(sen, first_length=max_len)
        token_ids_list.append(token_ids)
        segment_ids_list.append(segment_ids)

    result = model.predict([np.array(token_ids_list), np.array(segment_ids_list)])
    if mask_if:
        result = result * mask
    return np.mean(result, axis=1)


def dedup(news_df):
    news_df.drop_duplicates(["crawler_id"], inplace=True)

    news_df["text_len"] = news_df.content.apply(lambda x: len(x))

    news_df.sort_values(by=["text_len"], inplace=True)

    news_df.index = np.arange(news_df.shape[0])

    news_df["order_id"] = news_df.index.values

    _ = news_df.pop("text_len")

    texts = news_df.content.values.tolist()

    bert_dedup_tokenizer, keep_tokens = build_tokenizer(KerasBertWWMModel.dict_path)

    bert_dedup_model = build_model(KerasBertWWMModel.config_path, KerasBertWWMModel.checkpoint_path, keep_tokens)

    sentence_emb = extract_emb_feature(bert_dedup_model, bert_dedup_tokenizer, texts, 512, mask_if=True)

    sim_matrix = cosine_similarity(sentence_emb)

    sim_dict = {}

    for i in np.arange(sim_matrix.shape[0] - 1):
        for j in np.arange(sim_matrix.shape[1]):
            if j <= i:
                continue

            sim_dict[(i, j)] = sim_matrix[i][j]

    duplicated_doc = []

    for key in sim_dict.keys():
        if sim_dict[key] > BERT_SIMILARITY_THRESHOLD:
            duplicated_doc.append(key[0])

    dedup_news_df = news_df[~news_df["order_id"].isin(list(set(duplicated_doc)))]

    dedup_news_df.index = np.arange(dedup_news_df.shape[0])

    duplicated_doc_id = news_df[news_df["order_id"].isin(list(set(duplicated_doc)))].crawler_id.values

    _ = dedup_news_df.pop("order_id")

    return dedup_news_df, duplicated_doc_id
