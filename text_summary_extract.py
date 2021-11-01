import re
import json
import jieba
import glob
import torch
import datetime
import bert_seq2seq
import Levenshtein
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from itertools import chain
from torch.optim import Adam
from utils import get_logger
from sentiment import word2vec_model
from redis_db import RedisDBWrapper
import networkx as nx
from itertools import chain
from utils import get_stopwords
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset, DataLoader
from bert_seq2seq.tokenizer import Tokenizer, load_chinese_base_vocab
from bert_seq2seq.utils import load_bert, load_model_params, load_recent_model
from config import TextSummaryPyTorchConfig, TextSummaryTextRankConfig, text_summary_req_table_key, MessageType


logger = get_logger("text-summary-service")

_redis = RedisDBWrapper()

torch.cuda.set_device(1)

chinese_sentence_regex_pattern = "[。！；？!;?]"

chinese_sentence_seperators = ['。', '！', '？', '；', '!', ';', '?']

chinese_expression_regex_pattern = "[，。！；？：,!;?]"

chinese_expression_seperators = ['，', '。', '！', '；', '？', '：', '；', ',', '!', ';', '?']

chinese_intra_expression_seperators = ["，", "：", "、", "；", ",", ":", ";"]


def complete_sentence(summary):

    if summary[-1] not in chinese_expression_seperators:
        summary += "。"
        return summary

    if summary[-1] in chinese_intra_expression_seperators:
        summary = summary[:-1] + '。'
        return summary

    return summary


class TextSummaryPyTorchBertModel(object):

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        word2idx, keep_tokens = load_chinese_base_vocab(TextSummaryPyTorchConfig.vocab_path, simplfied=True)

        self.bert_model = load_bert(word2idx, model_name=TextSummaryPyTorchConfig.pretrained_model_name)

        load_model_params(self.bert_model, TextSummaryPyTorchConfig.pretrained_model_path, keep_tokens=keep_tokens)

        load_recent_model(self.bert_model, TextSummaryPyTorchConfig.model_path)

        self.bert_model.to(self.device)

        self.bert_model.eval()

    @staticmethod
    def postprocess(summary):

        summary = summary.replace("(图)", "").replace("组图：", "").replace("\"\"", "").\
            replace("组图，", "").replace("现场图", "")

        if summary.count("（") != summary.count("）"):
            summary = summary.replace("）", "").replace("（", "")

        if summary.count("(") != summary.count(")"):
            summary = summary.replace("(", "").replace(")", "")

        #if summary.count("\"") % 2 == 1:
        #    summary = summary.replace("\"", "")
        summary = summary.replace("\"", "")

        #if summary.count("“") != summary.count("”"):
        #    summary = summary.replace("“", "").replace("”", "")
        summary = summary.replace("“", "").replace("”", "")

        return summary

    @staticmethod
    def process_single_sentence(sentence):

        sentence = sentence.replace(" ", "")

        sentence = sentence.replace("租租", "租赁")

        p = sentence.find("什么是")
        if p > 0:
            sentence = sentence[:p]

        seg_list = jieba.cut(sentence)

        words = [item for item in seg_list if len(item.strip()) > 0]

        sim_mat = np.zeros([len(words), len(words)])

        for i in range(len(words)):
            for j in range(len(words)):
                if i < j:
                    sim_mat[i][j] = Levenshtein.jaro(words[i], words[j])

        high_sim_index = np.argwhere(sim_mat >= 0.95)
        del_index = []

        for idx in high_sim_index:
            if idx[1] in del_index:
                continue

            del_index.append(idx[1])

        remained_idx = list(set(np.arange(len(words)).tolist()).difference(set(del_index)))

        words = np.asarray(words)

        words = words[remained_idx]

        sentence = "".join(words)

        new_sentence = ""

        sentence_idx = 0

        sentence += "  "

        while sentence_idx < len(sentence) - 2:
            new_sentence += sentence[sentence_idx]

            if sentence[sentence_idx] == sentence[sentence_idx + 1] and sentence[sentence_idx] == sentence[sentence_idx + 2]:
                sentence_idx += 3
            else:
                sentence_idx += 1

        return new_sentence.strip()

    @staticmethod
    def fix_alphabetic_numeric_value(summary, text):

        PREV_TEXT_LEN = 5
        p = 0
        text_p = 0

        numbers = re.findall(r"[0-9.A-Za-z]+\D", summary)
        for number in numbers:
            #if number.find("年") > 0 or number.find("月") > 0 or number.find("日") > 0:
            #    continue

            p = summary.find(number, p + 1)
            if p == 0:
                continue

            prev_pos = p - PREV_TEXT_LEN if p - PREV_TEXT_LEN > 0 else 0
            prevText = summary[prev_pos: p]

            text_p = text.find(prevText, text_p + 1)

            text_numbers = re.findall(r"[0-9.A-Za-z]+\D", text[text_p:])
            if len(text_numbers) == 0:
                continue

            valid_text_number = text_numbers[0]
            for text_number in text_numbers:
                if prevText.find(text_number) < 0:
                    valid_text_number = text_number
                    break

            pattern = re.compile(prevText + r'([0-9.A-Za-z]+)\D')
            summary = pattern.sub(prevText + valid_text_number, summary)

        return summary

    @staticmethod
    def dudup_summary(summary):
        line_split = re.split(r'{}'.format(chinese_expression_regex_pattern), summary.strip())

        sim_mat = np.zeros([len(line_split), len(line_split)])

        for i in range(len(line_split)):
            for j in range(len(line_split)):
                if i < j:
                    sim_mat[i][j] = Levenshtein.jaro(line_split[i].replace(" ", ""), line_split[j].replace(" ", ""))

        high_sim_index = np.argwhere(sim_mat >= 0.8)
        del_index = []

        for idx in high_sim_index:
            if idx[1] in del_index:
                continue

            del_index.append(idx[1])

        remained_idx = list(set(np.arange(len(line_split)).tolist()).difference(set(del_index)))

        sentence_list = np.asarray(line_split)

        sentence_list = sentence_list[remained_idx]

        sentence_list = [sentence for sentence in sentence_list if len(sentence.strip()) > 0]

        for i, sentence in enumerate(sentence_list):
            sentence_list[i] = TextSummaryPyTorchBertModel.process_single_sentence(sentence)

        new_summary = ""

        summary = summary.replace(" ", "")

        for i in np.arange(len(sentence_list) - 1):

            p = summary.find(sentence_list[i + 1])
            if p < 0:
                delimit = "，"
            else:
                delimit = summary[p - 1]

            new_summary += sentence_list[i] + delimit

        if len(sentence_list[-1]) > 3:
            new_summary += sentence_list[-1]

        return complete_sentence(new_summary)

    def get_text_summary(self, text):

        summary = self.bert_model.generate(text, beam_size=10, device=self.device,
                                           out_max_length=TextSummaryPyTorchConfig.max_summary_len,
                                           max_length=TextSummaryPyTorchConfig.max_text_len)
        summary = summary.replace("[UNK]", "\"")

        new_summary = TextSummaryPyTorchBertModel.dudup_summary(summary)

        new_summary = TextSummaryPyTorchBertModel.postprocess(new_summary)

        new_summary = TextSummaryPyTorchBertModel.fix_alphabetic_numeric_value(new_summary, text)

        p = new_summary.find(" ")
        if p > 0:
            new_summary = new_summary[:p]

        p = new_summary.find("。")
        if p > 0:
            new_summary = new_summary[: p + 1]

        new_summary = new_summary.replace("[夜宵]", "")
        new_summary = new_summary.replace("[拌饭]", "")

        new_summary = new_summary.replace("百%", "100%")

        if len(new_summary) > len(text):
            return text

        return complete_sentence(new_summary)

    def get_text_overall_summary(self, text):

        try:
            text = text.replace("▲", "").replace("●", "")
            text = text.replace("??", "").replace("？？", "")
            text = text.replace("!!", "").replace("！！", "")
            text = text.replace("。。", "")
            text = text.replace(";;", "").replace("；；", "")
            text = text.replace("<", "").replace(">", "")
            text = text.replace("《", "").replace("》", "")
            text = text.replace("|", "，").replace("丨", "，")
            text = text.replace("——", "，")
            text = text.replace("\\\\", "").replace("//", "")

            if len(text) <= TextSummaryPyTorchConfig.max_text_len * 2:
                summary = self.get_text_summary(text).strip()
                if len(summary.strip()) == 0:
                    return text, text

                return summary, summary

            summary_1 = self.get_text_summary(text[: TextSummaryPyTorchConfig.max_text_len])

            next_text = text[TextSummaryPyTorchConfig.max_text_len:]
            p = next_text.find("。")
            if p < 0:
                p = next_text.find("！")

            if p < 0:
                p = next_text.find("？")

            if p < 0:
                return summary_1, summary_1

            next_text = next_text[p + 1:]
            if len(next_text) < TextSummaryPyTorchConfig.max_summary_len:
                return summary_1, summary_1

            summary_2 = self.get_text_summary(next_text)

            summary = summary_1 + summary_2

            summary = summary.strip()

            return summary_1, self.dudup_summary(summary)
        except Exception as e:
            logger.error("seq2seq summary failed: {}".format(e))

            return "", ""


class TextSummaryTextRankModel(object):

    _stopwords = get_stopwords()

    @staticmethod
    def segment(sentence):

        sentence = re.sub(r'[^\u4e00-\u9fa5]+', '', sentence)

        sentence_depart = jieba.cut(sentence.strip())

        word_list = []

        for word in sentence_depart:
            if word not in TextSummaryTextRankModel._stopwords:
                word_list.append(word)

        return word_list

    @staticmethod
    def postprocess(summary):

        summary = summary.replace("现场图", "")
        summary = summary.replace(" ", "")
        summary = summary.replace("\\n", "").replace("\n", "")
        summary = summary.replace("\"", "").replace("\\\\", "").replace("//", "")
        summary = summary.replace("“", "").replace("”", "")

        # 删除类似于如下句子中的
        # 第五，持续深耕金融机构，加大低成本开发贷的融资占比，优化财务结构，提升评级水平，降低整体的融资成本
        # 第五

        line_split = re.split(r'[，,、]', summary.strip())
        short_words = [x for x in line_split if len(x.strip()) < 4]

        if len(short_words) > 0:
            short_word = short_words[0]
        else:
            return summary

        p = summary.find(short_word)
        summary = summary[p + len(short_word):].strip()

        if summary[0] in ['，', ',', '、']:
            summary = summary[1:].strip()

        return summary


    @staticmethod
    def get_text_rank_summary(text):

        try:
            p = text.find("免责声明")
            if p > 0:
                text = text[:p]

            text = text.replace("▲", "").replace("●", "")
            text = text.replace("。。", "")
            text = text.replace(";;", "").replace("；；", "")
            text = text.replace("??", "").replace("？？", "")
            text = text.replace("!!", "").replace("！！", "")
            text = text.replace("<", "").replace(">", "")

            sentences_list = []
            line_split = re.split(r'{}'.format(chinese_sentence_regex_pattern), text.strip())
            line_split = [line.strip() for line in line_split if line.strip() not in chinese_sentence_seperators
                          and len(line.strip()) > 1]
            sentences_list.append(line_split)

            sentences_list = list(chain.from_iterable(sentences_list))

            sentence_word_list = []
            for sentence in sentences_list:
                line_seg = TextSummaryTextRankModel.segment(sentence)
                sentence_word_list.append(line_seg)

            _, word_embeddings = word2vec_model.get()

            sentence_vectors = []
            for sentence_words in sentence_word_list:
                if len(sentence_words) != 0:
                    v = sum([word_embeddings.get(w, np.zeros((TextSummaryTextRankConfig.word2vec_dim,))) for w in sentence_words]) / (len(sentence_words))
                else:
                    v = np.zeros((TextSummaryTextRankConfig.word2vec_dim,))
                sentence_vectors.append(v)

            sim_mat = np.zeros([len(sentences_list), len(sentences_list)])

            for i in range(len(sentences_list)):
                for j in range(len(sentences_list)):
                    if i != j:
                        sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1, TextSummaryTextRankConfig.word2vec_dim),
                                                          sentence_vectors[j].reshape(1, TextSummaryTextRankConfig.word2vec_dim))[0, 0]

            nx_graph = nx.from_numpy_matrix(sim_mat)

            scores = nx.pagerank(nx_graph, tol=TextSummaryTextRankConfig.convergence_tolerence)

            ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences_list)), reverse=True)

            if len(ranked_sentences) == 1:
                summary = TextSummaryTextRankModel.postprocess(ranked_sentences[0][1] + "。")
                return summary, summary

            sn = TextSummaryTextRankConfig.max_summary_sentences if len(ranked_sentences) > TextSummaryTextRankConfig.summary_sentences_threshold \
                else TextSummaryTextRankConfig.min_summary_sentences

            summary = ""

            for i in range(sn):
                summary += TextSummaryTextRankModel.postprocess(ranked_sentences[i][1]) + "。"

            top_summary = TextSummaryTextRankModel.postprocess(ranked_sentences[0][1]) + "。"

            return top_summary,  summary
        except Exception as e:
            logger.error("Text rank summary failed: {}".format(e))
            
            return "", ""


def match_summary(seq2seq_summary, text_rank_summary):

    line_split = re.split(r'{}'.format(chinese_expression_regex_pattern), seq2seq_summary.strip())
    line_split = [line.strip() for line in line_split if line.strip() not in chinese_expression_seperators and
                  len(line.strip()) > 1]
    last_seq2seq_expression = line_split[-1]

    text_rank_expressions = []
    line_split = re.split(r'{}'.format(chinese_expression_regex_pattern), text_rank_summary.strip())
    line_split = [line.strip() for line in line_split if line.strip() not in chinese_expression_seperators and
                  len(line.strip()) > 1]
    text_rank_expressions.append(line_split)

    text_rank_expressions = list(chain.from_iterable(text_rank_expressions))

    text_rank_sentences = []
    line_split = re.split(r'{}'.format(chinese_sentence_regex_pattern), text_rank_summary.strip())
    line_split = [line.strip() for line in line_split if line.strip() not in chinese_sentence_seperators and
                  len(line.strip()) > 1]
    text_rank_sentences.append(line_split)

    text_rank_sentences = list(chain.from_iterable(text_rank_sentences))

    p = seq2seq_summary.rfind(last_seq2seq_expression)

    if len(last_seq2seq_expression) < 4:
        return complete_sentence(seq2seq_summary[:p])

    seq2seq_summary_2 = seq2seq_summary[p:]

    for expression in text_rank_expressions:

        if expression.find(last_seq2seq_expression) >= 0:

            sentence = [x for x in text_rank_sentences if x.find(expression) >= 0][0]
            q = sentence.find(last_seq2seq_expression)

            seq2seq_summary_2 = seq2seq_summary_2.replace(last_seq2seq_expression, sentence[q:], 1)
            break

    return complete_sentence(seq2seq_summary[:p] + seq2seq_summary_2)


def get_sentence_vec(sentence):
    _, word_embeddings = word2vec_model.get()

    seq2seq_sentence = TextSummaryTextRankModel.segment(sentence)

    seq2seq_sentence_vec = np.array([word_embeddings.get(word, np.zeros((TextSummaryTextRankConfig.word2vec_dim,)))
                                     for word in seq2seq_sentence])

    return np.mean(seq2seq_sentence_vec, axis=0)


def get_complement_sentence(seq2seq_summary, text_rank_summary):

    seq2seq_sentence_vec = get_sentence_vec(seq2seq_summary)

    line_split = re.split(r'{}'.format(chinese_sentence_regex_pattern), text_rank_summary.strip())
    line_split = [line.strip() for line in line_split if line.strip() not in chinese_sentence_seperators
                  and len(line.strip()) > 1]

    text_rank_sentences = [TextSummaryTextRankModel.segment(line) for line in line_split]

    sim_all_scores = []

    _, word_embeddings = word2vec_model.get()

    for i, sentence in enumerate(text_rank_sentences):
        sentence_vec = np.array([word_embeddings.get(word, np.zeros((TextSummaryTextRankConfig.word2vec_dim,)))
                                 for word in sentence])

        sentence_vec = np.mean(sentence_vec, axis=0)

        score = cosine_similarity(seq2seq_sentence_vec.reshape(1, TextSummaryTextRankConfig.word2vec_dim),
                                  sentence_vec.reshape(1, TextSummaryTextRankConfig.word2vec_dim))[0, 0]

        sim_all_scores.append((i, score))

    sim_scores = [x for x in sim_all_scores if x[1] > 0.25]

    if len(sim_scores) > 0:
        sim_scores = sorted(sim_scores, key=lambda x: x[1])
    else:
        sim_scores = sorted(sim_all_scores, key=lambda x: x[1], reverse=True)

    return line_split[sim_scores[0][0]] + "。"


def extract_summary_by_text_rank(text):
    _, summary = TextSummaryTextRankModel.get_text_rank_summary(text)
    if len(summary.strip()) == 0:
        raise Exception("Text rank summary failed")

    return summary


def get_longest_sentence(text):
    sentences = re.split(r'{}'.format(chinese_sentence_regex_pattern), text.strip())
    sentences = [line.strip() for line in sentences if line.strip() not in chinese_sentence_seperators and len(line.strip()) > 1]

    sentences = sorted(sentences, key=lambda x: len(x), reverse=True)

    return sentences[0] + "。"


def extract_summary_by_seq2seq(model, text):
    summary, seq2seq_summary = model.get_text_overall_summary(text)
    if len(seq2seq_summary.strip()) == 0:
        return ""

    _, text_rank_summary = TextSummaryTextRankModel.get_text_rank_summary(text)

    if len(text_rank_summary.strip()) > 0:
        sentence = get_complement_sentence(summary, text_rank_summary)

        if summary[-1] not in chinese_sentence_seperators:
            summary += '。'

        summary = summary + sentence
    else:
        summary = seq2seq_summary

    if seq2seq_summary.count("总裁") >= 1:
        summary = extract_summary_by_mixed(model, text)

    p = seq2seq_summary.find('%')
    if 0 <= p < 10:
        summary = get_longest_sentence(text_rank_summary)

    return complete_sentence(summary)


def extract_summary_by_mixed_internal(model, text_rank_summary, text):

    _, seq2seq_summary = model.get_text_overall_summary(text_rank_summary)

    summary = match_summary(seq2seq_summary, text_rank_summary)

    rank_sentence = get_longest_sentence(text_rank_summary)

    summary_vec = get_sentence_vec(summary)

    rank_sentence_vec = get_sentence_vec(rank_sentence)

    score = cosine_similarity(summary_vec.reshape(1, TextSummaryTextRankConfig.word2vec_dim),
                              rank_sentence_vec.reshape(1, TextSummaryTextRankConfig.word2vec_dim))[0, 0]

    logger.info("summary_vec with rank_sentence_vec similarity: {}".format(score))

    if score > 0.85:
        seq2seq_summary, _ = model.get_text_overall_summary(text)

        seq2seq_summary_vec = get_sentence_vec(seq2seq_summary)
        score = cosine_similarity(summary_vec.reshape(1, TextSummaryTextRankConfig.word2vec_dim),
                                  seq2seq_summary_vec.reshape(1, TextSummaryTextRankConfig.word2vec_dim))[0, 0]

        logger.info("summary_vec with seq2seq_summary_vec similarity: {}".format(score))

        if score > 0.89:
            return seq2seq_summary if len(seq2seq_summary) > len(summary) else summary
        else:
            return seq2seq_summary + summary

    return summary + rank_sentence


def extract_summary_by_mixed(model, text):
    _, text_rank_summary = TextSummaryTextRankModel.get_text_rank_summary(text)
    if len(text_rank_summary.strip()) > 0:
        return extract_summary_by_mixed_internal(model, text_rank_summary, text)
    else:
        next_text = text[TextSummaryPyTorchConfig.max_text_len:]
        p = next_text.find("。")
        if p < 0:
            p = next_text.find("！")

        if p < 0:
            p = next_text.find("？")

        if p < 0:
            _, summary = model.get_text_overall_summary(next_text)
            return summary

        next_text = next_text[p + 1:]

        _, text_rank_summary = TextSummaryTextRankModel.get_text_rank_summary(next_text)

        if len(text_rank_summary.strip()) > 0:
            return extract_summary_by_mixed_internal(model, text_rank_summary, next_text)
        else:
            _, summary = model.get_text_overall_summary(next_text)
            return summary


def _execute():

    logger.info("start text summary model daemon")

    pytorchmodel = TextSummaryPyTorchBertModel()

    while True:
        try:
            message = _redis.pop_data(text_summary_req_table_key, True)
            logger.info(message)

            if 'type' not in message:
                raise Exception("Data miss type field")

            if message['type'] != MessageType.PREDICTED.value:
                raise Exception("Unrecognized type field")

            if "algorithm" in message and message["algorithm"] == "text_rank":
                summary = extract_summary_by_text_rank(message['content'])
            elif "algorithm" in message and message["algorithm"] == "single_seq2seq":
                _, summary = pytorchmodel.get_text_overall_summary(message['content'])
                if len(summary) == 0:
                    summary = extract_summary_by_text_rank(message['content'])
            elif "algorithm" in message and message["algorithm"] == "seq2seq":
                summary = extract_summary_by_seq2seq(pytorchmodel, message['content'])

                if len(summary) >= len(message['content']) // 3:
                    summary, _ = pytorchmodel.get_text_overall_summary(message['content'])

                if len(summary) == 0:
                    summary = extract_summary_by_text_rank(message['content'])
            else:
                summary = extract_summary_by_mixed(pytorchmodel, message['content'])
                if len(summary) == 0:
                    summary = extract_summary_by_text_rank(message['content'])

            logger.info(summary)

            data = {
                "type": MessageType.PREDICTED.value,
                "status": "ok",
                "uid": message['uid'],
                "summary": summary,
                "datetime": datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
            }

            if _redis.push_data(data, message['uid']) < 0:
                logger.error("Fail to push data to redis queue: {}".format(message['uid']))

        except Exception as e:
            logger.error(e)

            data = {
                "type": MessageType.PREDICTED.value,
                "status": "error",
                "uid": message['uid'],
                "summary": str(e),
                "datetime": datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
            }

            if _redis.push_data(data, message['uid']) < 0:
                logger.error("Fail to push data to redis queue: {}".format(message['uid']))


if __name__ == "__main__":
    _execute()
