from enum import Enum
import numpy as np
import jieba

from utils import get_keywords


class BinaryLabel(Enum):
    FALSE = 0
    TRUE = 1


class InternalLabel(Enum):
    NEGATIVE = 0
    NEUTRAL = 1
    POSITIVE = 2


class ExternalLabel(Enum):
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    NONSENSE = -10
    EXCEPTION = -100
    THIRD_PARTY_NEGATIVE = -200 #sentiment label is not decided by model


class Model(object):

    _label_maps = {InternalLabel.NEGATIVE: ExternalLabel.NEGATIVE,
                   InternalLabel.NEUTRAL: ExternalLabel.NEUTRAL,
                   InternalLabel.POSITIVE: ExternalLabel.POSITIVE}

    def __init__(self):
        self._model = None
        jieba.add_word("金服")
        jieba.add_word("花呗")
        jieba.add_word("借呗")
        jieba.add_word("我爱我家")

    def __del__(self):
        if self._model is not None:
            del self._model
            self._model = None

    def load(self, path):
        pass

    @staticmethod
    def extract(text, req_keywords):
        key_words = get_keywords(req_keywords)

        text = text.replace('\n', '')
        text = text.replace('【', '')
        text = text.replace('】', '')

        keywords_count = {}
        total_count = 0

        for word in key_words:
            count = text.count(word)
            if count == 0:
                continue
            keywords_count[word] = count
            total_count += count

        if len(keywords_count) == 1 and '贝壳粉' in text:
            return None

        if total_count == 0:
            if isinstance(req_keywords, str):
                req_keywords_list = req_keywords.split(",")
            elif isinstance(req_keywords, list):
                req_keywords_list = req_keywords
            else:
                return None
            for new_keywords in req_keywords_list:
                word_length = len(new_keywords)
                if word_length > 3:
                    cut_key = [x for x in jieba.cut(new_keywords.strip()) if len(x) > 1]
                    exist_key = [x for x in cut_key if x in text]
                    if word_length < 7:
                        if len(exist_key) != 0:
                            return text
                    else:
                        if len(exist_key) >= len(cut_key) * 0.5:
                            return text
            return None

        if total_count == 1:
            if len(text) / total_count > 500:
                extracted_text = ''

                key_index = text.index(list(keywords_count)[0])
                random_cut_num = np.random.randint(-50, 50)
                random_cut_index = 200 + random_cut_num
                if key_index < random_cut_index:
                    extracted_text += text[: key_index]
                else:
                    extracted_text += text[key_index - random_cut_index: key_index]

                if len(text) - key_index < random_cut_index:
                    extracted_text += text[key_index:]
                else:
                    extracted_text += text[key_index: key_index + random_cut_index]
            else:
                extracted_text = text
        else:
            extracted_text = text

        return extracted_text

    @staticmethod
    def modify_content(sentence):
        """
        Cut long text to get effect part, can be apply by pandas dataframe
        :param sentence: original sentence, string
        :return: modified sentence, string
        """
        if len(sentence) > 600:
            cut_index = len(sentence) // 10
            return sentence[:-cut_index]
        else:
            return sentence

    @staticmethod
    def preprocess(text, products=None):
        pass

    def predict(self, sentence):
        pass

    @staticmethod
    def build(train_data_df):
        pass

    @staticmethod
    def save(model, model_filepath):
        pass



