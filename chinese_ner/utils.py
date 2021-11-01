import os
import re
import pickle
import config
import jieba
import collections
import pandas as pd
import numpy as np
from functools import partial
from utils import get_city_dict
from sklearn.model_selection import train_test_split
from config import ner_model_config
from chinese_ner import logger as app_log

tag2id = {
    '': 0,
    'B_ns': 1,
    'B_nr': 2,
    'B_nt': 3,
    'M_nt': 4,
    'M_nr': 5,
    'M_ns': 6,
    'E_nt': 7,
    'E_nr': 8,
    'E_ns': 9,
    'o': 10
}

id2tag = {
    0: '',
    1: 'B_ns',
    2: 'B_nr',
    3: 'B_nt',
    4: 'M_nt',
    5: 'M_nr',
    6: 'M_ns',
    7: 'E_nt',
    8: 'E_nr',
    9: 'E_ns',
    10: 'o'
}

skipped_company_names = ["控股有限公司", "房地产发展", "房产集团", "房产（集团）", "房地产发展（集团）", "房地产发展集团"]

g_special_company_prefix = ["城市", "中华", "中大", "经纬", "美好", "百城", "供销", "实业", "新华", "天安", "大华", "天山",
                            "地铁", "高速", "房地产", "首都", "旅游", "交通", "黄金", "时代", "建设", "国家", "光明",
                            "投资", "海东", "印象", "国民", "正中", "珠江", "湘江", "两江", "城建", "地产开发", "城市建设",
                            "湘江新区", "两江新区", "铁路", "渤海", "栖霞", "新天地"]


def flatten(x):
    result = []
    for el in x:
        if isinstance(x, collections.Iterable) and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result


def padding(ids):
    max_len = ner_model_config["max_len"]

    if len(ids) >= max_len:
        return ids[:max_len]

    ids.extend([0] * (max_len - len(ids)))
    return ids


def x_padding(words, word2id):
    max_len = ner_model_config["max_len"]

    ids = list(word2id[words])
    if len(ids) >= max_len:
        return ids[:max_len]
    ids.extend([0] * (max_len - len(ids)))
    return ids


def seg_word_tags(train_data_path):
    if not os.path.exists(train_data_path):
        app_log.error("{} does not exist".format(train_data_path))
        return None, None

    global tag2id, id2tag

    try:
        folder_path = train_data_path[:train_data_path.rfind('/')]

        input_data = open(train_data_path, 'r')
        output_data = open(os.path.join(folder_path, "wordtag.txt"), 'w')
        for line in input_data.readlines():
            line = line.strip().split()

            if len(line) == 0:
                continue
            for word in line:
                word = word.split('/')
                if word[1] != 'o':
                    if len(word[0]) == 1:
                        output_data.write(word[0] + "/B_" + word[1] + " ")
                    elif len(word[0]) == 2:
                        output_data.write(word[0][0] + "/B_" + word[1] + " ")
                        output_data.write(word[0][1] + "/E_" + word[1] + " ")
                    else:
                        output_data.write(word[0][0] + "/B_" + word[1] + " ")
                        for j in word[0][1: len(word[0]) - 1]:
                            output_data.write(j + "/M_" + word[1] + " ")
                        output_data.write(word[0][-1] + "/E_" + word[1] + " ")
                else:
                    for j in word[0]:
                        output_data.write(j + "/o" + " ")
            output_data.write('\n')

        input_data.close()
        output_data.close()

        data_entries = []
        label_entries = []

        input_data = open(os.path.join(folder_path, "wordtag.txt"), 'r')
        for line in input_data.readlines():
            line = re.split('[，。；！：？、‘’“”]/[o]', line.strip())
            for sen in line:
                sen = sen.strip().split()
                if len(sen) == 0:
                    continue

                line_data = []
                line_label = []
                num_not_o = 0

                for word in sen:
                    word = word.split('/')
                    line_data.append(word[0])
                    line_label.append(tag2id[word[1]])

                    if word[1] != 'o':
                        num_not_o += 1

                if num_not_o != 0:
                    data_entries.append(line_data)
                    label_entries.append(line_label)

        input_data.close()

        return data_entries, label_entries
    except Exception as e:
        app_log.error("Fail to parse train data: {}".format(e))
        return None, None


def make_train_pkl(data_entries, label_entries):
    try:
        all_words = flatten(data_entries)
        sr_allwords = pd.Series(all_words)
        sr_allwords = sr_allwords.value_counts()
        set_words = sr_allwords.index
        set_ids = range(1, len(set_words) + 1)
        word2id = pd.Series(set_ids, index=set_words)
        id2word = pd.Series(set_words, index=set_ids)

        word2id["unknown"] = len(word2id) + 1

        df_data = pd.DataFrame({'words': data_entries, 'tags': label_entries}, index=range(len(data_entries)))
        df_data['x'] = df_data['words'].apply(partial(x_padding, word2id=word2id))
        df_data['y'] = df_data['tags'].apply(padding)
        x = np.asarray(list(df_data['x'].values))
        y = np.asarray(list(df_data['y'].values))

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=43)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=43)

        with open(config.ner_train_data_pkl_path, 'wb') as outp:
            pickle.dump(word2id, outp)
            pickle.dump(id2word, outp)
            pickle.dump(tag2id, outp)
            pickle.dump(id2tag, outp)
            pickle.dump(x_train, outp)
            pickle.dump(y_train, outp)
            pickle.dump(x_test, outp)
            pickle.dump(y_test, outp)
            pickle.dump(x_valid, outp)
            pickle.dump(y_valid, outp)

        app_log.info('** Finished saving the data.')
    except Exception as e:
        app_log.error("Fail to generate train pkl file : {}".format(e))


def post_process_names(names, original_name, striped):
    new_names = set()

    for name in names:
        if len(name) <= 2:
            seg_list = jieba.cut(original_name[striped:])
            words = [word.strip() for word in seg_list]
            word = words[1]
            if word == "(" or word == "（" and len(words) > 2:
                word = words[2]

            if words[0] + words[1] == name:
                word = words[2]

            if word == "(" or word == "（" and len(words) > 3:
                word = words[3]

            if len(word) >= 3:
                p = word.find("股份")
                if p > 0:
                    word = word[:p]
            # new_names.add(name.strip())
            name = name + word

        if name.strip() in skipped_company_names:
            continue

        new_names.add(name.strip().strip("(").strip("（"))

        p = name.find("房地产")
        if p > 0:
            name = name[:p]
            new_names.add(name + "地产")
            new_names.add(name + "房产")

    new_names.add(original_name.strip())

    return new_names


def parse_company_name(name):
    original_name = name
    all_names = []

    sentry = -1

    key = "集团"
    p = name.find(key)
    if p > 0 and name[p - 1] not in ['(', '（']:
        name = name[:p]
        sentry = p

    key = "控股"
    p = name.find(key)
    if p > 0 and name[p - 1] not in ['(', '（']:
        name = name[:p]
        sentry = p

    key = "股份"
    p = name.find(key)
    if p > 0:
        name = name[:p]
        sentry = p

    key = "有限"
    p = name.find(key)
    if p > 0:
        name = name[:p]
        sentry = p

    key = "公司"
    p = name.find(key)
    if p > 0:
        name = name[:p]
        sentry = p

    if len(name) <= 2:
        seg_list = jieba.cut(original_name)
        words = [word.strip() for word in seg_list]
        if words[0] == name:
            word = words[1]
        elif words[0] + words[1] == name:
            word = words[2]
        if len(word) >= 3:
            p = word.find("股份")
            if p > 0:
                word = word[:p]
            p = word.find("集团")
            if p > 0:
                word = word[:p]

        name = name + word

    all_names.append(name)

    keys = ["()", "（）"]
    for key in keys:
        q1 = name.find(key[0])
        if q1 >= 0:
            q2 = name.find(key[1])
            tokens = re.split('[' + key[0] + key[1] + ']', name)
            if sentry - q2 > 1 and len(tokens) >= 3:
                all_names.append(tokens[0] + tokens[2])
                all_names.append(tokens[0] + tokens[1] + tokens[2])
            elif sentry - q2 == 1 and len(tokens) >= 2:
                all_names.append(name[:q1])
                all_names.append(tokens[0] + tokens[1])

    seg_list = jieba.cut(name)
    words = [word.strip() for word in seg_list]

    striped = 0
    location = ""
    region_dict = get_city_dict()

    if words[0][-1] == "市" or words[0][-1] == '省':
        if words[0][:-1] in region_dict:
            striped = len(words[0])
            location = words[0][:-1]
    elif words[0] in region_dict or words[0] == "中国":
        striped = len(words[0])
        location = words[0]

    if striped == 0:
        return location, post_process_names(all_names, original_name, striped)

    striped_names = [name[striped:] for name in all_names]
    all_names.extend(striped_names)

    return location, post_process_names(all_names, original_name, striped)


def extract_prefix(name):
    seg_list = jieba.cut(name)
    words = [word.strip() for word in seg_list]

    location = ""
    location_dict = get_city_dict()

    if words[0][-1] == "市" or words[0][-1] == '省':
        if words[0][:-1] in location_dict:
            location = words[0][:-1]
    elif words[0] in location_dict or words[0] == "中国":
        location = words[0]

    if len(location) == 0:
        if len(words[0]) == 1:
            prefix = words[0] + words[1]
        else:
            prefix = words[0]
    else:
        if len(words[1]) == 1:
            prefix = words[1] + words[2]
        else:
            prefix = words[1]
    # 雅戈尔集团用jieba分词不会拆成["雅戈尔","集团"]
    if prefix.strip().find("雅戈尔集团") == 0:
        prefix = "雅戈尔"

    if prefix.strip().find("花样") >= 0 and name.strip().find("花样年集团") >= 0:
        prefix = "花样年"

    if prefix.strip().find("北大") >= 0 and name.strip().find("北大资源") >= 0:
        prefix = "北大资源"

    if prefix.strip().find("城市") >= 0 and name.strip().find("城市建设") >= 0:
        prefix = "城市建设"

    if prefix.strip().find("地产") >= 0 and name.strip().find("地产开发") >= 0:
        prefix = "地产开发"

    if prefix.strip().find("湘江") >= 0 and name.strip().find("湘江新区") >= 0:
        prefix = "湘江新区"

    if prefix.strip().find("两江") >= 0 and name.strip().find("两江新区") >= 0:
        prefix = "两江新区"

    return location, prefix.strip()


def handle_special_rules(text, company_name, prefix):
    global g_special_company_prefix

    related = False

    if company_name == "国家能源投资集团有限责任公司":
        if text.find("能源投资") >= 0:
            related = True
    elif company_name == "天山房地产开发集团有限公司":
        if text.find("天山房地产") >= 0 or text.find("天山房产") >= 0 or text.find("天山地产") >= 0:
            related = True
    elif company_name == "光明房地产集团股份有限公司":
        if text.find("光明房地产") >= 0 or text.find("光明房产") >= 0 or text.find("光明地产") >= 0:
            related = True
    elif company_name == "上海海东房地产有限公司":
        if text.find("海东房地产") >= 0 or text.find("海东房产") >= 0 or text.find("海东地产") >= 0:
            related = True
    elif company_name == "大华(集团)有限公司":
        if text.find("大华(集团)") >= 0 or text.find("大华集团") >= 0 or text.find("大华（集团）") >= 0:
            related = True
    elif company_name.find("中国房地产开发集团") >= 0:
        if text.find("中国房地产开发集团") >= 0 or text.find("中国房产开发") >= 0 or text.find("中房开发") >= 0:
            related = True
    elif prefix == "新天龙":
        if text.find("天龙八部") < 0:
            related = True
    elif prefix in g_special_company_prefix:
        p = company_name.find(prefix)
        if p < 0:
            return False

        name = company_name[p + len(prefix):]

        seg_list = jieba.cut(name)
        words = np.asarray([word.strip() for word in seg_list])

        if len(words[0]) == 1:
            prefix_next = words[0] + words[1]
        else:
            prefix_next = words[0]

        if text.find(prefix + prefix_next) >= 0:
            related = True

    return related


def check_related_company(text, company_name, model):
    global g_special_company_prefix

    if model is None:
        app_log.error("Ner model is None")
        return True

    if not isinstance(text, str) or len(text.strip()) == 0 or \
            not isinstance(company_name, str) or len(company_name.strip()) == 0:
        app_log.error("text or company name is empty")
        return False

    company_location, company_prefix = extract_prefix(company_name)
    if len(company_location) > 0 and company_location != "中国":
        if text.find(company_location) < 0:
            return False

    related = False
    q = 0
    while True:
        p = text.find(company_prefix, q)
        if p < 0:
            break

        fraction = text[p: p + config.ner_model_config["max_len"] - 1]

        nt_entities = []

        try:
            nt_entities = model.extract_nt(fraction)
        except Exception as e:
            app_log.error(e)

        for nt in nt_entities:
            if nt.find(company_prefix) >= 0:
                related = True
                break

        app_log.info("company_prefix = {}".format(company_prefix))
        if company_prefix in g_special_company_prefix:
            related = handle_special_rules(fraction, company_name, company_prefix)

        if related:
            break

        q = p + len(company_prefix)

    return related


g_special_company_names = {"天津房地产集团有限公司": ["天津房地产集团"], "华夏幸福基业股份有限公司": ["华夏幸福"]}


def check_related_company_v2(text, company_name):
    try:
        if company_name in g_special_company_names:
            alias_list = g_special_company_names[company_name]
        else:
            location, alias_list = parse_company_name(company_name)
            alias_list = list(alias_list)

            if len(alias_list) > 0:
                if len(location.strip()) > 0:
                    if location != "中国" and text.find(location) < 0:
                        return False

                if len(location.strip()) > 0:
                    if location != "中国":
                        filter_cond = lambda alias: alias.find(location) < 0
                    else:
                        filter_cond = lambda alias: alias.find(location) >= 0

                    alias_list = [alias for alias in alias_list if filter_cond(alias)]

            if len(alias_list) == 0:
                alias_list.append(company_name)

            alias_list = sorted(alias_list, key=lambda x: len(x))
            if len(alias_list) > 1:
                if alias_list[-1].find(alias_list[-2]) >= 0:
                    alias_list = alias_list[:-1]

        for alias in alias_list:
            if text.find(alias) >= 0:
                return True
    except Exception as e:
        app_log.error("check company[{}] relation error: {}".format(company_name, e))

    return False
