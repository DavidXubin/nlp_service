import os
import re
import sys
import config
from chinese_ner import utils
from chinese_ner import logger as app_log
from chinese_ner.bilstm_crf import Model
from ner_predict_service import check_all_risks
from ner_predict_service import load_ner_model, init_tokenizer

TRAIN_DUPLICATION = 6
TEST_DUPLICATION = 3


def make_company_train_data():
    global TRAIN_DUPLICATION, TEST_DUPLICATION

    if not os.path.exists(config.real_estate_company_path):
        app_log.error("{} does not exist".format(config.real_estate_company_path))
        return False

    with open(config.real_estate_company_path) as f:
        companies = f.readlines()

    if len(companies) == 0:
        return False

    if not os.path.exists(config.company_ner_train_template_path):
        app_log.error("{} does not exist".format(config.company_ner_train_template_path))
        return False

    with open(config.company_ner_train_template_path, 'r') as f:
        ner_train_template = f.read()

    train_sentences = []
    test_sentences = []

    for company in companies:
        _, company_alias_list = utils.parse_company_name(company)
        if len(company_alias_list) == 0:
            continue

        sentence = [ner_train_template.replace("<<公司>>", alias.strip()) for alias in company_alias_list]
        train_sentences.extend(sentence * TRAIN_DUPLICATION)
        test_sentences.extend(sentence * TEST_DUPLICATION)

    train_sentences = '\n'.join(train_sentences)
    test_sentences = '\n'.join(test_sentences)

    if not os.path.exists(config.orignal_ner_train_data_path):
        app_log.error("{} does not exist".format(config.orignal_ner_train_data_path))
        return False

    with open(config.orignal_ner_train_data_path, 'r') as f:
        train_data = f.read()

    train_data = train_sentences + "\n" + train_data
    train_data = train_data + "\n" + test_sentences

    f = open(config.ner_train_data_path, 'w')
    f.write(train_data)
    f.close()

    return True


def make_train_data():

    make_company_train_data()
    train_data_path = config.ner_train_data_path

    data_entries, label_entries = utils.seg_word_tags(train_data_path)
    if data_entries is None:
        app_log.error("Fail to segment word list with entities from train data")
        return

    utils.make_train_pkl(data_entries, label_entries)

    _model = Model(config.NerModelIntention.TRAIN)
    _model.train()


if __name__ == "__main__":
    if len(sys.argv) == 1 or len(sys.argv) == 2 and sys.argv[1] == "train":
        make_train_data()
    elif sys.argv[1] == "test":
        while True:
            text = input("Enter your text: ")
            cop = re.compile(re.compile(u'[^0-9a-zA-Z\u4e00-\u9fa5.，,。？“”]+', re.UNICODE))
            text = cop.sub(" ", text)
            if text.strip() == "quit":
                break
            company = input("Enter your company: ")
            related = utils.check_related_company_v2(text, company)
            print("Text related with company: {}".format(related))

    elif sys.argv[1] == "test_file":
        filepath = input("Enter your file path: ")
        with open(filepath, 'r') as f:
            text = f.read()
        print("Text length is {}".format(len(text)))
        company = input("Enter your company: ")
        related = utils.check_related_company_v2(text, company)
        print("Text related with company: {}".format(related))

    elif sys.argv[1] == "bert_test":
        model = load_ner_model()
        tokenizer = init_tokenizer()

        while True:
            text = input("Enter your text: ")
            cop = re.compile(re.compile(u'[^0-9a-zA-Z\u4e00-\u9fa5.，,。？“”]+', re.UNICODE))
            text = cop.sub(" ", text)
            if text.strip() == "quit":
                break
            company = input("Enter your company: ")
            risks = check_all_risks(text, company, tokenizer, model)
            print("Company has risks: {}".format(risks))

    elif sys.argv[1] == "bert_test_file":
        model = load_ner_model()
        tokenizer = init_tokenizer()

        filepath = input("Enter your file path: ")
        with open(filepath, 'r') as f:
            text = f.read()
        print("Text length is {}".format(len(text)))
        company = input("Enter your company: ")
        risks = check_all_risks(text, company, tokenizer, model)
        print("Company has risks: {}".format(risks))
