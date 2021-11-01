import os
from enum import Enum
from celery.schedules import crontab
from kombu import Exchange, Queue

cwd_path = os.getcwd()


class CelerySentimentConfig(object):
    broker_url = 'amqp://datainfra:datainfra@10.10.50.32:5672/sentiment'
    result_backend = 'redis://:datainfra@redis@10.10.50.32:6379/0'
    # broker_url = 'redis://:datainfra@10.241.0.42:6379/0'
    # result_backend = 'redis://:datainfra@10.241.0.42:6379/0'
    worker_redirect_stdouts_level = 'INFO'
    timezone = 'Asia/Shanghai'
    task_acks_late = True #允许重试
    accept_content = ['pickle', 'json']
    worker_concurrency = 10
    #设置并发worker数量
    worker_max_tasks_per_child = 200 #每个worker最多执行500个任务被销毁，可以防止内存泄漏
    task_time_limit = 300
    worker_prefetch_multiplier = 0  #超时时间
    worker_log_format = "[%(asctime)s] [%(levelname)s] [%(processName)s] - %(message)s"
    worker_task_log_format = "[%(asctime)s] [%(levelname)s] [%(processName)s] [%(task_name)s:%(task_id)s] - %(message)s"
    beat_schedule = {
        'add-model-update': {
            'task': 'sentiment.update.update',
            'schedule': crontab(minute=0, hour='*/6'),
            'args': ()
        },
        'add-sentiment-calibration': {
            'task': 'sentiment.update.calibrate',
            'schedule': crontab(minute=0, hour='*/4'),
            'args': ()
        }
    }
    beat_schedule_filename = os.path.join(cwd_path, "sentiment-celerybeat-schedule")
    imports = (
        "sentiment.predict",
        "sentiment.update"
    )
    task_routes = {
        'sentiment.predict.predict': {'queue': 'for_predict', 'routing_key': 'for_predict'},
        'sentiment.predict.extract_region': {'queue': 'for_region', 'routing_key': 'for_region'},
        'sentiment.predict.extract_company_risks': {'queue': 'for_company_risk', 'routing_key': 'for_company_risk'},
        'sentiment.predict.extract_text_summary': {'queue': 'for_text_summary', 'routing_key': 'for_text_summary'},
        'sentiment.update.update': {'queue': 'for_update', 'routing_key': 'for_update'},
        'sentiment.update.calibrate': {'queue': 'for_update', 'routing_key': 'for_update'},
        'sentiment.update.fetch_calibrated_sentiment_results': {'queue': 'for_update', 'routing_key': 'for_update'}
    }
    task_queues = (
        Queue('for_predict', exchange=Exchange('sentiment', type='direct'), routing_key='for_predict'),
        Queue('for_update', exchange=Exchange('sentiment', type='direct'), routing_key='for_update'),
        Queue('for_region', exchange=Exchange('sentiment', type='direct'), routing_key='for_region'),
        Queue('for_company_risk', exchange=Exchange('sentiment', type='direct'), routing_key='for_company_risk'),
        Queue('for_text_summary', exchange=Exchange('sentiment', type='direct'), routing_key='for_text_summary')
    )


class CeleryTopicConfig(object):
    broker_url = 'amqp://datainfra:datainfra@10.10.50.32:5672/topic'
    result_backend = 'redis://:datainfra@redis@10.10.50.32:6379/1'
    # broker_url = 'redis://:datainfra@10.241.0.42:6379/0'
    # result_backend = 'redis://:datainfra@10.241.0.42:6379/0'
    worker_redirect_stdouts_level = 'INFO'
    timezone = 'Asia/Shanghai'
    task_acks_late = True #允许重试
    accept_content = ['pickle', 'json']
    worker_concurrency = 3
    worker_max_tasks_per_child = 200 #每个worker最多执行500个任务被销毁，可以防止内存泄漏
    task_time_limit = 1200
    worker_prefetch_multiplier = 0  #超时时间
    worker_log_format = "[%(asctime)s] [%(levelname)s] [%(processName)s] - %(message)s"
    worker_task_log_format = "[%(asctime)s] [%(levelname)s] [%(processName)s] [%(task_name)s:%(task_id)s] - %(message)s"
    beat_schedule = {
        'fetch_data': {
            'task': 'topic_analysis.run.fetch_data',
            'schedule': crontab(minute=10, hour=0),
            'args': ()
        },
        'trigger_dedup': {
            'task': 'topic_analysis.run.trigger_dedup',
            'schedule': crontab(minute=0, hour='*/4'),
            'args': ()
        }
    }
    beat_schedule_filename = os.path.join(cwd_path, "topic-celerybeat-schedule")
    imports = (
        "topic_analysis.run"
    )
    task_routes = {
        'topic_analysis.run.fetch_data': {'queue': 'for_fetch_data', 'routing_key': 'for_fetch_data'},
        'topic_analysis.run.fetch_results': {'queue': 'for_fetch_results', 'routing_key': 'for_fetch_results'},
        'topic_analysis.run.fetch_batch_results': {'queue': 'for_fetch_results', 'routing_key': 'for_fetch_results'},
        'topic_analysis.run.trigger_dedup': {'queue': 'for_dedup', 'routing_key': 'for_dedup'},
        'topic_analysis.run.fetch_dedup_results': {'queue': 'for_dedup', 'routing_key': 'for_dedup'}
    }
    task_queues = (
        Queue('for_fetch_data', exchange=Exchange('topic_analysis', type='direct'), routing_key='for_fetch_data'),
        Queue('for_fetch_results', exchange=Exchange('topic_analysis', type='direct'), routing_key='for_fetch_results'),
        Queue('for_dedup', exchange=Exchange('topic_analysis', type='direct'), routing_key='for_dedup')
    )


preferred_model = "TEXTCNN"
# preferred_model = "LSTM"

preprocess_keyword = "我爱我家"

nonsense_topic = ["体育",
                  "娱乐",
                  "游戏"]

spark_config = {
    "spark.driver.memory": "5g",
    "spark.executor.memory": "5g",
    "spark.driver.maxResultSize": "5g",
    "spark.executor.cores": 2,
    "spark.executor.instances": 4
}

spark_user = "xubin.xu"

batch_insert = 5


class MessageType(Enum):
    PREDICTED = "predicted"
    CALIBRATED = "calibrated"
    UPDATE_MODEL = "update_model"
    MARKET_TOPIC = "market_topic"
    PUBLIC_RELATION_TOPIC = "public_relation_topic"
    REAL_ESTATE_COMPANY_TOPIC = "real_estate_company_topic"
    DEDUP_DOC = "dedup"
    HOT_COMPANY_TOPIC = "hot_company_topic"


class TopicMessageType(Enum):
    ANALYZE_DATA = "analyze_data"
    GET_RESULT = "get_result"


class RedisConfig(object):
    host = '10.10.50.32'
    password = 'datainfra@redis'
    # host = '10.241.0.42'
    # password = 'datainfra'
    port = 6379


vocab_path = os.path.join(cwd_path, "resources/vocab.txt")

stopwords_path = os.path.join(cwd_path, "resources/CNstopwords.txt")

keywords_path = os.path.join(cwd_path, "resources/keywords.txt")

city_dict_path = os.path.join(cwd_path, "resources/city.json")

preprocess_model_path = [os.path.join(cwd_path, "resources/wawj_model.bin")]

word2vec_model_path = [os.path.join(cwd_path, "resources/news_model.txt")]

fasttext_model_path = [os.path.join(cwd_path, "resources/fasttext_model.bin")]

category_model_path = [os.path.join(cwd_path, "resources/category.h5")]

textcnn_model_path = [
                    os.path.join(cwd_path, "resources/textcnn_model.yml"),
                    os.path.join(cwd_path, "resources/textcnn_model.h5"),
                  ]

lstm_model_path = [
                    os.path.join(cwd_path, "resources/lstm_model.yml"),
                    os.path.join(cwd_path, "resources/lstm_model.h5"),
                  ]


topic_local_dict_path = os.path.join(cwd_path, "resources/localdict.txt")

topic_local_market_data_path = os.path.join(cwd_path, "topic_tmp/market_texts.csv")

topic_local_public_relation_data_path = os.path.join(cwd_path, "topic_tmp/public_relation_texts.csv")

topic_local_real_estate_data_path = os.path.join(cwd_path, "topic_tmp/real_estate_texts.csv")

topic_local_market_result_path = os.path.join(cwd_path, "topic_tmp/market_result.csv")

topic_local_market_top_n_per_topic_path = os.path.join(cwd_path, "topic_tmp/market_result_top_n_per_topic.csv")

topic_local_real_estate_result_path = os.path.join(cwd_path, "topic_tmp/real_estate_result.csv")

topic_local_real_estate_top_n_per_topic_path = os.path.join(cwd_path, "topic_tmp/real_estate_result_top_n_per_topic.csv")

topic_local_public_relation_result_path = os.path.join(cwd_path, "topic_tmp/public_relation_result.csv")

real_estate_company_path = os.path.join(cwd_path, "resources/real_estate_companies.txt")

concurrent_get_topic_data_number = 10

predict_table_key = "sentiment_predict_for_test"

update_message_key = "sentiment_update_for_test"

calibrated_sentiment_result_message_key = "calibrated_sentiment_result_for_test"

topic_message_key = "topic_analysis_for_test"

spark_topic_message_key = "spark_topic_analysis_for_test"

topic_result_message_key = "topic_result_for_test"

predict_table = "infra_ml.sentiment_predict_data_for_test"

calibrated_table = "infra_ml.sentiment_calibrated_data_for_test"

init_train_data_table = "infra_ml.sentiment_sent_train_data"

topic_data_table = "infra_ml.topic_data_for_test"

topic_result_table = "infra_ml.topic_results_for_test"

top_n_per_topic_result_table = "infra_ml.top_n_per_topic_results_for_test"

topic_last_process_date_key = "_analysis_last_process_day"

read_lock_filepath = os.path.join(cwd_path, "locks/read_lock")

write_lock_filepath = os.path.join(cwd_path, "locks/write_lock")


class TopicRequestConfig(object):
    default_page_num = 1
    default_page_size = 200
    request_url = "http://echelon.bkjk.cn/api/v2.0/crawler_data"
    username = "echelon"
    password = ""
    timeout = 100
    status_ok = "OK"
    success_message = "succeed"
    day_range = 31
    max_threads = 5


class CompanyType(Enum):
    MARKET = "市场部"
    PUBLIC_RELATION = "合规部"
    ANT_FINANCIAL = "蚂蚁金服"
    REAL_ESTATE = "开发商"


class GetResultTimeRange(Enum):
    ONE_WEEK = "one_week"
    HALF_MONTH = "half_month"
    ONE_MONTH = "one_month"
    ONE_DAY = "one_day"
    THREE_DAYS = "three_days"


get_result_time_range = {
                            GetResultTimeRange.ONE_WEEK.value: 7,
                            GetResultTimeRange.HALF_MONTH.value: 14,
                            GetResultTimeRange.ONE_MONTH.value: 31,
                            GetResultTimeRange.ONE_DAY.value: 1,
                            GetResultTimeRange.THREE_DAYS.value: 3
                        }

top_n_texts_per_topic = 3

topic_data_tmp_path = os.path.join(cwd_path, "topic_tmp/tmp_data")

orignal_ner_train_data_path = os.path.join(cwd_path, "resources/MSRA/train.txt")

ner_train_data_path = os.path.join(cwd_path, "resources/MSRA/ner_train.txt")

ner_train_data_pkl_path = os.path.join(cwd_path, "resources/MSRA/ner_train.pkl")

ner_model_path = os.path.join(cwd_path, "resources/ner_model/")

chinese_character_vector_path = os.path.join(cwd_path, "resources/chinese_charactor_vec.txt")

company_ner_train_template_path = os.path.join(cwd_path, "resources/company_ner_train.tpl")

ner_model_config = {
    "epochs": 3,
    "batch_size": 32,
    "lr": 0.001,
    "embedding_dim": 100,
    "pretrained": True,
    "dropout_keep": 0.5,
    "max_len": 50
}

max_ner_check_run = 3

ner_predict_req_table_key = "ner_predict_req_for_test"

company_risk_keywords_file = os.path.join(cwd_path, "resources/company_risk_keywords.txt")

calibrated_max_processes = 5

tidb_connection_config = {
    "host": "tidb.bkjk.cn",
    "user": "datainfra",
    "password": "",
    "db": "infra_ml",
    "port": 4000
}

calibrated_result_table = "infra_ml.sentiment_calibrated_data_result"

calibrated_result_standby_table = "infra_ml.sentiment_calibrated_data_result_standby"

calibrated_sentiment_label_csv_path = os.path.join(cwd_path, "resources/calibrated_sentiment_labels.csv")

calibrate_last_process_date_key = "calibrate_last_process_day"

calibrate_tmp_folder = os.path.join(cwd_path, "tmp_calibrate")


class CalibrateStatus(Enum):
    ONGOING = "ongoing"
    UPDATED = "updated"


dw_crawler_data_table = "spark_datacollector_mongo.echelon_data"

dw_crawler_data_file_path = os.path.join(cwd_path, "topic_tmp/all_crawler_data.csv")

sentiment_data_file_path = os.path.join(cwd_path, "topic_tmp/sentiment_data.csv")

preprocessed_real_estate_data_path = os.path.join(cwd_path, "topic_tmp/preprocessed_real_estate_texts.csv")

bert_ner_model_path = [
    os.path.join(cwd_path, "resources/bert_ner_model.yml"),
    os.path.join(cwd_path, "resources/best_ner_model.weights"),
]

chinese_bert_corpus_dir = os.path.join(cwd_path, "resources/chinese_L-12_H-768_A-12")

chinese_bert_subject_extraction_train_path = os.path.join(cwd_path, "resources/event_type_entity_extract_train.csv")

chinese_bert_subject_extraction_test_path = os.path.join(cwd_path, "resources/event_type_entity_extract_eval.csv")

chinese_bert_subject_extraction_random_order_filepath = os.path.join(cwd_path, "resources/random_order_train.json")

chinese_bert_subject_extraction_dev_filepath = os.path.join(cwd_path, "resources/dev_pred.json")

chinese_bert_subject_extraction_test_result_filepath = os.path.join(cwd_path, "resources/test_result.txt")


class RiskRelation(Enum):
    UNRELATED = "unrelated"
    RELATED_WITH_RISK = "related_with_risk"
    RELATED_WITHOUT_RISK = "related_without_risk"


class TextSummaryPyTorchConfig(object):
    vocab_path = os.path.join(cwd_path, "resources/chinese_roberta_wwm_ext_L-12_H-768_A-12_pytorch/vocab.txt")
    pretrained_model_name = "roberta"
    pretrained_model_path = os.path.join(cwd_path, "resources/chinese_roberta_wwm_ext_L-12_H-768_A-12_pytorch/pytorch_bert_model.bin")
    model_path = os.path.join(cwd_path, "resources/bert_auto_title_model_epoch_5.bin")
    batch_size = 16
    lr = 1e-5
    max_text_len = 320
    max_summary_len = 100


class TextSummaryTextRankConfig(object):
    word2vec_dim = 256
    summary_sentences_threshold = 10
    min_summary_sentences = 2
    max_summary_sentences = 3
    convergence_tolerence = 1e-02


class TextSummaryMethod(Enum):
    PYTORCH_BERT = "pytorch_bert"
    KERAS_BERT = "keras_bert"
    TEXTRANK = "text_rank"


text_summary_req_table_key = "text_summary_req_for_test"


# 使用bert模型去重
class KerasBertWWMModel(object):
    checkpoint_path = os.path.join(cwd_path, "resources/chinese_wwm_ext_L-12_H-768_A-12/bert_model.ckpt")
    config_path = os.path.join(cwd_path, "resources/chinese_wwm_ext_L-12_H-768_A-12/bert_config.json")
    dict_path = os.path.join(cwd_path, "resources/chinese_wwm_ext_L-12_H-768_A-12/vocab.txt")


class DedupMethod(Enum):
    TFIDF = "tf_idf"
    BERT = "bert"
    MIXED = "mixed"


dedup_req_key = "dedup_req_test"

duplicated_doc_path = os.path.join(cwd_path, "dup_tmp/duplicated_result_")

raw_data_path = os.path.join(cwd_path, "dup_tmp/raw_data.csv")


hot_company_keyword_config_path = os.path.join(cwd_path, "resources/hot_company_keywords.csv")

hot_company_keyword_data_path = os.path.join(cwd_path, "resources/hot_company_data.csv")

oversea_topic_config_path = os.path.join(cwd_path, "resources/oversea_topic_config.csv")

oversea_topic_data_path = os.path.join(cwd_path, "resources/oversea_topic_data.csv")
