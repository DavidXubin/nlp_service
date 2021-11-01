import os
import re
import json
import datetime
import pandas as pd
from multiprocessing import Process
import config
from rw_flock import RWFlocker
from pyspark_manager import PySparkMgr
from sentiment.models.model import ExternalLabel, InternalLabel
from sentiment.models.fasttext import FastTextModel
from sentiment.models.lstm import LSTMModel
from sentiment.models.textcnn import TextCNNModel
from redis_db import RedisDBWrapper
from utils import get_logger, has_company_risk_keywords, insert_dataframe_into_db
from chinese_ner.bilstm_crf import Model as NerModel
from chinese_ner.utils import check_related_company_v2
from database import Database

logger = get_logger("sentiment-update-service")

_model_factory = {
    "Fasttext": (FastTextModel, config.fasttext_model_path),
    "LSTM": (LSTMModel, config.lstm_model_path),
    "TEXTCNN": (TextCNNModel, config.textcnn_model_path)
}

_redis = RedisDBWrapper()


def set_calibrate_status(company_type, status):
    if status in [config.CalibrateStatus.ONGOING.value, config.CalibrateStatus.UPDATED.value]:
        _redis.get_handler().set(company_type + "_calibrated_status", status)


def get_calibrate_status(company_type):
    return _redis.get_handler().get(company_type + "_calibrated_status")


def set_last_calibrate_day(day):
    _redis.get_handler().set(config.calibrate_last_process_date_key, day)


def get_last_calibrate_day():
    return _redis.get_handler().get(config.calibrate_last_process_date_key)


def update_train_data(spark):

    sql = "select count(*) from " + config.calibrated_table
    calibrated_count_df = spark.sql(sql).toPandas()
    if calibrated_count_df.values[0][0] == 0:
        return None

    sql = "select * from " + config.calibrated_table
    df = spark.sql(sql).toPandas()

    RWFlocker.lock(RWFlocker.WRITE)
    df.to_csv(config.calibrated_sentiment_label_csv_path, sep=',', header=True, index=False)
    RWFlocker.unlock()

    sql = "select p.uid, p.label, c.label as new_label, p.content " \
          "from " + config.predict_table + " as p left join " + config.calibrated_table + " as c on p.uid = c.uid"
    new_data_df = spark.sql(sql).toPandas()

    for idx in new_data_df.index:
        if not pd.isnull(new_data_df.loc[idx, "new_label"]):
            new_data_df.loc[idx, 'label'] = new_data_df.loc[idx, 'new_label']

    ext2int_label_maps = {ExternalLabel.NEGATIVE.value: InternalLabel.NEGATIVE.value,
                          ExternalLabel.NEUTRAL.value: InternalLabel.NEUTRAL.value,
                          ExternalLabel.POSITIVE.value: InternalLabel.POSITIVE.value}

    new_data_df = new_data_df[new_data_df['label'] >= ExternalLabel.NEGATIVE.value]
    new_data_df['label'] = new_data_df['label'].map(lambda x: ext2int_label_maps[x])

    new_data_df.drop(['new_label'], axis=1, inplace=True)
    new_data_df['label'] = new_data_df['label'].astype(int)

    sql = "select id as uid, label, real_content as content from " + config.init_train_data_table
    init_data_df = spark.sql(sql).toPandas()

    train_data_df = pd.concat([new_data_df, init_data_df], ignore_index=True)

    return train_data_df


def _update_model(spark_config, spark_user, model_class, model_path):
    try:
        spark_manager = PySparkMgr(spark_config, "sentiment-update-Model-pyspark")
        spark, sc = spark_manager.start(spark_user)
        logger.info(spark)

        train_data_df = update_train_data(spark)
        if train_data_df is None:
            return
        model = model_class.build(train_data_df)

        RWFlocker.lock(RWFlocker.WRITE)
        model_class.save(model, model_path)
        RWFlocker.unlock()
    except Exception as e:
        logger.error(e)
    finally:
        RWFlocker.unlock()


def _get_calibrated_sentiments(spark_config, spark_user):
    try:
        spark_manager = PySparkMgr(spark_config, "sentiment-get-calibrated-data-pyspark")
        spark, sc = spark_manager.start(spark_user)
        logger.info(spark)

        sql = "select * from " + config.calibrated_table
        df = spark.sql(sql).toPandas()

        RWFlocker.lock(RWFlocker.WRITE)
        df.to_csv(config.calibrated_sentiment_label_csv_path, sep=',', header=True, index=False)
        RWFlocker.unlock()
    except Exception as e:
        logger.error(e)
    finally:
        RWFlocker.unlock()


def _get_calibrate_data():
    calibrated_data = {}
    if os.path.exists(os.path.join(config.calibrate_tmp_folder, "calibrated_data")):
        with open(os.path.join(config.calibrate_tmp_folder, "calibrated_data"), 'r') as f:
            calibrated_data = json.loads(f.read())

    predicted_data = {}
    if os.path.exists(os.path.join(config.calibrate_tmp_folder, "predicted_data")):
        with open(os.path.join(config.calibrate_tmp_folder, "predicted_data"), 'r') as f:
            predicted_data = json.loads(f.read())

    logger.info("calibrated_data len is {}; predicted_data len is {}".format(len(calibrated_data), len(predicted_data)))

    return calibrated_data, predicted_data


def _calibrate_real_estate_company_sentiment(start_idx, end_idx):

    try:
        data_path = config.topic_local_real_estate_data_path
        if not os.path.exists(data_path):
            raise Exception("{} does not exist".format(data_path))

        indices = [x for x in range(start_idx, end_idx + 1)]
        text_df = pd.read_csv(data_path, sep=',')
        text_df = text_df.loc[indices, :]

        calibrated_data, predicted_data = _get_calibrate_data()

        target_df = pd.DataFrame([], columns=["crawler_id", "sentiment_id", "company_type",
                                              "products", "keywords", "label", "pub_date", "datetime"])

        date = datetime.datetime.now().strftime('%Y-%m-%d')
        target_idx = 0

        for idx in text_df.index:
            crawler_id = text_df.loc[idx, "crawler_id"]
            if pd.isnull(crawler_id):
                continue

            content = text_df.loc[idx, "content"]
            if pd.isnull(content) or len(content.strip()) == 0:
                continue

            products = text_df.loc[idx, "products"]
            if pd.isnull(products):
                continue

            sentiment_id = text_df.loc[idx, "sentiment_id"]
            if pd.isnull(sentiment_id):
                sentiment_id = None

            products = re.split("[,，]", products)
            company_names = [product for product in products if product.strip() != "开发商"]
            if len(company_names) == 0:
                related = True
            else:
                related = check_related_company_v2(content, company_names[0])

            label = ExternalLabel.NONSENSE.value
            if related:
                has_risk, keywords = has_company_risk_keywords(content)

                if has_risk:
                    logger.info("{} has risk keywords: {}".format(crawler_id, keywords))
                    label = ExternalLabel.NEGATIVE.value
                else:
                    if sentiment_id is not None:
                        if sentiment_id in calibrated_data:
                            label = calibrated_data[sentiment_id]
                        elif sentiment_id in predicted_data:
                            label = predicted_data[sentiment_id]
                            if label == ExternalLabel.THIRD_PARTY_NEGATIVE.value:
                                label = ExternalLabel.NEGATIVE.value
                        else:
                            label = ExternalLabel.NEUTRAL.value
                    else:
                        label = ExternalLabel.NEUTRAL.value

            keywords = text_df.loc[idx, "keywords"]
            if pd.isnull(keywords):
                keywords = None

            pub_date = text_df.loc[idx, "pub_date"]
            if pd.isnull(pub_date):
                pub_date = date

            target_df.loc[target_idx] = {
                "crawler_id": crawler_id,
                "sentiment_id": sentiment_id,
                "company_type": config.CompanyType.REAL_ESTATE.value,
                "keywords": keywords,
                "products": company_names[0] if len(company_names) > 0 else "开发商",
                "label": label,
                "pub_date": pub_date,
                "datetime": date
            }

            target_idx += 1

        spark_manager = PySparkMgr(config.spark_config, "calibrated-real_estate-pyspark")
        spark, sc = spark_manager.start(config.spark_user)
        if spark is None:
            raise Exception("Fail to launch spark session")
        logger.info(spark)
        logger.info("Start to save dataframe for real estate from {} to {}".format(start_idx, end_idx))

        insert_dataframe_into_db(target_df, spark, config.calibrated_result_table)

    except Exception as e:
        logger.info("Calibrate real estate company error: {}".format(e))


def _calibrate_common_sentiment(company_type, start_idx, end_idx):

    data_path = config.topic_local_market_data_path
    if company_type == config.CompanyType.PUBLIC_RELATION.value:
        data_path = config.topic_local_public_relation_data_path

    try:
        if not os.path.exists(data_path):
            raise Exception("{} does not exist".format(data_path))

        indices = [x for x in range(start_idx, end_idx + 1)]
        text_df = pd.read_csv(data_path, sep=',')
        text_df = text_df.loc[indices, :]

        calibrated_data, predicted_data = _get_calibrate_data()

        target_df = pd.DataFrame([], columns=["crawler_id", "sentiment_id", "company_type",
                                              "products", "keywords", "label", "pub_date", "datetime"])

        date = datetime.datetime.now().strftime('%Y-%m-%d')
        target_idx = 0

        for idx in text_df.index:
            crawler_id = text_df.loc[idx, "crawler_id"]
            if pd.isnull(crawler_id) or crawler_id is None:
                continue

            sentiment_id = text_df.loc[idx, "sentiment_id"]
            if not pd.isnull(sentiment_id):
                if sentiment_id in calibrated_data:
                    label = calibrated_data[sentiment_id]
                elif sentiment_id in predicted_data:
                    label = predicted_data[sentiment_id]
                else:
                    label = ExternalLabel.NEUTRAL.value
            else:
                label = ExternalLabel.NEUTRAL.value
                sentiment_id = None

            keywords = text_df.loc[idx, "keywords"]
            if pd.isnull(keywords):
                keywords = None

            products = text_df.loc[idx, "products"]
            if pd.isnull(products):
                products = None

            pub_date = text_df.loc[idx, "pub_date"]
            if pd.isnull(pub_date):
                pub_date = date

            target_df.loc[target_idx] = {
                "crawler_id": crawler_id,
                "sentiment_id": sentiment_id,
                "company_type": company_type,
                "keywords": keywords,
                "products": products,
                "label": label,
                "pub_date": pub_date,
                "datetime": date
            }

            target_idx += 1

        spark_manager = PySparkMgr(config.spark_config, "calibrated-" + company_type + "-pyspark")
        spark, sc = spark_manager.start(config.spark_user)
        if spark is None:
            raise Exception("Fail to launch spark session")
        logger.info(spark)
        logger.info("Start to save dataframe for {} from {} to {}".format(company_type, start_idx, end_idx))

        insert_dataframe_into_db(target_df, spark, config.calibrated_result_table)
    except Exception as e:
        logger.info("Calibrate {} error: {}".format(company_type, e))


def _calibrate_text_sentiment(company_type):

    try:
        data_path = config.topic_local_real_estate_data_path
        if company_type == config.CompanyType.MARKET:
            data_path = config.topic_local_market_data_path
        elif company_type == config.CompanyType.PUBLIC_RELATION:
            data_path = config.topic_local_public_relation_data_path

        if not os.path.exists(data_path):
            raise Exception("{} does not exist".format(data_path))

        text_df = pd.read_csv(data_path, sep=',')
        logger.info("text df shape is {}".format(text_df.shape))
        data_cnt = text_df.shape[0]
        del text_df

        if data_cnt <= 0:
            logger.warn("{} is empty".format(data_path))

        max_processes = config.calibrated_max_processes
        if company_type != config.CompanyType.REAL_ESTATE:
            max_processes = 2

        if data_cnt < max_processes:
            concurrent_number = 1
            step_data_range = data_cnt
        else:
            concurrent_number = max_processes
            step_data_range = int(data_cnt / max_processes)

        all_processes = []
        end_idx = -1

        for i in range(concurrent_number):
            start_idx = end_idx + 1
            end_idx = start_idx + step_data_range - 1

            if i == concurrent_number - 1:
                end_idx = data_cnt - 1

            logger.info("Process[{}]: start={}, range={}".format(i, start_idx, end_idx))

            if company_type == config.CompanyType.REAL_ESTATE:
                p = Process(target=_calibrate_real_estate_company_sentiment,
                            args=(start_idx, end_idx,))
            elif company_type in [config.CompanyType.MARKET, config.CompanyType.PUBLIC_RELATION]:
                p = Process(target=_calibrate_common_sentiment,
                            args=(company_type.value, start_idx, end_idx,))
            p.start()
            all_processes.append(p)

        for p in all_processes:
            p.join()

    except Exception as e:
        logger.error("Calibrating text sentiment of {} has error: {}".format(company_type.value, e))


def calibrate_text_sentiment():

    try:
        if not os.path.exists(config.topic_data_tmp_path):
            raise Exception("No data for calibration")

        if not os.path.exists(config.calibrate_tmp_folder):
            os.makedirs(config.calibrate_tmp_folder)

        db = Database(config.tidb_connection_config)
        logger.info("Tidb handler is {}".format(db))

        RWFlocker.lock(RWFlocker.WRITE)

        db.delete_all(config.calibrated_result_standby_table)
        db.backup(config.calibrated_result_standby_table, config.calibrated_result_table)
        db.delete_all(config.calibrated_result_table)

        for company_type in config.CompanyType:
            set_calibrate_status(company_type.value, config.CalibrateStatus.ONGOING.value)

        RWFlocker.unlock()

        end_date = datetime.datetime.now()
        start_date = end_date + datetime.timedelta(days=-config.TopicRequestConfig.day_range - 2)
        start_date = start_date.strftime("%Y-%m-%d")
        end_date = end_date.strftime("%Y-%m-%d")

        range_str = "datetime >= '" + start_date + "' and datetime <= '" + end_date + "'"
        logger.info("range str is {}".format(range_str))
        calibrated_data = db.select_all(config.calibrated_table, range_str, "uid, label")
        if len(calibrated_data) > 0:
            calibrated_data = {data['uid']: data['label'] for data in calibrated_data}
            logger.info("calibrated data length is {}".format(len(calibrated_data)))

            with open(os.path.join(config.calibrate_tmp_folder, "calibrated_data"), 'w') as f:
                f.write(json.dumps(calibrated_data))

        predicted_data = db.select_all(config.predict_table, range_str, "uid, label")
        if len(predicted_data) > 0:
            predicted_data = {data['uid']: data['label'] for data in predicted_data}
            logger.info("predicted data length is {}".format(len(predicted_data)))

            with open(os.path.join(config.calibrate_tmp_folder, "predicted_data"), 'w') as f:
                f.write(json.dumps(predicted_data))

        db.close()

        all_processes = []
        for company_type in config.CompanyType:
            if company_type not in [config.CompanyType.MARKET,
                                    config.CompanyType.PUBLIC_RELATION,
                                    config.CompanyType.REAL_ESTATE]:
                continue

            p = Process(target=_calibrate_text_sentiment, args=(company_type, ))
            p.start()
            all_processes.append(p)

        for p in all_processes:
            p.join()

        RWFlocker.lock(RWFlocker.WRITE)
        for company_type in config.CompanyType:
            set_calibrate_status(company_type.value, config.CalibrateStatus.UPDATED.value)

        set_last_calibrate_day(datetime.datetime.now().strftime('%Y-%m-%d'))
        RWFlocker.unlock()
    except Exception as e:
        logger.error("Calibrating text sentiment has error: {}".format(e))
    finally:
        db.close()
        RWFlocker.unlock()


def _execute():
    logger.info("start update model daemon")

    while True:
        try:
            message = _redis.pop_data(config.update_message_key, True)
            logger.info(message)

            if 'type' not in message:
                raise Exception("Data miss type field")

            if message['type'] not in [config.MessageType.UPDATE_MODEL.value, config.MessageType.CALIBRATED.value]:
                raise Exception("Unrecognized type field")

            if message['type'] == config.MessageType.UPDATE_MODEL.value:
                model_class = _model_factory.get(config.preferred_model)
                logger.info("model class is {}".format(model_class[0]))
                if model_class is None:
                    raise Exception("{} does not exist".format(config.preferred_model))

                p = Process(target=_update_model,
                            args=(config.spark_config, config.spark_user, model_class[0], model_class[1], ))
                p.start()
                p.join()
            elif message['type'] == config.MessageType.CALIBRATED.value:
                calibrate_text_sentiment()

        except Exception as e:
            logger.error(e)


if __name__ == "__main__":
    _execute()
