import config
from pyspark_manager import PySparkMgr
from redis_db import RedisDBWrapper
from utils import get_logger

logger = get_logger("sentiment-save-data-service")

_spark_config = {
    "spark.driver.memory": "2g",
    "spark.executor.memory": "2g",
    "spark.driver.maxResultSize": "2g",
    "spark.executor.cores": 1,
    "spark.executor.instances": 4
}

_spark_manager = PySparkMgr(_spark_config, "sentiment-save-data-pyspark")
_spark, _sc = _spark_manager.start(config.spark_user)

_redis = RedisDBWrapper()


def save_predict_data(data):

    try:
        logger.info("Predicted: {} : {}".format(data['uid'], data['label']))

        insert_sql = "insert into " + config.predict_table + " values"

        keywords = data['keywords']
        if keywords:
            keywords = str(keywords).lstrip('[').rstrip(']')
            keywords = keywords.replace("'", '')
            keywords = "'" + keywords + "'"
        else:
            keywords = "null"

        products = data['products']
        if products:
            products = str(products).lstrip('[').rstrip(']')
            products = products.replace("'", '')
            products = "'" + products + "'"
        else:
            products = "null"

        record = ["'" + data['uid'] + "'", keywords, products, '"""' + data['text'] + '"""', str(data['label']),
                  "'" + data['proba'] + "'", "'" + data['datetime'] + "'"]

        record = ','.join(record)
        insert_sql += "(" + record + ")"

        _spark.executeTidbSql(insert_sql)
    except Exception as e:
        logger.error(e)


def save_calibrate_data(data):
    try:
        logger.info("Calibrated: {} : {}".format(data['id'], data['text_polarity']))

        insert_sql = "insert into " + config.calibrated_table + " values"

        record = ["'" + data['id'] + "'", str(data['text_polarity']), "'" + data['datetime'] + "'"]

        record = ','.join(record)
        insert_sql += "(" + record + ")"

        _spark.executeTidbSql(insert_sql)
    except Exception as e:
        logger.error(e)


def _execute():

    while True:
        try:
            message = _redis.pop_data(config.predict_table_key, True)
            if 'type' not in message:
                raise Exception("Data miss type field")

            if message['type'] == config.MessageType.PREDICTED.value:
                save_predict_data(message)
            elif message['type'] == config.MessageType.CALIBRATED.value:
                save_calibrate_data(message)
            else:
                raise Exception("Unrecognized type field")

        except Exception as e:
            logger.error(e)


if __name__ == "__main__":
    _execute()

