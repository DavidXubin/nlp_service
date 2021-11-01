import datetime
from tornado.log import app_log
import config
from sentiment import celery_app
from redis_db import RedisDBWrapper
from rw_flock import RWFlocker
from database import Database
from sentiment_update_service import get_calibrate_status, get_last_calibrate_day

_redis = RedisDBWrapper()


@celery_app.task
def update():
    app_log.info("update model")

    try:
        message = {
            "type": config.MessageType.UPDATE_MODEL.value,
            "datetime": datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        }

        if _redis.push_data(message, config.update_message_key) < 0:
            app_log.error("Fail to push data to redis queue: {}".format(config.update_message_key))

    except Exception as e:
        app_log.error(e)


@celery_app.task
def calibrate():
    app_log.info("calibrate sentiment data")

    try:
        message = {
            "type": config.MessageType.CALIBRATED.value,
            "datetime": datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        }

        if _redis.push_data(message, config.update_message_key) < 0:
            app_log.error("Fail to push data to redis queue: {}".format(config.update_message_key))

    except Exception as e:
        app_log.error(e)


@celery_app.task
def fetch_calibrated_sentiment_results(time_range):

    app_log.info("Fetch calibrated sentiment results")

    try:
        resp_header = {"status": "OK", "message": ""}

        if time_range not in [config.GetResultTimeRange.ONE_WEEK.value, config.GetResultTimeRange.HALF_MONTH.value,
                              config.GetResultTimeRange.ONE_MONTH.value]:
            raise Exception("Fetching calibrated sentiment time range is incorrect")

        end_date = get_last_calibrate_day()
        start_date = datetime.datetime.strptime(end_date, '%Y-%m-%d') + \
                     datetime.timedelta(days=-config.get_result_time_range[time_range])
        start_date = start_date.strftime("%Y-%m-%d")

        range_str = "pub_date >= '" + start_date + "' and pub_date <= '" + end_date + "'"
        app_log.info("Fetch calibrated data：{}".format(range_str))

        db = Database(config.tidb_connection_config)

        RWFlocker.lock(RWFlocker.READ)

        calibrate_status = []
        for company_type in config.CompanyType:
            if company_type not in [config.CompanyType.MARKET,
                                    config.CompanyType.PUBLIC_RELATION,
                                    config.CompanyType.REAL_ESTATE]:
                continue
            calibrate_status.append(get_calibrate_status(company_type.value))

        if config.CalibrateStatus.ONGOING.value in calibrate_status:
            table = config.calibrated_result_standby_table
        elif config.CalibrateStatus.UPDATED.value in calibrate_status:
            table = config.calibrated_result_table
        else:
            raise Exception("Fetching calibrated failed: {}".format(calibrate_status))

        calibrated_data = db.select_all(table, range_str, "crawler_id, label, pub_date")

        RWFlocker.unlock()

        results = {"result": []}
        for data in calibrated_data:
            results["result"].append(
                {
                    "id": data['crawler_id'],
                    "text_polarity": str(data["label"]),
                    "date_time": data['pub_date'].strftime("%Y-%m-%d")
                }
            )

        return {**resp_header, **results}

    except Exception as e:
        app_log.error(e)

        resp_header["status"] = "ERROR"
        resp_header["message"] = str(e)
        results = {"result": []}

        return {**resp_header, **results}
    finally:
        db.close()
        RWFlocker.unlock()
