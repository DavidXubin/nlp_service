import json
import datetime
from functools import partial
import tornado.httpclient
import tornado.web
from tornado.options import define, options
from tornado.log import app_log
from config import predict_table_key as predict_table_key, MessageType, GetResultTimeRange, RiskRelation
from redis_db import RedisDBWrapper
from sentiment.predict import predict, extract_region, extract_company_risks, extract_text_summary
from sentiment.update import fetch_calibrated_sentiment_results
from topic_analysis.run import fetch_results, fetch_batch_results, fetch_dedup_results

define("port", default=20001, help="run on the given port", type=int)


class BaseHandler(tornado.web.RequestHandler):

    def set_default_headers(self):
        app_log.info("setting headers!!!")
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with, content-type")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')

    def options(self):
        # no body
        self.set_status(200)
        self.finish("ok")

    def wait_for_result(self, task, callback=None):
        if task.ready():
            callback(task.result)
        else:
            tornado.ioloop.IOLoop.current().add_callback(
                partial(self.wait_for_result, task, callback)
            )


class PredictHandler(BaseHandler):

    @tornado.gen.coroutine
    def post(self):
        try:
            contents = keywords = products = None

            if "content" in self.request.arguments:
                contents = self.get_argument('content')

            if "keywords" in self.request.arguments:
                keywords = self.get_argument('keywords')

            if "products" in self.request.arguments:
                products = self.get_argument('products')

            if contents is None:
                post_data = self.request.body
                if type(post_data) == bytes:
                    post_data = post_data.decode('utf-8')

                params = json.loads(post_data)
                app_log.info(params)

                contents = params.get('content')
                keywords = params.get('keywords')
                products = params.get('products')

            if contents is None or len(contents) == 0:
                self.set_status(400, "Empty text content")
                self.finish(json.dumps(
                        {
                            'id': '-1',
                            'text_polarity': '-100',
                            'region': '其他'
                        }
                    )
                )
                return

            task = predict.apply_async(args=[contents, keywords, products])
            result = yield tornado.gen.Task(self.wait_for_result, task)

            self.set_status(200)
            self.finish(json.dumps(
                    {
                        'id': result[0],
                        'text_polarity': str(result[1]),
                        'region': str(result[2])
                    }
                )
            )

        except Exception as e:
            app_log.error(e)
            self.set_status(501)
            self.finish(json.dumps(
                    {
                        'id': '-1',
                        'text_polarity': '-100',
                        'region': '其他'
                    }
                )
            )


_redis = RedisDBWrapper()


class UpdateHandler(BaseHandler):

    @tornado.gen.coroutine
    def post(self):

        try:
            post_data = self.request.body
            if type(post_data) == bytes:
                post_data = post_data.decode('utf-8')

            post_data = post_data.replace("'", "\"")
            post_data = json.loads(post_data)
            app_log.info(post_data)

            for data in post_data:
                app_log.info("calibrate {}".format(data))

                if 'id' not in data:
                    raise Exception("Miss 'id' field in data")
                if 'text_polarity' not in data:
                    raise Exception("Miss 'text_polarity' field in data")

                data['type'] = MessageType.CALIBRATED.value
                data['datetime'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

                app_log.info(data)
                if _redis.push_data(data, predict_table_key) < 0:
                    raise Exception("Fail to push calibrated data({}) to redis queue: {}".format(data, predict_table_key))

            self.set_status(200)
            self.finish(json.dumps(
                    {
                        'status': 'OK',
                        'message': ""
                    }
                )
            )

        except Exception as e:
            app_log.error(e)
            self.set_status(501)
            self.finish(json.dumps(
                    {
                        'status': 'ERROR',
                        'message': str(e)
                    }
                )
            )


class GetTopicsHandler(BaseHandler):

    @tornado.gen.coroutine
    def get(self):

        try:
            company = None
            time_range = None

            if 'company' in self.request.arguments:
                company = self.get_argument('company')

            if 'time_range' in self.request.arguments:
                time_range = self.get_argument('time_range')

            if company is None:
                post_data = self.request.body
                if type(post_data) == bytes:
                    post_data = post_data.decode('utf-8')

                params = json.loads(post_data)
                app_log.info(params)

                company = params.get('company')
                time_range = params.get('time_range')

            if time_range is None:
                time_range = GetResultTimeRange.ONE_WEEK.value

            task = fetch_results.apply_async(args=[company, time_range])
            result = yield tornado.gen.Task(self.wait_for_result, task)
            self.set_status(200)
            self.finish(json.dumps(result))

        except Exception as e:
            app_log.error(e)
            self.set_status(501)
            self.finish(json.dumps(
                    {
                        'status': 'ERROR',
                        'message': str(e),
                        'result': []
                    }
                )
            )


class GetBatchTopicsHandler(BaseHandler):

    @tornado.gen.coroutine
    def get(self):

        try:
            post_data = self.request.body
            if type(post_data) == bytes:
                post_data = post_data.decode('utf-8')

            params = json.loads(post_data)
            app_log.info(params)

            companies = params.get('companies')
            time_range = params.get('time_range')
            batch_mode = params.get('batch_mode')

            if time_range is None:
                time_range = GetResultTimeRange.ONE_WEEK.value

            if batch_mode is None:
                batch_mode = "business"

            task = fetch_batch_results.apply_async(args=[companies, time_range, batch_mode, ])
            result = yield tornado.gen.Task(self.wait_for_result, task)
            self.set_status(200)
            self.finish(json.dumps(result))

        except Exception as e:
            app_log.error(e)
            self.set_status(501)
            self.finish(json.dumps(
                    {
                        'status': 'ERROR',
                        'message': str(e),
                        'result': []
                    }
                )
            )


class GetCalibratedSentimentHandler(BaseHandler):

    @tornado.gen.coroutine
    def get(self):

        try:
            time_range = None

            if 'time_range' in self.request.arguments:
                time_range = self.get_argument('time_range')

            if time_range is None and self.request.body is not None and len(self.request.body.strip()) > 0:
                post_data = self.request.body
                if type(post_data) == bytes:
                    post_data = post_data.decode('utf-8')

                params = json.loads(post_data)
                app_log.info(params)

                time_range = params.get('time_range')

            if time_range is None:
                time_range = GetResultTimeRange.ONE_WEEK.value

            task = fetch_calibrated_sentiment_results.apply_async(args=[time_range, ])
            result = yield tornado.gen.Task(self.wait_for_result, task)
            self.set_status(200)
            self.finish(json.dumps(result))

        except Exception as e:
            app_log.error(e)
            self.set_status(501)
            self.finish(json.dumps(
                    {
                        'status': 'ERROR',
                        'message': str(e),
                        'result': []
                    }
                )
            )


class ExtractRegion(BaseHandler):

    @tornado.gen.coroutine
    def post(self):

        try:
            contents = None
            subject = None

            if 'content' in self.request.arguments:
                contents = self.get_argument('content')

            if 'subject' in self.request.arguments:
                subject = self.get_argument('subject')

            if contents is None and self.request.body is not None and len(self.request.body.strip()) > 0:
                post_data = self.request.body
                if type(post_data) == bytes:
                    post_data = post_data.decode('utf-8')

                params = json.loads(post_data)

                contents = params.get('content')
                subject = params.get('subject')

            if contents is None or len(contents.strip()) == 0:
                raise Exception("Content is empty")

            task = extract_region.apply_async(args=[contents, subject, ])
            result = yield tornado.gen.Task(self.wait_for_result, task)
            self.set_status(200)
            self.finish(json.dumps(
                {
                    'id': result[0],
                    'region': str(result[1]),
                    'is_related': result[2]
                }
            )
            )

        except Exception as e:
            app_log.error(e)
            self.set_status(501)
            self.finish(json.dumps(
                {
                    'id': '-1',
                    'region': '其他',
                    'is_related': False
                }
            )
            )


class ExtractCompanyRisks(BaseHandler):

    @tornado.gen.coroutine
    def post(self):

        try:
            contents = None
            subject = None

            if 'content' in self.request.arguments:
                contents = self.get_argument('content')

            if 'subject' in self.request.arguments:
                subject = self.get_argument('subject')

            if contents is None and self.request.body is not None and len(self.request.body.strip()) > 0:
                post_data = self.request.body
                if type(post_data) == bytes:
                    post_data = post_data.decode('utf-8')

                params = json.loads(post_data)

                contents = params.get('content')
                subject = params.get('subject')

            if contents is None or len(contents.strip()) == 0:
                raise Exception("Content is empty")

            task = extract_company_risks.apply_async(args=[contents, subject, ])
            result = yield tornado.gen.Task(self.wait_for_result, task)
            self.set_status(200)
            self.finish(json.dumps(
                {
                    'id': result[0],
                    'relationship': result[1],
                    'risks': result[2]
                }
            )
            )

        except Exception as e:
            app_log.error(e)
            self.set_status(501)
            self.finish(json.dumps(
                    {
                        'id': '-1',
                        'relationship': RiskRelation.UNRELATED.value,
                        'risks': []
                    }
                )
            )


class GenerateTextSummary(BaseHandler):

    @tornado.gen.coroutine
    def post(self):

        try:
            contents = None
            algorithm = None

            if 'content' in self.request.arguments:
                contents = self.get_argument('content')

            if 'algorithm' in self.request.arguments:
                algorithm = self.get_argument('algorithm')

            if contents is None and self.request.body is not None and len(self.request.body.strip()) > 0:
                post_data = self.request.body
                if type(post_data) == bytes:
                    post_data = post_data.decode('utf-8')

                params = json.loads(post_data)

                contents = params.get('content')
                algorithm = params.get('algorithm')

            if contents is None or len(contents.strip()) == 0:
                raise Exception("Content is empty")

            if algorithm is None or len(algorithm.strip()) == 0:
                algorithm = "mixed"

            task = extract_text_summary.apply_async(args=[contents, algorithm, ])
            result = yield tornado.gen.Task(self.wait_for_result, task)

            self.set_status(200)

            self.finish(str(
                    {
                        'id': result[0],
                        'status': result[1],
                        'summary': result[2]
                    }
                )
            )

        except Exception as e:
            app_log.error(e)
            self.set_status(501)

            self.finish(str(
                    {
                        'id': '-1',
                        'status': "error",
                        'summary': str(e)
                    }
                )
            )


class GetDuplicatedDocHandler(BaseHandler):

    @tornado.gen.coroutine
    def get(self):

        try:
            time_range = None

            if 'time_range' in self.request.arguments:
                time_range = self.get_argument('time_range')

            if time_range is None and self.request.body is not None and len(self.request.body.strip()) > 0:
                post_data = self.request.body
                if type(post_data) == bytes:
                    post_data = post_data.decode('utf-8')

                params = json.loads(post_data)
                app_log.info(params)

                time_range = params.get('time_range')

            if time_range is None:
                time_range = GetResultTimeRange.ONE_WEEK.value

            task = fetch_dedup_results.apply_async(args=[time_range, ])
            result = yield tornado.gen.Task(self.wait_for_result, task)
            self.set_status(200)
            self.finish(json.dumps(result))

        except Exception as e:
            app_log.error(e)
            self.set_status(501)
            self.finish(json.dumps(
                    {
                        'status': 'ERROR',
                        'message': str(e),
                        'result': []
                    }
                )
            )


if __name__ == "__main__":
    tornado.options.parse_command_line()

    app = tornado.web.Application(handlers=[
        (r"/api/text_sentiment/analysis", PredictHandler),
        (r"/api/text_sentiment/update", UpdateHandler),
        (r"/api/text_sentiment/calibrated", GetCalibratedSentimentHandler),
        (r"/api/text_sentiment/updated", GetCalibratedSentimentHandler),
        (r"/api/v1.0/topic_analysis", GetTopicsHandler),
        (r"/api/v1.0/batch/topic_analysis", GetBatchTopicsHandler),
        (r"/api/text_sentiment/analysis/region", ExtractRegion),
        (r"/api/text_sentiment/analysis/company_risks", ExtractCompanyRisks),
        (r"/api/text_summary", GenerateTextSummary),
        (r"/api/get_duplicated_doc", GetDuplicatedDocHandler)
        ]
    )

    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()

