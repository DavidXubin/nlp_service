import os
import py4j
import time
import random
import logging
import findspark


class PySparkMgr():
    def __init__(self, args, appName=None):
        self.__sc = None
        self.__appName = appName
        self.__args = args

    def start(self, user, use_udf=False):
        epoch_time = int(time.mktime(time.localtime()))
        random_num = random.randint(0, 10000)
        if self.__appName is None:
            self.__appName = user + "-pyspark-" + str(epoch_time) + "-" + str(random_num)
        else:
            self.__appName = self.__appName + "-" + str(epoch_time) + "-" + str(random_num)

        value = self.__args.get('spark.home')
        if not value:
            value = "/data/software/spark-2.3.1-bin-hadoop2.7"

        findspark.init(spark_home=value, extra_env_file="/etc/dw-env.sh", python_path="python3")

        value = self.__args.get('spark.driver.memory')
        if not value:
            value = "3g"
        findspark.set_driver_mem(value)

        value = self.__args.get('spark.executor.memory')
        if not value:
            value = "3g"
        findspark.set_executor_mem(value)

        value = self.__args.get('spark.py.files')
        if value is not None:
            findspark.add_python_zips(value)

        findspark.set_app_name(self.__appName)
        findspark.end()

        import pyspark
        from pyspark import SparkConf
        from pyspark.sql import SparkSession

        from pyspark.context import SparkContext
        if os.environ.get("SPARK_EXECUTOR_URI"):
            SparkContext.setSystemProperty("spark.executor.uri", os.environ["SPARK_EXECUTOR_URI"])

        SparkContext._ensure_initialized()
        pySpark = None

        sc_conf = SparkConf()
        sc_conf.set('spark.locality.wait', 30000)
        sc_conf.set('spark.sql.autoBroadcastJoinThreshold', -1)
        sc_conf.set('spark.scheduler.minRegisteredResourcesRatio', 1)

        value = self.__args.get('spark.driver.maxResultSize')
        if not value:
            value = "3g"
        sc_conf.set('spark.driver.maxResultSize', value)

        value = self.__args.get('spark.rpc.message.maxSize')
        if value is not None:
            sc_conf.set('spark.rpc.message.maxSize', value)

        value = self.__args.get('spark.executor.cores')
        if not value:
            value = '1'
        executor_cores = int(value)

        sc_conf.set('spark.executor.cores', executor_cores)

        value = self.__args.get('spark.executor.instances')
        if not value:
            value = '4'
        executor_instances = int(value)

        sc_conf.set('spark.cores.max', executor_cores * executor_instances)

        if use_udf:
            sc_conf.set("spark.sql.execution.arrow.enabled", True)

        try:
            # Try to access HiveConf, it will raise exception if Hive is not added
            SparkContext._jvm.org.apache.hadoop.hive.conf.HiveConf()
            spark = SparkSession.builder.enableHiveSupport().config(conf=sc_conf).getOrCreate()

            from py4j.java_gateway import java_import
            from pyspark.context import SparkContext
            gw = SparkContext._gateway
            java_import(gw.jvm, "org.apache.spark.sql.TiSparkSession")
            pySpark = gw.jvm.TiSparkSession.builder().getOrCreate()
        except py4j.protocol.Py4JError:
            spark = SparkSession.builder.config(conf=sc_conf).getOrCreate()
        except TypeError:
            spark = SparkSession.builder.config(conf=sc_conf).getOrCreate()

        sc = spark.sparkContext
        sc.setJobGroup("", "Started By : {}".format(user), False)

        return pySpark if pySpark else spark, sc
