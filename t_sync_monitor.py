import sched
import time
import sys
import pymysql
import pymysql.cursors

master_db_conf = {
    "host": "",
    "user": "",
    "password": "",
    "db": "",
    "port": 3306, #you can modify it
    "charset": "utf8"
}


slave_db_conf = {
    "host": "",
    "user": "",
    "password": "",
    "db": "",
    "port": 3306, #you can modify it
    "charset": "utf8"
}


tidb_conf = {
    "host": "tidb.bkjk.cn",
    "user": "",
    "password": "",
    "db": "",
    "port": 4000,
    "charset": "utf8"
}


class Database:

    def __init__(self, conf):
        self.connected = False
        self.__conn = None

        if type(conf) is not dict:
            print('Database connection config should be of dict type!')
        else:
            for key in ['host', 'port', 'user', 'password', 'db']:
                if key not in conf.keys():
                    print("Database connection config lacks {}".format(key))
            if 'charset' not in conf.keys():
                conf['charset'] = 'utf8'
        try:
            self.__conn = pymysql.connect(
                host=conf['host'],
                port=conf['port'],
                user=conf['user'],
                passwd=conf['password'],
                db=conf['db'],
                charset=conf['charset'],
                cursorclass=pymysql.cursors.DictCursor)

            self.connected = True
        except pymysql.Error as e:
            print("Fail to connect to db: {}".format(e))


    def insert(self, table, values):
        sql_top = 'INSERT INTO ' + table + ' ('
        sql_tail = ') VALUES ('
        try:
            for key, val in values.items():
                sql_top += key + ','

                if isinstance(val, int):
                    val = str(val)
                elif isinstance(val, str):
                    val = "'" + val + "'"

                sql_tail += val + ','
            sql = sql_top[:-1] + sql_tail[:-1] + ')'

            with self.__conn.cursor() as cursor:
                cursor.execute(sql)
            self.__conn.commit()
            return self.__conn.insert_id()
        except pymysql.Error as e:
            self.__conn.rollback()
            print("Fail to insert into db: {}".format(e))
            return None

    def selectMaxTimestamp(self, table, field):
        sql = "SELECT max({}) from {}".format(field, table)
        try:
            with self.__conn.cursor() as cursor:
                cursor.execute(sql)
            self.__conn.commit()
            return list(cursor.fetchall()[0].values())[0]
        except pymysql.Error as e:
            print("Fail to select one from db: {}".format(e))
            return None

    def getMysqlEpochTime(self):
        try:
            with self.__conn.cursor() as cursor:
                cursor.execute("select REPLACE(unix_timestamp(current_timestamp(3)),'.','')")
            self.__conn.commit()
            return int(list(cursor.fetchall()[0].values())[0])
        except pymysql.Error as e:
            self.__conn.rollback()
            print("Fail to get mysql epoch time: {}".format(e))
            return None


    def close(self):
        try:
            if self.connected:
                self.__conn.close()

            self.connected = False
            self.__conn = None
        except pymysql.Error as e:
            print("Fail to close on db: {}".format(e))

    def __del__(self):
        self.close()


master_db = None

slave_db = None

ti_db = None

monitor_db = None

get_now_milli_time = lambda: int(time.time() * 1000)

my_scheduler = sched.scheduler(time.time, time.sleep)


def checkDBSyncTime(interval):
    global master_db, slave_db, tidb, monitor_db, tidb_conf

    dbName = tidb_conf["db"]

    maxMasterTs = master_db.selectMaxTimestamp("ts_monitor", "ts")
    if maxMasterTs is None:
        print("Fail to get max timestamp for master database[{}]".format(dbName))
        return

    maxSlaveTs = slave_db.selectMaxTimestamp("ts_monitor", "ts")
    if maxSlaveTs is None:
        print("Fail to get max timestamp for slave database[{}]".format(dbName))
        return

    maxTidbTs = tidb.selectMaxTimestamp("ts_monitor", "ts")
    if maxTidbTs is None:
        print("Fail to get max timestamp for tidb database[{}]".format(dbName))
        return

    monitorDBTs = monitor_db.getMysqlEpochTime()
    if monitorDBTs is None:
        print("Fail to get current tidb epoch time")
        return

    #主备库延迟毫秒数
    m2sDelay = maxSlaveTs - maxMasterTs

    #syncer延迟毫秒数
    s2tDelay = maxTidbTs - maxSlaveTs

    #主库到TiDB延迟毫秒数
    m2tDelay = maxTidbTs - maxMasterTs

    insertMonitorData = {
        "m_max_ts": maxMasterTs,
        "s_max_ts": maxSlaveTs,
        "t_max_ts": maxTidbTs,
        "m_delay_s": m2sDelay,
        "s_delay_t": s2tDelay,
        "m_delay_t": m2tDelay,
        "db": dbName,
        "py_ts": get_now_milli_time(),
        "db_ts": monitorDBTs
    }

    if monitor_db.insert("t_syncer_monitor", insertMonitorData) is None:
        print("Failed to insert sync monitor data into db_syncer_delay.t_syncer_monitor")
    else:
        my_scheduler.enter(interval, 0, checkDBSyncTime, (interval,))


def scheduleMonitor(interval=60):
    global my_scheduler

    my_scheduler.enter(0, 0, checkDBSyncTime, (interval,))
    my_scheduler.run()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Please input the database name to monitor")
        exit()

    dbName = sys.argv[1].strip()
    print("Start to monitor database: {}".format(dbName))

    master_db_conf["db"] = dbName
    slave_db_conf["db"] = dbName
    tidb_conf["db"] = dbName
    monitor_db_conf = tidb_conf.copy()
    monitor_db_conf["db"] = "db_syncer_delay"

    master_db = Database(master_db_conf)
    slave_db = Database(slave_db_conf)
    tidb = Database(tidb_conf)
    monitor_db = Database(monitor_db_conf)

    scheduleMonitor(2)

    if master_db is not None:
        del master_db

    if slave_db is not None:
        del slave_db

    if tidb is not None:
        del tidb

    if monitor_db is not None:
        del monitor_db