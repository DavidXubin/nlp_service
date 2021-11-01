import pymysql
import pymysql.cursors
from tornado.log import app_log


class Database:
    """ Python连接到 MySQL 数据库及相关操作 """
    """
    conf: 类参数，数据库的连接参数配置字典，含host、port、user、pw、db、charset(可选,默认utf8)
    connected: 属性，True数据库连接成功，False连接失败

    insert(self, table, val_obj): 方法，插入数据到数据表
        table: 数据表名称
        val_obj: 待插入数据的字段名和值的键值对字典
        返回: 成功则返回新插入数据的主键ID，失败返回False

    update(self, table, val_obj, range_str): 方法，更新数据表中的数据
        table: 数据表名称
        val_obj: 待更新数据的字段名和值的键值对字典
        range_str: 更新范围的条件语句字符串
        返回: 成功返回更新的行数，失败返回False

    delete(self, table, range_str): 方法，在数据表中删除数据
        table: 数据表名称
        range_str: 删除范围的条件语句字符串
        返回: 成功返回删除的行数，失败返回False

    select_one(self, table, factor_str, field='*'): 方法，查询表中符合条件唯一的一条数据
        table: 数据表名称
        factor_str: 查询唯一条件语句字符串
        field: 查询结果返回哪些字段，多个用逗号分隔，可选参数，默认返回所有字段
        返回: 成功返回一条数据的字段名与值的一维字典，失败返回False

    select_more(self, table, range_str, field='*'): 方法，查询表中符合条件的所有数据
        table: 数据表名称
        range_str: 查询条件语句字符串
        field: 查询结果返回哪些字段，多个用逗号分隔，可选参数，默认返回所有字段
        返回: 成功返回多条数据的字段名与值的二维字典，失败返回False

    count(self, table, range_str='1'): 方法，统计数据表中符合条件的总函数
        table: 数据表名称
        range_str: 查询条件语句字符串，可选参数，默认表中所有行数
        返回: 成功返回符合条件的行数，失败返回False

    sum(self, table, field, range_str='1'): 方法，对数据表中某数值类型字段求和
        table: 数据表名称
        field: 需要求和的字段，可以是多个字段的计算公式
        range_str: 需要求和的条件语句字符串，可选参数，默认表中所有行
        返回: 成功返回求和结果，失败返回False

    close(self): 方法，关闭数据库连接，对象销毁时也会自动关闭，所以多数时候不用特意调用
    """
    def __init__(self, conf):
        self.connected = False
        self.__conn = None

        if type(conf) is not dict:
            app_log.error('Database connection config should be of dict type!')
        else:
            for key in ['host', 'port', 'user', 'password', 'db']:
                if key not in conf.keys():
                    app_log.error("Database connection config lacks {}".format(key))
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
            app_log.error("Fail to connect to db: {}".format(e))

    # 插入数据到数据表
    def insert(self, table, val_obj):
        sql_top = 'INSERT INTO ' + table + ' ('
        sql_tail = ') VALUES ('
        try:
            for key, val in val_obj.items():
                sql_top += key + ','
                sql_tail += val + ','
            sql = sql_top[:-1] + sql_tail[:-1] + ')'
            with self.__conn.cursor() as cursor:
                cursor.execute(sql)
            self.__conn.commit()
            return self.__conn.insert_id()
        except pymysql.Error as e:
            self.__conn.rollback()
            app_log.error("Fail to insert into db: {}".format(e))
            return None

    # 更新数据到数据表
    def update(self, table, val_obj, range_str):
        sql = 'UPDATE ' + table + ' SET '
        try:
            for key, val in val_obj.items():
                sql += key + '=' + val + ','
            sql = sql[:-1] + ' WHERE ' + range_str
            with self.__conn.cursor() as cursor:
                cursor.execute(sql)
            self.__conn.commit()
            return cursor.rowcount
        except pymysql.Error as e:
            app_log.error("Fail to update on db: {}".format(e))
            self.__conn.rollback()
            return None

    # 删除数据在数据表中
    def delete(self, table, range_str):
        sql = 'DELETE FROM ' + table + ' WHERE ' + range_str
        try:
            with self.__conn.cursor() as cursor:
                cursor.execute(sql)
            self.__conn.commit()
            return cursor.rowcount
        except pymysql.Error as e:
            app_log.error("Fail to delete from db: {}".format(e))
            self.__conn.rollback()
            return None

    # 删除数据在数据表中
    def delete_all(self, table):
        sql = 'truncate table ' + table
        try:
            with self.__conn.cursor() as cursor:
                cursor.execute(sql)
            self.__conn.commit()
            return cursor.rowcount
        except pymysql.Error as e:
            app_log.error("Fail to delete all from db: {}".format(e))
            self.__conn.rollback()
            return None

    # 查询唯一数据在数据表中
    def select_one(self, table, factor_str, field='*'):
        sql = 'SELECT ' + field + ' FROM ' + table + ' WHERE ' + factor_str
        try:
            with self.__conn.cursor() as cursor:
                cursor.execute(sql)
            self.__conn.commit()
            return cursor.fetchall()[0]
        except pymysql.Error as e:
            app_log.error("Fail to select one from db: {}".format(e))
            return None

    # 查询多条数据在数据表中
    def select_all(self, table, range_str, field='*'):
        sql = 'SELECT ' + field + ' FROM ' + table + ' WHERE ' + range_str
        try:
            with self.__conn.cursor() as cursor:
                cursor.execute(sql)
            self.__conn.commit()
            return cursor.fetchall()
        except pymysql.Error as e:
            app_log.error("Fail to select all from db: {}".format(e))
            return None

    # 统计某表某条件下的总行数
    def count(self, table, range_str='1'):
        sql = 'SELECT count(*)res FROM ' + table + ' WHERE ' + range_str
        try:
            with self.__conn.cursor() as cursor:
                cursor.execute(sql)
            self.__conn.commit()
            return cursor.fetchall()[0]['res']
        except pymysql.Error as e:
            app_log.error("Fail to count on db: {}".format(e))
            return None

    # 统计某字段（或字段计算公式）的合计值
    def sum(self, table, field, range_str='1'):
        sql = 'SELECT SUM(' + field + ') AS res FROM ' + table + ' WHERE ' + range_str
        try:
            with self.__conn.cursor() as cursor:
                cursor.execute(sql)
            self.__conn.commit()
            return cursor.fetchall()[0]['res']
        except pymysql.Error as e:
            app_log.error("Fail to sum on db: {}".format(e))
            return None

    def backup(self, target_table, source_table):
        sql = 'INSERT INTO ' + target_table + ' SELECT * FROM ' + source_table
        try:
            with self.__conn.cursor() as cursor:
                cursor.execute(sql)
            self.__conn.commit()
            return True
        except pymysql.Error as e:
            self.__conn.rollback()
            app_log.error("Fail to insert into {} from {}: e".format(target_table, source_table, e))
            return False

    # 关闭数据库连接
    def close(self):
        try:
            if self.connected:
                self.__conn.close()

            self.connected = False
            self.__conn = None
        except pymysql.Error as e:
            app_log.error("Fail to close on db: {}".format(e))

    # 销毁对象时关闭数据库连接
    def __del__(self):
        self.close()
