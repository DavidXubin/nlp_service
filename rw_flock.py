import os
import fcntl
from tornado.log import app_log


_write_lock_filepath = os.path.join(os.getcwd(), "locks/write_lock")


class RWFlocker(object):

    __w_lockfd = None

    READ = 0
    WRITE = 1

    @staticmethod
    def acquire_write_lock(mode):
        while True:
            RWFlocker.__w_lockfd = open(_write_lock_filepath, 'r')

            try:
                fcntl.flock(RWFlocker.__w_lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)

                file = open(_write_lock_filepath + ".owner", 'w')
                file.write(str(mode))
                file.close()
                break
            except Exception as e:
                app_log.warn(e)
                RWFlocker.__w_lockfd.close()
                RWFlocker.__w_lockfd = None

            if RWFlocker.__w_lockfd is None and os.path.exists(_write_lock_filepath + ".owner"):
                try:
                    file = open(_write_lock_filepath + ".owner", 'r')
                    owner = file.read().strip()
                    if mode == RWFlocker.READ and owner == str(RWFlocker.READ):
                        break
                except Exception as e:
                    app_log.warn(e)
                finally:
                    file.close()

    @staticmethod
    def lock(mode):
        if mode == RWFlocker.READ or mode == RWFlocker.WRITE:
            RWFlocker.acquire_write_lock(mode)
        else:
            app_log.error("unrecognized lock type")

    @staticmethod
    def unlock():
        if RWFlocker.__w_lockfd:
            if os.path.exists(_write_lock_filepath + ".owner"):
                os.unlink(_write_lock_filepath + ".owner")

            fcntl.flock(RWFlocker.__w_lockfd, fcntl.LOCK_UN)
            RWFlocker.__w_lockfd.close()
            RWFlocker.__w_lockfd = None

