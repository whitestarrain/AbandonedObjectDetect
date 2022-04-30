import threading

from app.entry.deploy_host_conf import DeployHostConf
from app.service.dao_utils import DaoUtils


class DeployHostConfDao(object):
    @staticmethod
    def select_all_valid_conf():
        session = DaoUtils.get_session()
        host = session.query(DeployHostConf).filter_by(status=1).all()
        session.close()
        return host

    @staticmethod
    def select_all():
        session = DaoUtils.get_session()
        hosts = session.query(DeployHostConf).all()
        session.close()
        return hosts


class DeployHostConfService(object):

    def __init__(self, interval=10):
        super(DeployHostConfService, self).__init__()
        self.scheduler_counter = 0
        self.interval = interval
        self.host_conf_list = DeployHostConfDao.select_all_valid_conf()
        self.update_thread = threading.Thread(target=self.update_host_conf_list, args=())

    def start_update(self):
        self.update_thread.start()

    def update_host_conf_list(self):
        while True:
            if self.update_thread is None:
                return
            self.host_conf_list = DeployHostConfDao.select_all_valid_conf()
            time.sleep(self.interval)

    def get_host_conf(self) -> DeployHostConf:
        self.scheduler_counter += 1
        if self.scheduler_counter > len(self.host_conf_list) - 1:
            self.scheduler_counter = 0

        return self.host_conf_list[self.scheduler_counter]

    def stop_update(self):
        self.update_thread = None


# 开启一个更新线程


if __name__ == '__main__':
    import time

    service = DeployHostConfService(interval=2)
    service.start_update()
    for i in range(10):
        print(service.get_host_conf().port)
    service.stop_update()
