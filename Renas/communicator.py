import socket

from multiprocessing import Manager
from multiprocessing.managers import SyncManager

from info_str import NAS_CONFIG

class Communicator:
    def __init__(self):
        self._setting = NAS_CONFIG['cmnct']
        is_ps = self._setting['is_ps']
        ps_host = self._setting['ps_host']
        # There might be other ways to get the IP address
        server_addr = socket.gethostbyname(ps_host.split(":")[0])
        server_port = int(ps_host.split(":")[1])

        self.__is_ps = is_ps
        self.manager = SyncManager(address=(), authkey=b'abc')
        self.__start()

        self.task = self.manager.Queue()
        self.result = self.manager.Queue()
        self.end_flag = self.manager.Queue()
        self.data_sync = self.manager.Queue()
        self.idle_gpuq = Manager().Queue()
        self.data_count = 0

        return

    def __start(self):
        if self.__is_ps:
            self.manager.start()
        else:
            while True:
                try:
                    self.manager.connect()
                    break
                except:
                    print("waiting for connecting ...")
        return