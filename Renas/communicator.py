from multiprocessing import Manager
# from multiprocessing import Queue, Value
# from multiprocessing.managers import BaseManager

# class QueueManager(BaseManager):
#     pass

class Communicator:
    def __init__(self, is_ps, ps_host):
        # There might be other ways to get the IP address
        server_addr = socket.gethostbyname(ps_host.split(":")[0])
        server_port = int(ps_host.split(":")[1])

        self.__is_ps = is_ps
        self.manager = Manager(address=(server_addr, server_port), authkey=b'abc')
        self.task = self.manager.Queue()
        self.result = self.manager.Queue()
        self.end_flag = self.manager.Queue()
        self.data_sync = self.manager.Queue()
        self.data_count = 0
        
        self.__start()
        return

    def __del__(self):
        self.manager.shutdown()
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