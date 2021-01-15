import threading
import time
import multiprocessing

print_lock = multiprocessing.Lock()

class RWLock:
    def print_switch(self):
        while True:
            with print_lock:
                print(self.name, self.__read_switch.get_count(), self.__write_switch.get_count())
            time.sleep(1)

    def __init__(self, name=None):
        self.read_count = 0
        self.write_count = 0
        self.rmutex = threading.Lock()
        self.wmutex = threading.Lock()
        self.read_try = threading.Lock()
        self.resource = threading.Lock()


    def reader_acquire(self):
        self.read_try.acquire()
        self.rmutex.acquire()
        self.read_count += 1
        if self.read_count == 1:
            self.resource.acquire()
        self.rmutex.release()
        self.read_try.release()

    def reader_release(self):
        self.rmutex.acquire()
        self.read_count -= 1
        if self.read_count == 0:
            self.resource.release()
        self.rmutex.release()

    def writer_acquire(self):
        self.wmutex.acquire()
        self.write_count += 1
        if self.write_count == 1:
            self.read_try.acquire()
        self.wmutex.release()
        self.resource.acquire()

    def writer_release(self):
        self.resource.release()
        self.wmutex.acquire()
        self.write_count -= 1
        if self.write_count == 0:
            self.read_try.release()
        self.wmutex.release()


