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

    def __init__(self, name):
        self.__read_switch = _LightSwitch()
        self.__write_switch = _LightSwitch()
        self.__no_readers = threading.Lock()
        self.__no_writers = threading.Lock()
        self.__readers_queue = threading.Lock()
        self.name = name

        riri = threading.Thread(target=self.print_switch)
        riri.start()

    def reader_acquire(self):
        # print "reader a"
        self.__readers_queue.acquire()
        self.__no_readers.acquire()
        self.__read_switch.acquire(self.__no_writers)
        self.__no_readers.release()
        self.__readers_queue.release()

    def reader_release(self):
        # print "reader r"
        self.__read_switch.release(self.__no_writers)

    def writer_acquire(self):
        print("writer a")
        self.__write_switch.acquire(self.__no_readers)
        self.__no_writers.acquire()

    def writer_release(self):
        print("writer r")
        self.__no_writers.release()
        self.__write_switch.release(self.__no_readers)


class _LightSwitch:

    def get_count(self):
        return self.__counter

    def __init__(self):
        self.__counter = 0
        self.__mutex = threading.Lock()

    def acquire(self, lock):
        self.__mutex.acquire()
        self.__counter += 1
        if self.__counter == 1:
            lock.acquire()
        self.__mutex.release()

    def release(self, lock):
        self.__mutex.acquire()
        self.__counter -= 1
        if self.__counter == 0:
            lock.release()
        self.__mutex.release()
