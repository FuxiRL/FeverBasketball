# coding=utf-8
"""
The base agent class for single-agent training.
"""
from collections import deque
# from algorithm.config import *
from algorithm.method import *
from utils.logger import *


class Interface(object):
    def __init__(self, agent):
        self.agent = agent
        self.is_learn = True
        self.cache = deque()
        self.episode = []
        self.episode_r = 0
        self.last_result = None
        self.self_test_model_index = None

    def SendSample(self, s, a, r, s_, avas, done, no_train=False, **kwargs):
        self.action = self.process([s, a, r, s_, avas, done], no_train=no_train)

    def ReceiveAction(self, **kwargs):
        return int(self.action)

    def process(self, exp, no_train=False):
        self.is_learn = not self.agent.is_test
        # for training
        if exp[0] is not None and len(exp[0]) > 0:
            if self.is_learn and not no_train:
                self.agent.method.store(exp)
                self.episode.append(exp)
                self.episode_r += exp[2]
        # if done:
        if exp[-1]:
            if len(self.episode) == 0:
                return
            if self.is_learn and not no_train:
                self.agent.lock.acquire()
                self.agent.method.episode_done()
                self.agent.lock.release()
                self.agent.put_log([float(self.episode_r)])
            self.agent.method.write_summary_scalar('episode_r', self.episode_r, self.agent.method.episode_cnt)
            self.episode = []
            self.episode_r = 0
            return None
        else:  # get action
            action = self.agent.method.act(exp[-3], exp[-2], no_train)
            return action


class Agent(object):
    def __init__(self, method_name, state_dim, action_num, model_name, log_port, is_test):
        self.is_test = is_test
        # self.config = configs[method_name]
        self.method = methods[method_name](self, state_dim, action_num, model_name, is_test)
        self.lock = threading.Lock()
        self.interfaces = []
        # for logging
        self.logger = LogHandler(model_name, "127.0.0.1", log_port)
        self.write_queue = Queue()
        print("new agent state=%d action=%d test=%d name=%s log_port=%d test=%d" % (state_dim, action_num, is_test,
                                                                                    model_name, log_port, is_test))

    def add_interface(self, uuid=None):
        tmp_interface = Interface(self)
        self.interfaces.append(tmp_interface)
        return tmp_interface

    def start(self):
        if not self.is_test:
            # log thread
            self.log_thread = threading.Thread(target=self.write_log)
            self.log_thread.setDaemon(False)
            self.log_thread.start()

    def put_log(self, log):
        self.write_queue.put(log)

    def write_log(self):
        while True:
            log = self.write_queue.get()
            self.logger.push("episode_r", log[0])






