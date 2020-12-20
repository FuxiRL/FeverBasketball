# coding=utf-8
"""
The base agent class for multi-agent training.
"""

from collections import deque
# from algorithm.config import *
from algorithm.method import *
from utils.logger import *


class Interface(object):
    """
    Interface for MA-agent, taking the whole team as a unit.
    """
    def __init__(self, agent, uuid, style):
        self.agent = agent
        self.style = style
        self.is_learn = True
        self.cache = deque()
        self.episode = []
        self.episode_r = 0
        self.last_result = None
        self.self_test_model_index = None
        self.exp_tot = [None, None, None]
        self.uuid = uuid

    def SendSample(self, id, scene, S, s, a, r, S_, s_, avas, done, no_train=False, **kwargs):
        if self.style == 'centralized':
            self.action = self.process(id, scene, [S, s, a, r, S_, s_, avas, done], no_train=no_train)
        else:
            assert self.style == 'decentralized'
            self.action = self.process(id, scene, [s, a, r, s_, avas, done], no_train=no_train)

    def ReceiveAction(self, **kwargs):
        return int(self.action)

    def process(self, id, scene, exp, no_train=False):
        self.is_learn = not self.agent.is_test

        # for training
        if exp[0] is not None and len(exp[0]) > 0:
            if self.is_learn and not no_train:
                self.agent.method.store(self.uuid, id, scene, exp)  # to distinguish the exps
                self.episode.append(exp)
                _r = exp[3] if self.style == 'centralized' else exp[2]
                self.episode_r += _r

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
            action = self.agent.method.act(exp[-3], exp[-2], id, no_train)  # state, available actions
            return action


class MAagent(object):
    def __init__(self, method_name, state_dim_global, state_dim, action_num, model_name, log_port, is_test):
        self.is_test = is_test
        self.style = 'centralized' if method_name in ['vdn', 'qmix'] else 'decentralized'
        # self.config = configs[method_name]
        if self.style == 'centralized':
            self.method = methods[method_name](self, state_dim_global, state_dim, action_num, model_name, is_test)
        else:
            assert self.style == 'decentralized'
            self.method = methods[method_name](self, state_dim, action_num, model_name, is_test)
        self.lock = threading.Lock()
        self.interfaces = []
        # for logging
        self.logger = LogHandler(model_name, "127.0.0.1", log_port)
        self.write_queue = Queue()
        self.model_name = model_name
        print("new ma_agent: method={} model_name={} global_state={} local_state={} action={} log_port={} test={}".
              format(method_name, model_name, state_dim_global, state_dim, action_num, log_port, is_test))

    def add_interface(self, uuid):  # game client uuid and player id
        tmp_interface = Interface(self, uuid, self.style)
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






