import tensorflow as tf
import numpy as np
import datetime
import time
import os
import random
import threading
import queue


class Method(object):
    def __init__(self, agent, num_s, num_a, name, test, lr=0.0001, gamma=0.99, replace_target_iter=2000, memory_size=2000000,
                 batch_size=256, epsilon=0., epsilon_decay=0.0001):
        self.agent = agent
        self.name = name
        self.num_s = num_s
        self.num_a = num_a
        self.lr = lr if self.name != '33_ballclear' else lr/2
        self.beta = 1  # learning coefficient for negative td-error
        self.gamma = gamma
        self.replace_target_iter = replace_target_iter if self.name != '33_ballclear' else 5*replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.test = test
        self.epsilon_min = 0.1
        self.learn_step_cnt = 0  # total learning step
        self.episode_cnt = 0
        self.memory = []
        self.memory_counter = 0
        self.load_latest_checkpoints = False
        self.update_queue = queue.Queue()
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + '/target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + '/eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()
        # if not self.test:
        self.train_writer = tf.summary.FileWriter("./data/logs/" + self.name + '/' + 'hyq_lr_' + str(self.lr) + '_' +
                                                  datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'),
                                                  self.sess.graph)
        tf.summary.merge_all()
        self.cost_his = []
        self.saver = tf.train.Saver(max_to_keep=10000)

        if not self.test:
            self.sess.run(tf.global_variables_initializer())
            print('##### training mode: {} #####'.format(self.name))
        else:
            tf.reset_default_graph()
            path = './data/checkpoints/' + self.name + '/'
            if self.load_latest_checkpoints:
                checkpoint = tf.train.latest_checkpoint(path)
            else:
                model_list = {'33_attack': 360000, '33_defense': 450000, '33_freeball': 480000,
                              '33_ballclear': 320000, '33_assist': 390000}
                checkpoint = path + 'dqn.ckpt-' + str(model_list[self.name])
            self.saver.restore(self.sess, checkpoint)
            print('##### testing mode: {} | model:{} #####'.format(self.name, checkpoint))

        # start learning thread
        self.start_learning_thread()

    def _build_net(self):
        with tf.variable_scope(self.name):
            # ------------------ all inputs ------------------------
            self.s = tf.placeholder(tf.float32, [None, self.num_s], name='s')  # input State
            self.s_ = tf.placeholder(tf.float32, [None, self.num_s], name='s_')  # input Next State
            self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
            self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action
            self.done = tf.placeholder(tf.float32, [None, ], name='done')  # input Done info
            self.q_m_ = tf.placeholder(tf.float32, [None, ], name='q_value_next_max')
            w_initializer, b_initializer = tf.random_normal_initializer(0., 0.1), tf.constant_initializer(0.0)

            # ------------------ build evaluate_net ------------------
            with tf.variable_scope('eval_net'):
                e1 = tf.layers.dense(self.s, 128, tf.nn.relu, kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer, name='e1')
                e1 = tf.layers.dense(e1, 128, tf.nn.relu, kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer, name='e2')
                e1 = tf.layers.dense(e1, 64, tf.nn.relu, kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer, name='e3')
                self.q_eval = tf.layers.dense(e1, self.num_a, kernel_initializer=w_initializer,
                                              bias_initializer=b_initializer, name='q_e')

            # ------------------ build target_net ------------------
            with tf.variable_scope('target_net'):
                t1 = tf.layers.dense(self.s_, 128, tf.nn.relu, kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer, name='t1')
                t1 = tf.layers.dense(t1, 128, tf.nn.relu, kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer, name='t2')
                t1 = tf.layers.dense(t1, 64, tf.nn.relu, kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer, name='t3')
                self.q_next = tf.layers.dense(t1, self.num_a, kernel_initializer=w_initializer,
                                              bias_initializer=b_initializer, name='q_t')

            with tf.variable_scope('q_target'):
                q_target = self.r + (1 - self.done) * self.gamma * self.q_m_ #tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')
                self.q_target = tf.stop_gradient(q_target)
            with tf.variable_scope('q_eval'):
                a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
                self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)
            with tf.variable_scope('loss'):
                self.td_error = self.q_target - self.q_eval_wrt_a
                self.hyq_squared_error_op = tf.where(tf.less(self.td_error, 0),
                                                     self.beta * tf.square(self.td_error),
                                                     tf.square(self.td_error))
                self.loss = tf.reduce_mean(self.hyq_squared_error_op, name='loss')
                # self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
            with tf.variable_scope('train'):
                self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def act(self, state, avas, id, no_train):  # epsilon greedy
        if np.random.uniform() > self.epsilon or self.test or no_train:  # pick the argmax action
            s = np.array(state)
            if len(s.shape) < 2:
                s = np.array(state)[np.newaxis, :]
            q_eval = self.sess.run(self.q_eval, feed_dict={self.s: s})[0]
            q_eval[avas == 0] = -float('inf')  # unavailable actions should never be selected !
            action = np.argmax(q_eval)
        else:  # pick random action
            avail_action_dim = sum(avas)
            action = np.random.randint(0, avail_action_dim)

        return action

    def store(self, uuid, id, scene, exp):
        # if None not in exp:
        self.memory_counter += 1
        if len(self.memory) > self.memory_size:
            # random replacement
            index = np.random.randint(0, self.memory_size)
            self.memory[index] = exp
        else:
            self.memory.append(exp)

        # check if it is ok to update
        if self.memory_counter % 5 == 0 and len(self.memory) > self.batch_size:
            self.update_queue.put(1)

    # the updating thread
    def start_learning_thread(self):
        if not self.test:
            learning_thread = threading.Thread(target=self.learn, name=self.name+'_learning_thread')
            learning_thread.setDaemon(True)
            learning_thread.start()
            print('start {} learning thread...'.format(self.name))
        else:
            print('testing...')

    def update(self):
        if self.learn_step_cnt % 10000 == 0:
            print(self.name, 'update ----> learn_step_cnt', self.learn_step_cnt)
        batch_exp = random.sample(self.memory, self.batch_size)
        s, a, r, s_, avas, done = [[] for _ in range(6)]
        for exp in batch_exp:
            s.append(exp[0])
            a.append(exp[1])
            r.append(exp[2])
            s_.append(exp[3])
            avas.append(exp[4])
            done.append(exp[5])

        # update
        q_ = self.sess.run(self.q_next, feed_dict={self.s_: s_})
        q_[np.array(avas)[:, :] == 0] = - 999999  # mask unavailable actions
        q_m_ = np.max(q_, axis=1)
        _, cost = self.sess.run([self._train_op, self.loss],
                                feed_dict={self.s: s, self.a: a, self.r: r, self.s_: s_, self.done: done, self.q_m_:q_m_})
        self.write_summary_scalar('loss', cost, self.learn_step_cnt)
        self.write_summary_scalar('epsilon', self.epsilon, self.learn_step_cnt)
        self.write_summary_scalar('memory_cnt', self.memory_counter, self.learn_step_cnt)
        self.save()  # save model
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)  # decay epsilon
        self.learn_step_cnt += 1

        # check to do the soft replacement of target net
        if self.learn_step_cnt % self.replace_target_iter == 0 and self.learn_step_cnt:
            self.sess.run(self.target_replace_op)

    def learn(self):
        while True:
            if self.update_queue.get():
                self.update()

    def save(self):
        if self.learn_step_cnt % 10000 == 0 and self.learn_step_cnt > 0 and not self.test:
            path = './data/checkpoints/' + self.name + '/'
            if not os.path.exists(path): os.makedirs(path)
            model_name = os.path.join(path, 'dqn.ckpt')
            save_path = self.saver.save(self.sess, model_name, global_step=self.learn_step_cnt)
            # print('save model %s' % save_path)

    def episode_done(self):
        self.episode_cnt += 1
        # self.config.eps = max(self.config.eps - self.config.eps_dec, self.config.eps_min)

    def write_summary_scalar(self, tag, value, iteration):
        self.train_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)]), iteration)





