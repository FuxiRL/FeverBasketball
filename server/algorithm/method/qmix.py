# coding=utf-8
import tensorflow as tf
import numpy as np
import datetime
import time
import os
import random
import threading
import queue


class Method(object):
    """ For Qmix """
    def __init__(self, agent, num_global_s, num_s, num_a, name, test, lr=0.0001, gamma=0.99, replace_target_iter=2000,
                 memory_size=2000000, batch_size=256, epsilon=1, epsilon_decay=0.0001):
        self.agent = agent
        self.name = name
        self.num_global_s = num_global_s
        self.num_s = num_s
        self.num_a = num_a
        self.lr = lr
        self.gamma = gamma
        self.replace_target_iter = replace_target_iter
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
        self.load_latest_checkpoints = True
        self.exp_splicer = {}
        self.exp_splicing_lock = threading.RLock()
        self.update_queue = queue.Queue()
        self._build_net()
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + '/target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + '/eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()
        # if not self.test:
        self.train_writer = tf.summary.FileWriter("./data/logs/" + self.name + '/' + 'dqn_lr_' + str(self.lr) + '_' +
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

    def _build_net(self):  # we use parameter sharing among agents
        with tf.variable_scope(self.name):
            # ------------------ all inputs ------------------------
            self.S = tf.placeholder(tf.float32, [None, self.num_global_s], name='S')  # input Global State
            self.s = tf.placeholder(tf.float32, [None, self.num_s], name='s1')  # input state for agent1
            # self.s2 = tf.placeholder(tf.float32, [None, self.num_s], name='s2')  # input state for agent2
            # self.s3 = tf.placeholder(tf.float32, [None, self.num_s], name='s3')  # input state for agent3
            self.S_ = tf.placeholder(tf.float32, [None, self.num_global_s], name='S_')  # input Next Global State
            self.s_ = tf.placeholder(tf.float32, [None, self.num_s], name='s1_')  # input next state for agent1
            # self.s2_ = tf.placeholder(tf.float32, [None, self.num_s], name='s2_')  # input next state for agent2
            # self.s3_ = tf.placeholder(tf.float32, [None, self.num_s], name='s3_')  # input next state for agent3
            self.R = tf.placeholder(tf.float32, [None, ], name='R')  # input Reward
            self.a1 = tf.placeholder(tf.int32, [None, ], name='a1')  # input Action for agent1
            self.a2 = tf.placeholder(tf.int32, [None, ], name='a2')  # input Action for agent2
            self.a3 = tf.placeholder(tf.int32, [None, ], name='a3')  # input Action for agent3
            self.done = tf.placeholder(tf.float32, [None, ], name='done')  # input Done info ???

            self.q1_a1 = tf.placeholder(tf.float32, [None, ], name='q1_eval_value_wrt_a1')
            self.q2_a2 = tf.placeholder(tf.float32, [None, ], name='q2_eval_value_wrt_a2')
            self.q3_a3 = tf.placeholder(tf.float32, [None, ], name='q3_eval_value_wrt_a3')
            self.q1_m_ = tf.placeholder(tf.float32, [None, ], name='q1_value_next_max')
            self.q2_m_ = tf.placeholder(tf.float32, [None, ], name='q2_value_next_max')
            self.q3_m_ = tf.placeholder(tf.float32, [None, ], name='q3_value_next_max')

            w_initializer, b_initializer = tf.random_normal_initializer(0., 0.1), tf.constant_initializer(0.0)

            # ------------------ build evaluate_net ------------------
            with tf.variable_scope('eval_net'):
                a_fc1 = tf.layers.dense(self.s, 128, tf.nn.relu, kernel_initializer=w_initializer,
                                        bias_initializer=b_initializer, name='agent_fc1_e')
                a_fc2 = tf.layers.dense(a_fc1, 128, tf.nn.relu, kernel_initializer=w_initializer,
                                        bias_initializer=b_initializer, name='agent_fc2_e')
                a_fc3 = tf.layers.dense(a_fc2, 64, tf.nn.relu, kernel_initializer=w_initializer,
                                        bias_initializer=b_initializer, name='agent_fc3_e')
                self.q_eval = tf.layers.dense(a_fc3, self.num_a, kernel_initializer=w_initializer,
                                              bias_initializer=b_initializer, name='q_e')

            # ------------------ build target_net ------------------
            with tf.variable_scope('target_net'):
                a_fc1_ = tf.layers.dense(self.s_, 128, tf.nn.relu, kernel_initializer=w_initializer,
                                         bias_initializer=b_initializer, name='agent_fc1_t')
                a_fc2_ = tf.layers.dense(a_fc1_, 128, tf.nn.relu, kernel_initializer=w_initializer,
                                         bias_initializer=b_initializer, name='agent_fc2_t')
                a_fc3_ = tf.layers.dense(a_fc2_, 64, tf.nn.relu, kernel_initializer=w_initializer,
                                         bias_initializer=b_initializer, name='agent_fc3_t')
                self.q_next = tf.layers.dense(a_fc3_, self.num_a, kernel_initializer=w_initializer,
                                              bias_initializer=b_initializer, name='q_t')

            # ------------------ build mixing_net ------------------
            with tf.variable_scope('mixing_net'):
                # a1_indices = tf.stack([tf.range(tf.shape(self.a1)[0], dtype=tf.int32), self.a1], axis=1, name='a1_indices')
                # a2_indices = tf.stack([tf.range(tf.shape(self.a2)[0], dtype=tf.int32), self.a2], axis=1, name='a2_indices')
                # a3_indices = tf.stack([tf.range(tf.shape(self.a3)[0], dtype=tf.int32), self.a3], axis=1, name='a3_indices')
                # q1_a = tf.gather_nd(params=self.q1_eval, indices=a1_indices, name='q1_eval_wrt_a')
                # q2_a = tf.gather_nd(params=self.q2_eval, indices=a2_indices, name='q2_eval_wrt_a')
                # q3_a = tf.gather_nd(params=self.q3_eval, indices=a3_indices, name='q3_eval_wrt_a')
                self.q_concat = tf.stack([self.q1_a1, self.q2_a2, self.q3_a3], axis=1, name='q_concat')
                self.q_concat_ = tf.stack([self.q1_m_, self.q2_m_, self.q3_m_], axis=1, name='q_concat_next')  # todo

                # with tf.variable_scope('eval_net'):
                with tf.variable_scope('hyper_layer1'):
                    non_abs_w1 = tf.layers.dense(inputs=self.S, units=3*32, kernel_initializer=w_initializer,
                                                 bias_initializer=b_initializer, name='non_abs_w1')
                    self.w1 = tf.reshape(tf.abs(non_abs_w1), shape=[-1, 3, 32], name='w1')
                    self.b1 = tf.layers.dense(inputs=self.S, units=32, kernel_initializer=w_initializer,
                                              bias_initializer=b_initializer, name='non_abs_b1')
                with tf.variable_scope('hyper_layer2'):
                    non_abs_w2 = tf.layers.dense(inputs=self.S, units=32 * 1, kernel_initializer=w_initializer,
                                                 bias_initializer=b_initializer, name='non_abs_w2')
                    self.w2 = tf.reshape(tf.abs(non_abs_w2), shape=[-1, 32, 1], name='w2')
                    bef_b2 = tf.layers.dense(inputs=self.S, units=32, activation=tf.nn.relu,
                                             kernel_initializer=w_initializer, bias_initializer=b_initializer, name='bef_b2')
                    self.b2 = tf.layers.dense(inputs=bef_b2, units=1, kernel_initializer=w_initializer,
                                              bias_initializer=b_initializer, name='non_abs_b2')

                with tf.variable_scope('layer_mix_eval'):
                    lin1 = tf.matmul(tf.reshape(self.q_concat, shape=[-1, 1, 3]), self.w1) + tf.reshape(self.b1, shape=[-1, 1, 32])
                    a1 = tf.nn.elu(lin1, name='a1')
                    self.Q_tot = tf.reshape(tf.matmul(a1, self.w2), shape=[-1, 1]) + self.b2

                with tf.variable_scope('layer_mix_target'):
                    lin1_ = tf.matmul(tf.reshape(self.q_concat_, shape=[-1, 1, 3]), self.w1) + tf.reshape(self.b1, shape=[-1, 1, 32])
                    a1_ = tf.nn.elu(lin1_, name='a1_')
                    self.Q_tot_ = tf.reshape(tf.matmul(a1_, self.w2), shape=[-1, 1]) + self.b2

            # todo: add q_target, loss, train_op
            with tf.variable_scope('q_target'):
                q_target = self.R + (1 - self.done) * self.gamma * self.Q_tot_
                self.q_target = tf.stop_gradient(q_target)
            with tf.variable_scope('loss'):
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.Q_tot, name='TD_error'))
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
        # we need to do exp splicing here to make exp of different game clients and interfaces together
        self.exp_splicing_lock.acquire()

        if uuid not in self.exp_splicer:
            self.exp_splicer[uuid] = [None, None, None]
        self.exp_splicer[uuid][id] = exp
        # print(self.exp_splicer)
        if None not in self.exp_splicer[uuid]:

            # time to splice the exps
            S1, s1, a1, r1, S1_, s1_, avas1, done1 = self.exp_splicer[uuid][0]
            S2, s2, a2, r2, S2_, s2_, avas2, done2 = self.exp_splicer[uuid][1]
            S3, s3, a3, r3, S3_, s3_, avas3, done3 = self.exp_splicer[uuid][2]
            R = r1 + r2 + r3
            done = all([done1, done2, done3])

            # reconstruct the experience
            EXP1 = [S1, s1, s2, s3, a1, a2, a3, R, S1_, s1_, s2_, s3_, avas1, avas2, avas3, done]
            EXP2 = [S2, s1, s2, s3, a1, a2, a3, R, S2_, s1_, s2_, s3_, avas1, avas2, avas3, done]
            EXP3 = [S3, s1, s2, s3, a1, a2, a3, R, S3_, s1_, s2_, s3_, avas1, avas2, avas3, done]
            EXPs = [EXP1, EXP2, EXP3]

            for EXP in EXPs:
                # if None not in exp:
                self.memory_counter += 1
                if len(self.memory) > self.memory_size:
                    # random replacement
                    index = np.random.randint(0, self.memory_size)
                    self.memory[index] = EXP
                else:
                    self.memory.append(EXP)
            self.exp_splicer[uuid] = [None, None, None]

            # check if it is ok to update
            if self.memory_counter % 5 == 0 and len(self.memory) > self.batch_size:
                self.update_queue.put(1)

        self.exp_splicing_lock.release()

    # the updating thread
    def start_learning_thread(self):
        if not self.test:
            print('start {} learning thread...'.format(self.name))
            learning_thread = threading.Thread(target=self.learn, name=self.name+'_learning_thread')
            learning_thread.setDaemon(True)
            learning_thread.start()
        else:
            print('testing...')

    def learn(self):
        while True:
            if self.update_queue.get():
                self.update()

    def update(self):
        # sample batch exp from memory
        if self.learn_step_cnt % 10000 == 0:
            print(self.name, 'update ----> learn_step_cnt', self.learn_step_cnt)
        batch_exp = random.sample(self.memory, self.batch_size)
        S, s1, s2, s3, a1, a2, a3, R, S_, s1_, s2_, s3_, avas1, avas2, avas3, done = [[] for _ in range(16)]
        for exp in batch_exp:
            S.append(exp[0])
            s1.append(exp[1])
            s2.append(exp[2])
            s3.append(exp[3])
            a1.append(exp[4])
            a2.append(exp[5])
            a3.append(exp[6])
            R.append(exp[7])
            S_.append(exp[8])
            s1_.append(exp[9])
            s2_.append(exp[10])
            s3_.append(exp[11])
            avas1.append(exp[12])
            avas2.append(exp[13])
            avas3.append(exp[14])
            done.append(exp[15])
        # to get q_tot
        q1_eval = self.sess.run(self.q_eval, feed_dict={self.s: s1})
        q2_eval = self.sess.run(self.q_eval, feed_dict={self.s: s2})
        q3_eval = self.sess.run(self.q_eval, feed_dict={self.s: s3})
        row_index = [_ for _ in range(self.batch_size)]
        q1_a1 = q1_eval[row_index, a1]
        q2_a2 = q2_eval[row_index, a2]
        q3_a3 = q3_eval[row_index, a3]

        # to get q_tot_
        q1_ = self.sess.run(self.q_next, feed_dict={self.s_: s1_})
        q1_[np.array(avas1)[:, :] == 0] = - 999999  # mask unavailable actions
        q2_ = self.sess.run(self.q_next, feed_dict={self.s_: s2_})
        q2_[np.array(avas2)[:, :] == 0] = - 999999  # mask unavailable actions
        q3_ = self.sess.run(self.q_next, feed_dict={self.s_: s3_})
        q3_[np.array(avas3)[:, :] == 0] = - 999999  # mask unavailable actions
        q1_m_ = np.max(q1_, axis=1)
        q2_m_ = np.max(q2_, axis=1)
        q3_m_ = np.max(q3_, axis=1)
        q_tot_ = self.sess.run(self.Q_tot_,
                               feed_dict={self.S: S_, self.q1_m_: q1_m_, self.q2_m_: q2_m_, self.q3_m_: q3_m_})

        # update
        _, cost = self.sess.run([self._train_op, self.loss],
                                feed_dict={self.S: S, self.q1_a1: q1_a1, self.q2_a2: q2_a2, self.q3_a3: q3_a3,
                                           self.a1: a1, self.a2: a2, self.a3: a3,
                                           self.R: R, self.Q_tot_: q_tot_, self.done: done})
        # print('cost', cost)

        self.write_summary_scalar('loss', cost, self.learn_step_cnt)
        self.write_summary_scalar('epsilon', self.epsilon, self.learn_step_cnt)
        self.write_summary_scalar('memory_cnt', self.memory_counter, self.learn_step_cnt)
        self.save()  # save model
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)  # decay epsilon
        self.learn_step_cnt += 1

        # check to do the soft replacement of target net
        if self.learn_step_cnt % self.replace_target_iter == 0 and self.learn_step_cnt:
            self.sess.run(self.target_replace_op)

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





