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
    """ VDN algorithm with parameter sharing among teammates. """
    def __init__(self, agent, num_global_s, num_s, num_a, name, test, lr=0.001, gamma=0.99, replace_target_iter=2000,
                 memory_size=2000000, batch_size=256, epsilon=0.1, epsilon_decay=0.0001):
        self.agent = agent
        self.agent_num = 3  # todo: need to be put into config for covering 2v2
        self.name = name
        self.num_global_s = num_global_s
        self.num_s = num_s
        self.num_a = [num_a, num_a, num_a]  # todo: also need to cover different positions
        self.lr = lr
        self.gamma = gamma
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.low_level_concat = False
        self.high_level_concat = False
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.test = test
        self.epsilon_min = 0.1
        self.tau = 0.01  # soft replacement
        self.learn_step_cnt = 0  # total learning step
        self.episode_cnt = 0
        self.memory = []
        self.memory_counter = 0
        self.load_latest_checkpoints = True
        self.exp_splicer = {}
        self.exp_splicing_lock = threading.RLock()
        self.update_queue = queue.Queue()
        self.sess = tf.Session()
        self.graph = tf.Graph()
        # with self.sess.as_default():
        #     with self.graph.as_default():
        self._build_net()
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + '/target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + '/eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]


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
            checkpoint = tf.train.latest_checkpoint(path)
            self.saver.restore(self.sess, checkpoint)
            print('##### testing mode: {} | model:{} #####'.format(self.name, checkpoint))

        # start learning thread
        # self.start_learning_thread()

    def _build_net(self):  # we use parameter sharing among agents
        # print('build_net for', self.name)
        # with tf.variable_scope(self.name):
        # ------------------ all inputs placeholders ------------------------
        self.s1 = tf.placeholder(tf.float32, [None, self.num_s], name='s1')  # input state for agent1
        self.s2 = tf.placeholder(tf.float32, [None, self.num_s], name='s2')  # input state for agent2
        self.s3 = tf.placeholder(tf.float32, [None, self.num_s], name='s3')  # input state for agent3
        self.S = [self.s1, self.s2, self.s3]
        self.s1_ = tf.placeholder(tf.float32, [None, self.num_s], name='s1_')  # input next state for agent1
        self.s2_ = tf.placeholder(tf.float32, [None, self.num_s], name='s2_')  # input next state for agent2
        self.s3_ = tf.placeholder(tf.float32, [None, self.num_s], name='s3_')  # input next state for agent3
        self.S_ = [self.s1_, self.s2_, self.s3_]
        self.R = tf.placeholder(tf.float32, [None, ], name='R')  # input Reward
        self.done = tf.placeholder(tf.float32, [None, ], name='done')  # input Done info
        self.a = [None] * self.agent_num
        for i in range(self.agent_num):
            self.a[i] = tf.placeholder(tf.int32, [None, ], name='a_'+str(i))
        self.max_target_q = tf.placeholder(tf.float32, [None, ], name='max_target_sum_q')

        # construct subQ net for each agent
        self.sub_q_nets = [None] * self.agent_num
        self.first_layer_ops = [None] * self.agent_num  # ops: operation
        self.max_action_ops = [None] * self.agent_num
        if self.low_level_concat or self.high_level_concat:
            self.first_layer_phds = [None] * self.agent_num
            for i in range(self.agent_num):
                self.first_layer_phds[i] = tf.placeholder(tf.float32, [None, 128], name='flp_'+str(i))
        else:
            self.first_layer_phds = None
        for i in range(self.agent_num):
            self.sub_q_nets[i], self.first_layer_ops[i] = self.build_sub_q_net(i, self.S[i], self.low_level_concat,
                                                                               self.high_level_concat,
                                                                               self.first_layer_phds)
            # self.max_action_ops[i] = tf.argmax(self.sub_q_nets[i], axis=1)

        # construct sumQ net
        self.sum_q_net_input_phds = [None] * self.agent_num
        self.sum_q_net_prod = [None] * self.agent_num
        for i in range(self.agent_num):
            self.sum_q_net_input_phds[i] = tf.placeholder(tf.float32, [None, self.num_a[i]], name='sum_q_'+str(i))
            self.sum_q_net_prod[i] = tf.multiply(self.sum_q_net_input_phds[i], tf.one_hot(self.a[i], self.num_a[i]))
        self.sum_q_net = tf.reduce_sum(tf.concat(self.sum_q_net_prod, axis=1), axis=1, keep_dims=True)

        # construct target subQ net for each agent
        sub_q_params = [None] * self.agent_num
        for i in range(self.agent_num):
            sub_q_params[i] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'_SubQ_Net_' + str(i))
        ema = tf.train.ExponentialMovingAverage(decay=1-self.tau)

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        sub_q_target_update = [None] * self.agent_num
        for i in range(self.agent_num):
            sub_q_target_update[i] = ema.apply(sub_q_params[i])

        self.target_sub_q_nets = [None] * self.agent_num
        self.target_first_layer_ops = [None] * self.agent_num
        if self.low_level_concat or self.high_level_concat:
            self.target_first_layer_phds = [None] * self.agent_num
            for i in range(self.agent_num):
                self.target_first_layer_phds[i] = tf.placeholder(tf.float32, [None, 128], name="target_flp_" + str(i))
        else:
            self.target_first_layer_phds = None

        for i in range(self.agent_num):
            self.target_sub_q_nets[i], self.target_first_layer_ops[i] = \
                self.build_sub_q_net(i, self.S_[i], self.low_level_concat,
                                     self.high_level_concat, self.target_first_layer_phds,
                                     reuse=True, custom_getter=ema_getter)

        # begin to define loss and gradient
        with tf.control_dependencies(sub_q_target_update):
            self.q_target = self.R + self.gamma * (1 - self.done) * self.max_target_q
            self.q_target = tf.stop_gradient(self.q_target)

            self.squared_td_error = tf.square(self.sum_q_net - self.q_target)
            self.mse_loss = tf.reduce_mean(self.squared_td_error, name='mse_loss')

            # define train process for each of the agents
            self.op_grad_on_subQ = []
            self.gradient_from_sum_q_phds = [None] * self.agent_num
            self.optimizers = [None] * self.agent_num
            self.train_ops = [None] * self.agent_num
            for i in range(self.agent_num):
                self.op_grad_on_subQ.append(tf.squeeze(tf.gradients(self.mse_loss, self.sum_q_net_input_phds[i]),
                                                       axis=[0]))
                self.gradient_from_sum_q_phds[i] = tf.placeholder(tf.float32, shape=[None, self.num_a[i]],
                                                                  name='gradients_from_sum_q_'+str(i))
                self.optimizers[i] = tf.train.AdamOptimizer(learning_rate=self.lr)
                trainable_var_i = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'_SubQ_Net_'+str(i))
                parameter_gradients = tf.gradients(self.sub_q_nets[i], trainable_var_i, self.gradient_from_sum_q_phds[i])
                self.train_ops[i] = self.optimizers[i].apply_gradients(zip(parameter_gradients, trainable_var_i))

    def build_sub_q_net(self, agent_index, state_phd, low_level_concat=False, high_level_concat=False,
                        first_layer_phds=None, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope(self.name + '_SubQ_Net_' + str(agent_index), reuse=reuse, custom_getter=custom_getter):
            # the first hidden layer
            net = tf.layers.dense(state_phd, 128, tf.nn.relu, name="l1", trainable=trainable)
            first_layer_output = net
            # low level concat
            if low_level_concat:
                for i in range(self.agent_num):
                    if i == agent_index:
                        continue
                    else:
                        net = tf.concat([net, first_layer_phds[i]], axis=1)
            # the second hidden layer
            net = tf.layers.dense(net, 128, tf.nn.relu, name="l2", trainable=trainable)
            # high level concat
            if high_level_concat:
                for i in range(self.agent_num):
                    if i == agent_index:
                        continue
                    else:
                        net = tf.concat([net, first_layer_phds[i]], axis=1)
            # the third hidden layer
            net = tf.layers.dense(net, 64, tf.nn.relu, name="l3", trainable=trainable)
            # the output layer
            # note that the output is a vector which contains Q-values of all actions in one state
            q_sa = tf.layers.dense(net, self.num_a[agent_index], activation=tf.identity, name='qs', trainable=trainable)

            return q_sa, first_layer_output

    def act(self, state, avas, id, no_train):  # epsilon greedy
        if np.random.uniform() > self.epsilon or self.test or no_train:  # pick the argmax action
            s = np.array(state)
            first_layer_outputs = None
            if len(s.shape) < 2:
                s = np.array(state)[np.newaxis, :]
            feed_dict = {}
            feed_dict.update({self.S[id]: s})
            if self.low_level_concat or self.high_level_concat:
                if first_layer_outputs is None:
                    first_layer_outputs = [None] * self.agent_num
                    for j in range(self.agent_num):
                        first_layer_outputs[j] = self.sess.run(self.first_layer_ops[j], feed_dict=feed_dict)
                        feed_dict.update({self.first_layer_ops[j]: first_layer_outputs[j]})

            q_eval = self.sess.run(self.sub_q_nets[id], feed_dict=feed_dict)[0]
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
                self.update()
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
            # S.append(exp[0])
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
        S = [s1, s2, s3]
        A = [a1, a2, a3]
        S_ = [s1_, s2_, s3_]
        AVAS = [avas1, avas2, avas3]
        feed_dict = {}
        for i in range(self.agent_num):
            feed_dict.update({self.S[i]: S[i]})
            feed_dict.update({self.S_[i]: S_[i]})
        feed_dict.update({self.R: R})
        feed_dict.update({self.done: done})

        # the concat feature
        if self.low_level_concat or self.high_level_concat:
            for i in range(self.agent_num):
                first_layer_i = self.sess.run(self.first_layer_ops[i], feed_dict=feed_dict)
                target_first_layer_i = self.sess.run(self.target_first_layer_ops[i], feed_dict=feed_dict)
                feed_dict.update({self.first_layer_phds[i]: first_layer_i})
                feed_dict.update({self.target_first_layer_phds[i]: target_first_layer_i})
        # get target sum q by combining action mask
        max_target_q = 0
        for i in range(self.agent_num):
            target_sub_q = self.sess.run(self.target_sub_q_nets[i], feed_dict=feed_dict)
            target_sub_q[np.array(AVAS[i])[:, :] == 0] = - 999999
            # q1_[np.array(avas1)[:, :] == 0] = - 999999  # mask unavailable actions
            max_target_q += np.max(target_sub_q, axis=1)
        feed_dict.update({self.max_target_q: max_target_q})

        # the gradient from sumQ
        for i in range(self.agent_num):
            sub_q_i = self.sess.run(self.sub_q_nets[i], feed_dict=feed_dict)
            feed_dict.update({self.sum_q_net_input_phds[i]: sub_q_i})
            feed_dict.update({self.a[i]: A[i]})
        for i in range(self.agent_num):
            sum_q_gradient_i = self.sess.run(self.op_grad_on_subQ[i], feed_dict=feed_dict)
            feed_dict.update({self.gradient_from_sum_q_phds[i]: sum_q_gradient_i})

        # optimize each subQ net
        for i in range(self.agent_num):
            self.sess.run(self.train_ops[i], feed_dict=feed_dict)
        # get the cost
        cost = self.sess.run(self.mse_loss, feed_dict=feed_dict)

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

    def episode_done(self):
        self.episode_cnt += 1
        # self.config.eps = max(self.config.eps - self.config.eps_dec, self.config.eps_min)

    def write_summary_scalar(self, tag, value, iteration):
        self.train_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)]), iteration)





