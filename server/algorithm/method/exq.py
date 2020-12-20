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
                 batch_size=256, epsilon=0.1, epsilon_decay=0.0001, beta_max=1, beta_min=1e-4, beta_decay_episode=10000):
        self.agent = agent
        self.name = name
        self.num_s = num_s
        self.num_a = num_a
        self.lr = lr if self.name != '33_ballclear' else lr/2
        # we try to decay beta
        self.beta = beta_max  # beta_max = 1, learning coefficient for negative td-error
        self.beta_min = beta_min
        self.beta_decay_factor = pow(beta_min/beta_max, 1.0/beta_decay_episode)
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
        self._build_net()  # include both the opt net and the avg net

        t_params_opt = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+'_opt' + '/target_net')
        e_params_opt = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+'_opt' + '/eval_net')
        t_params_avg = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+'_avg' + '/target_net')
        e_params_avg = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+'_avg' + '/eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op_opt = [tf.assign(t, e) for t, e in zip(t_params_opt, e_params_opt)]
            self.target_replace_op_avg = [tf.assign(t, e) for t, e in zip(t_params_avg, e_params_avg)]

        self.sess = tf.Session()
        # if not self.test:
        self.train_writer = tf.summary.FileWriter("./data/logs/" + self.name + '/' + 'exq_lr_' + str(self.lr) + '_' +
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
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.num_s], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.num_s], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action
        self.done = tf.placeholder(tf.float32, [None, ], name='done')  # input Done info
        self.q_m_opt = tf.placeholder(tf.float32, [None, ], name='q_value_next_max_opt')
        self.q_m_avg = tf.placeholder(tf.float32, [None, ], name='q_value_next_max_avg')
        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.1), tf.constant_initializer(0.0)

        with tf.variable_scope(self.name + '_opt'):
            # ------------------ build evaluate_net ------------------
            with tf.variable_scope('eval_net'):
                e1 = tf.layers.dense(self.s, 128, tf.nn.relu, kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer, name='e1')
                e1 = tf.layers.dense(e1, 128, tf.nn.relu, kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer, name='e2')
                e1 = tf.layers.dense(e1, 64, tf.nn.relu, kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer, name='e3')
                self.q_eval_opt = tf.layers.dense(e1, self.num_a, kernel_initializer=w_initializer,
                                                  bias_initializer=b_initializer, name='q_e')
            # ------------------ build target_net ------------------
            with tf.variable_scope('target_net'):
                t1 = tf.layers.dense(self.s_, 128, tf.nn.relu, kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer, name='t1')
                t1 = tf.layers.dense(t1, 128, tf.nn.relu, kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer, name='t2')
                t1 = tf.layers.dense(t1, 64, tf.nn.relu, kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer, name='t3')
                self.q_next_opt = tf.layers.dense(t1, self.num_a, kernel_initializer=w_initializer,
                                                  bias_initializer=b_initializer, name='q_t')

            with tf.variable_scope('q_target'):
                q_target_opt = self.r + (1 - self.done) * self.gamma * self.q_m_opt #tf.reduce_max(self.q_next_opt, axis=1, name='Qmax_s_')
                self.q_target_opt = tf.stop_gradient(q_target_opt)
            with tf.variable_scope('q_eval'):
                a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
                self.q_eval_wrt_a_opt = tf.gather_nd(params=self.q_eval_opt, indices=a_indices)
            with tf.variable_scope('loss'):
                self.td_error_opt = self.q_target_opt - self.q_eval_wrt_a_opt
                self.squared_error_opt = tf.where(tf.less(self.td_error_opt, 0),
                                                     self.beta * tf.square(self.td_error_opt),
                                                     tf.square(self.td_error_opt))
                self.loss_opt = tf.reduce_mean(self.squared_error_opt, name='loss')
                # self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
            with tf.variable_scope('train'):
                self._train_op_opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss_opt)

        with tf.variable_scope(self.name+'_avg'):
            # ------------------ build evaluate_net ------------------
            with tf.variable_scope('eval_net'):
                e1 = tf.layers.dense(self.s, 128, tf.nn.relu, kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer, name='e1')
                e1 = tf.layers.dense(e1, 128, tf.nn.relu, kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer, name='e2')
                e1 = tf.layers.dense(e1, 64, tf.nn.relu, kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer, name='e3')
                self.q_eval_avg = tf.layers.dense(e1, self.num_a, kernel_initializer=w_initializer,
                                                  bias_initializer=b_initializer, name='q_e')
            # ------------------ build target_net ------------------
            with tf.variable_scope('target_net'):
                t1 = tf.layers.dense(self.s_, 128, tf.nn.relu, kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer, name='t1')
                t1 = tf.layers.dense(t1, 128, tf.nn.relu, kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer, name='t2')
                t1 = tf.layers.dense(t1, 64, tf.nn.relu, kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer, name='t3')
                self.q_next_avg = tf.layers.dense(t1, self.num_a, kernel_initializer=w_initializer,
                                                  bias_initializer=b_initializer, name='q_t')

            with tf.variable_scope('q_target'):
                q_target_avg = self.r + (1 - self.done) * self.gamma * self.q_m_avg #tf.reduce_max(self.q_next_avg, axis=1, name='Qmax_s_')
                self.q_target_avg = tf.stop_gradient(q_target_avg)
            with tf.variable_scope('q_eval'):
                a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
                self.q_eval_wrt_a_avg = tf.gather_nd(params=self.q_eval_avg, indices=a_indices)
            with tf.variable_scope('loss'):
                self.td_error_avg = self.q_target_avg - self.q_eval_wrt_a_avg
                self.squared_error_avg = tf.square(self.td_error_avg)
                self.loss_avg = tf.reduce_mean(self.squared_error_avg, name='loss')
            with tf.variable_scope('train'):
                self._train_op_avg = tf.train.AdamOptimizer(self.lr).minimize(self.loss_avg)

    def act(self, state, avas, id, no_train):  # epsilon greedy
        if np.random.uniform() > self.epsilon or self.test or no_train:  # pick the argmax action
            s = np.array(state)
            if len(s.shape) < 2:
                s = np.array(state)[np.newaxis, :]
            # using opt net and avg net to choose action
            # todo:
            q_eval_opt, q_eval_avg = self.sess.run([self.q_eval_opt, self.q_eval_avg], feed_dict={self.s: s})
            q_eval_opt, q_eval_avg = q_eval_opt[0], q_eval_avg[0]
            # add available actions
            q_eval_opt[avas == 0] = -float('inf')
            q_eval_avg[avas == 0] = -float('inf')
            # get the max q_eval values of opt net
            opt_max_q_eval = np.max(q_eval_opt)
            # get all max actions
            opt_max_act_num = 0
            is_max_action_flags = [False] * self.num_a
            for x in range(len(q_eval_opt)):
                q_opt = q_eval_opt[x]
                if abs(opt_max_q_eval - q_opt) < 1e-3:
                    opt_max_act_num += 1
                    is_max_action_flags[x] = True
            assert opt_max_act_num >= 1
            # if there are multiple actions with the maximal optimistic values,
            # then choose the one with the maximal average value
            if opt_max_act_num > 1:
                avg_q_eval_mod = q_eval_avg[:]
                avg_min_q = np.min(avg_q_eval_mod)
                for x in range(len(avg_q_eval_mod)):
                    if not is_max_action_flags[x]:
                        avg_q_eval_mod[x] = avg_min_q - 100.0
                avg_max_action_list = []
                avg_max_q = np.max(avg_q_eval_mod)
                for x in range(len(avg_q_eval_mod)):
                    q_avg = avg_q_eval_mod[x]
                    if abs(avg_max_q - q_avg) < 1e-3:
                        avg_max_action_list.append(x)
                avg_max_action_num = len(avg_max_action_list)
                assert avg_max_action_num >= 1
                return avg_max_action_list[random.randint(0, avg_max_action_num-1)]
            else:
                opt_max_action = np.argmax(q_eval_opt)
                # if the optimistic max action is also the average max action
                # then directly return it
                avg_q_of_opt_max = q_eval_avg[opt_max_action]
                if abs(avg_q_of_opt_max - np.max(q_eval_avg)) < 1e-3:
                    return opt_max_action
                # else choose the optimistic max action with probability 0.5
                # choose the average max action with probability 0.5
                elif random.uniform(0, 1.0) < 0.5:
                    return opt_max_action
                else:
                    avg_max_action_list = []
                    avg_max_q = np.max(q_eval_avg)
                    for x in range(len(q_eval_avg)):
                        avg_q = q_eval_avg[x]
                        if abs(avg_max_q - avg_q) < 1e-3:
                            avg_max_action_list.append(x)
                    avg_max_action_num = len(avg_max_action_list)
                    assert avg_max_action_num >= 1
                    return avg_max_action_list[random.randint(0, avg_max_action_num-1)]

        else:  # pick random action
            avail_action_dim = sum(avas)
            action = np.random.randint(0, avail_action_dim)
        return action

    def store(self, uuid, id, scene, exp):
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
        # calc q target next for opt, avg
        q_next_opt = self.sess.run(self.q_next_opt, feed_dict={self.s_: s_})
        q_next_avg = self.sess.run(self.q_next_avg, feed_dict={self.s_: s_})
        q_next_opt[np.array(avas)[:, :] == 0] = - 999999  # mask unavailable actions
        q_next_avg[np.array(avas)[:, :] == 0] = - 999999  # mask unavailable actions
        q_m_opt = np.max(q_next_opt, axis=1)
        q_m_avg = np.max(q_next_avg, axis=1)

        _, _, old_opt_q, opt_td_errors, opt_loss, avg_td_errors, avg_loss = \
            self.sess.run([self._train_op_opt, self._train_op_avg,
                           self.q_eval_opt, self.td_error_opt, self.loss_opt,
                           self.td_error_avg, self.loss_avg],
                          feed_dict={self.s: s, self.a: a, self.r: r, self.s_: s_, self.done: done,
                                     self.q_m_opt: q_m_opt, self.q_m_avg: q_m_avg})
        # compare the difference of opt value before and after update
        new_opt_q = self.sess.run(self.q_eval_opt,
                                  feed_dict={self.s: s, self.a: a, self.r: r, self.s_: s_, self.done: done})
        old_opt_q, new_opt_q = old_opt_q[0], new_opt_q[0]
        q_delta_abs = 0
        for sa_index in range(len(new_opt_q)):
            q_delta_abs += abs(old_opt_q[sa_index] - new_opt_q[sa_index])
        # the difference of qvalue before and after the update
        q_delta_abs /= len(new_opt_q)
        self.write_summary_scalar("Batch_Qsa_Diff", q_delta_abs, self.learn_step_cnt)

        opt_td_square_error = 0
        avg_td_square_error = 0
        opt_td_square_error_used = 0
        opt_td_square_error_unused = 0
        positive_count = 0
        for sa_index in range(len(opt_td_errors)):
            opt_td_error = opt_td_errors[sa_index]
            avg_td_error = avg_td_errors[sa_index]
            avg_tds = avg_td_error ** 2
            opt_tds = opt_td_error ** 2
            avg_td_square_error += avg_tds
            opt_td_square_error += opt_tds
            if opt_td_error > 0:
                opt_td_square_error_used += opt_tds
                positive_count += 1
            else:
                opt_td_square_error_unused += opt_tds
        opt_td_square_error /= len(opt_td_errors)
        avg_td_square_error /= len(opt_td_errors)
        if positive_count == 0:
            opt_td_square_error_used = 0
        else:
            opt_td_square_error_used /= positive_count

        opt_td_square_error_unused /= (len(opt_td_errors) - positive_count)
        self.write_summary_scalar("Batch_MSE_Avg", avg_td_square_error, self.learn_step_cnt)
        self.write_summary_scalar("Batch_MSE_Opt", opt_td_square_error, self.learn_step_cnt)
        self.write_summary_scalar("Batch_MSE_Opt_Used", opt_td_square_error_used, self.learn_step_cnt)
        self.write_summary_scalar("Batch_MSE_Opt_Unused", opt_td_square_error_unused, self.learn_step_cnt)

        # some general summary
        self.write_summary_scalar('loss_opt', opt_loss, self.learn_step_cnt)
        self.write_summary_scalar('loss_avg', avg_loss, self.learn_step_cnt)
        self.write_summary_scalar('epsilon', self.epsilon, self.learn_step_cnt)
        self.write_summary_scalar('beta', self.beta, self.learn_step_cnt)
        self.write_summary_scalar('memory_cnt', self.memory_counter, self.learn_step_cnt)
        self.save()  # save model
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)  # decay epsilon
        self.beta = max(self.beta_min, self.beta * self.beta_decay_factor)  # decay beta
        self.learn_step_cnt += 1

        # check to do the soft replacement of target net
        if self.learn_step_cnt % self.replace_target_iter == 0 and self.learn_step_cnt:
            self.sess.run(self.target_replace_op_opt)
            self.sess.run(self.target_replace_op_avg)

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

