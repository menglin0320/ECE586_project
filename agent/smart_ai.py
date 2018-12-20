from agent.FCN_networks import define_networks
import tensorflow as tf
import tensorflow.contrib.layers as layers
from util import *
import os


class smart_agent(object):
    def __init__(self, name):
        self.name = name
        self.dice = []
        self.bid = []
        self.all_actions = get_all_actions()
        self.initial_lr = 0.002
        self.train_iter = 0
        self.build_model()
        self.setup()

    def setup(self):
        self.sess = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=3)
        checkpoint_dir = 'model_save'
        saved_path = tf.train.latest_checkpoint(checkpoint_dir)
        if (saved_path):
            # tf.reset_default_graph()
            self.summary_writer.restore(self.sess, saved_path)
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
            self.start_step = int(step)
            print('model restored', self.start_step)
        else:
            self.sess.run(tf.global_variables_initializer())
            print('model initialized')
    def initialize(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def see_dice(self, dice):
        self.dice = dice

    def see_bid(self, bid):
        self.bid = bid

    def get_action_mask(self):
        valid_actions = np.asarray(get_valid_actions(self.all_actions, self.bid))
        if valid_actions[0] == 'liar':
            valid_actions_mask = np.zeros(len(self.all_actions))
            valid_actions_mask[-1] = 1
            return [valid_actions_mask]
        valid_actions_start = ((valid_actions[0][0]) - 1) * 6 + valid_actions[0][1] - 1
        valid_actions_mask = np.zeros(len(self.all_actions))
        valid_actions_mask[valid_actions_start:] = 1
        return [valid_actions_mask]

    def build_model(self, reuse=False):
        with tf.variable_scope(self.name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
                assert tf.get_variable_scope().reuse
        policy_state_holder, logits, Value_state_holder, Value = define_networks()

        self.policy_state_holder = policy_state_holder
        self.value_state_holder = Value_state_holder
        self.Value = Value
        self.valid_action_mask = tf.placeholder(tf.float32, [None, 61], name='valid_action_mask')
        exp_action_logits = tf.math.exp(logits)
        exp_valid_action_logits = tf.multiply(exp_action_logits, self.valid_action_mask)
        self.action_probs = exp_valid_action_logits / tf.expand_dims(tf.reduce_sum(exp_valid_action_logits, axis=-1),1)

        self.Value_target = tf.placeholder(tf.float32, [None], name='Value_target')
        self.Value_loss = 1 / 2 * tf.reduce_mean(tf.square(self.Value - self.Value_target))
        Value_counter_dis = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
        self.Value_lr = tf.train.exponential_decay(self.initial_lr, Value_counter_dis, 300000, 0.96, staircase=True)
        self.Value_opt = layers.optimize_loss(loss=self.Value_loss, learning_rate=self.Value_lr,
                                              optimizer=tf.train.AdamOptimizer,
                                              clip_gradients=100., global_step=Value_counter_dis)

        self.Policy_target = tf.placeholder(tf.float32, [None, 61], name='Policy_target')
        self.Policy_loss = 1 / 2 * tf.reduce_mean(tf.square(self.action_probs - self.Policy_target))
        Policy_counter_dis = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
        self.Policy_lr = tf.train.exponential_decay(self.initial_lr, Policy_counter_dis, 300000, 0.96, staircase=True)
        self.Policy_opt = layers.optimize_loss(loss=self.Policy_loss, learning_rate=self.Policy_lr,
                                               optimizer=tf.train.AdamOptimizer,
                                               clip_gradients=100., global_step=Policy_counter_dis)

    def act(self):
        whole_state_in = get_state(self.bid, self.dice)
        valid_action_mask = self.get_action_mask()
        [action_probs] = self.sess.run(self.action_probs, feed_dict={self.policy_state_holder: whole_state_in, \
                                                                     self.valid_action_mask: valid_action_mask})
        # print(action_probs)
        policy_action = np.argmax(action_probs)
        epsilan = 0.05
        valid_actions = get_valid_actions(self.all_actions, self.bid)
        action = epsilan_greedy(valid_actions, policy_action, self.train_iter, epsilan)
        return action_probs, action, valid_action_mask

    def get_next_state_values(self, history, player_observations):
        # TODO calc Value function for first state and reuse them
        # This function can be greatly optimized, just being lazy here
        value_for_policy_list = []
        cur_player = 1
        for node in history:
            valid_actions = np.asarray(get_valid_actions(self.all_actions, node))
            nextnode_values = []
            cur_player = 1 - cur_player
            for action in valid_actions:
                whole_state_in = get_value_state(action, player_observations[cur_player])
                [V] = self.sess.run(self.Value, feed_dict={self.value_state_holder: whole_state_in})
                nextnode_values.append(V[0])
            value_for_policy_list.append(nextnode_values)
        return value_for_policy_list

    def train_policy_net(self, states, policy_target, masks):
        masks = np.asarray(masks)
        _, policy_loss = self.sess.run([self.Policy_opt, self.Policy_loss], feed_dict={self.policy_state_holder: states,
                                                                                       self.valid_action_mask: masks,
                                                                                       self.Policy_target: policy_target})
        self.train_iter += 1
        return policy_loss

    def train_Value_net(self, states, Value_target):
        _, Value_loss = self.sess.run([self.Value_opt, self.Value_loss], feed_dict={self.value_state_holder: states,
                                                                                    self.Value_target: Value_target})
        return Value_loss
