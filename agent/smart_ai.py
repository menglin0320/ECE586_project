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
        self.initial_lr = 0.0002
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
            self.saver.restore(self.sess, saved_path)
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
        self.action_probs = exp_valid_action_logits / tf.expand_dims(tf.reduce_sum(exp_valid_action_logits, axis=-1), 1)

        self.Value_target = tf.placeholder(tf.float32, [None, 1], name='Value_target')
        self.Value_loss = 1 / 2 * tf.reduce_mean(tf.square(self.Value - self.Value_target))

        self.entry_diffs = tf.square(self.Value - self.Value_target)
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

    def act(self, bid, dice):
        whole_state_in = get_state(bid, dice)
        valid_action_mask = self.get_action_mask()
        [action_probs] = self.sess.run(self.action_probs, feed_dict={self.policy_state_holder: whole_state_in, \
                                                                     self.valid_action_mask: valid_action_mask})
        # print(action_probs)
        policy_action = np.argmax(action_probs)
        epsilan = 0.3
        valid_actions = get_valid_actions(self.all_actions, self.bid)
        action = epsilan_greedy(valid_actions, policy_action, self.train_iter, epsilan)
        return action_probs, action, valid_action_mask

    def get_next_value(self, bid, player_observations, cur_player):
        # let the opponent move
        opponent = 1 - cur_player
        _, action, _ = self.act(bid, player_observations[opponent])

        if action == 'liar':
            value_state_bid = ['checked', bid]
        else:
            value_state_bid = action
        value_state = get_value_state(value_state_bid, player_observations[cur_player])

        [V] = self.sess.run(self.Value, feed_dict={self.value_state_holder: value_state})
        return V

    def get_next_state_values(self, history, player_observations):
        # TODO calc Value function for first state and reuse them
        # This function can be greatly optimized, just being lazy here
        value_for_policy_list = []
        cur_player = 1
        for node in history:
            valid_actions = np.asarray(get_valid_actions(self.all_actions, node))
            # valid_actions = valid_actions.tolist()
            # valid_actions.append('checked')
            valid_actions = np.asarray(valid_actions)
            nextnode_values = []
            cur_player = 1 - cur_player
            for i, action in enumerate(valid_actions):
                if action == 'liar':
                    action = ['liar', node]
                    value_state = get_value_state(action, player_observations[cur_player])
                    [V] = self.sess.run(self.Value, feed_dict={self.value_state_holder: value_state})
                    nextnode_values.append(V[0])
                else:
                    V = self.get_next_value(action, player_observations, cur_player)
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
        states = np.asarray(states)
        Value_target = np.asarray(Value_target)
        Value_target = np.expand_dims(Value_target, 0).transpose()
        _, Value_loss, Values = self.sess.run([self.Value_opt, self.Value_loss, self.Value], feed_dict={self.value_state_holder: states,
                                                                                self.Value_target: Value_target})
        return Value_loss,Values
