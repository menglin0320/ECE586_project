import tensorflow as tf
import tensorflow.contrib.layers as layers

def Policy_network():
    state_holder = tf.placeholder(tf.float32, (None, 91))
    first_hidden = layers.fully_connected(state_holder,
                                          num_outputs=256,
                                          activation_fn=tf.nn.relu,
                                          scope='policy_hidden_1')
    second_hidden = layers.fully_connected(first_hidden,
                                          num_outputs=128,
                                          activation_fn=tf.nn.relu,
                                          scope='policy_hidden_2')
    third_hidden = layers.fully_connected(second_hidden,
                                          num_outputs=64,
                                          activation_fn=tf.nn.relu,
                                          scope='policy_hidden_3')
    logits = layers.fully_connected(third_hidden,
                                   num_outputs=61,
                                   activation_fn=None,
                                   scope='2logits')
    return state_holder, logits

def Value_network():
    state_holder = tf.placeholder(tf.float32, (None, 92))
    first_hidden = layers.fully_connected(state_holder,
                           num_outputs=256,
                           activation_fn=tf.nn.relu,
                           scope='value_hidden_1')
    second_hidden = layers.fully_connected(first_hidden,
                           num_outputs=128,
                           activation_fn=tf.nn.relu,
                           scope='value_hidden_2')
    third_hidden = layers.fully_connected(second_hidden,
                                          num_outputs=64,
                                          activation_fn=tf.nn.relu,
                                          scope='value_hidden_3')
    Value = layers.fully_connected(third_hidden,
                                          num_outputs=1,
                                          activation_fn=tf.nn.tanh,
                                          scope='2value')

    return state_holder, Value

def define_networks():
    policy_state_holder, logits = Policy_network()
    Value_state_holder, Value = Value_network()
    return policy_state_holder, logits, Value_state_holder, Value