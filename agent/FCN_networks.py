import tensorflow as tf
import tensorflow.contrib.layers as layers

def Policy_network():
    state_holder = tf.placeholder(tf.float32, (None, 13))
    extra_feature = state_holder[:,0:6] - state_holder[:,7:]
    first_in = tf.concat((state_holder, extra_feature), axis = 1)
    first_hidden = layers.fully_connected(first_in,
                                          num_outputs=256,
                                          activation_fn=tf.nn.relu6,
                                          scope='policy_hidden_1')
    logits = layers.fully_connected(first_hidden,
                                   num_outputs=61,
                                   activation_fn=None,
                                   scope='2logits')
    return state_holder, logits

def Value_network():
    state_holder = tf.placeholder(tf.float32, (None, 15))
    extra_feature = state_holder[:,0:6] - state_holder[:,9:]
    first_in = tf.concat((state_holder, extra_feature), axis = 1)

    first_hidden = layers.fully_connected(first_in,
                           num_outputs=256,
                           activation_fn=tf.nn.relu,
                           scope='value_hidden_1')
    Value = layers.fully_connected(first_hidden,
                                          num_outputs=1,
                                          activation_fn=tf.nn.tanh,
                                          scope='2value')
    return state_holder, Value

def define_networks():
    policy_state_holder, logits = Policy_network()
    Value_state_holder, Value = Value_network()
    return policy_state_holder, logits, Value_state_holder, Value