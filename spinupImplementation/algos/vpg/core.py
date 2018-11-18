# Core module of VPG
# This module provides functions which provides functions for the mlp actor critic
# and functions to work with it.

import numpy as np
import tensorflow as tf
import scipy.signal
from gym.spaces import Box, Discrete

def combined_shape(length, shape=None):
    """
    Returns a shape based on the length (size of something), and the dimensions of that space.
    For CartPole-v0, observations is a Box space, and actions is a discrete space. 
    """

    # check if the shape is None, if so, we ignore it
    # (length, )
    if shape is None:
        return (length,)

    # check if the shape is scalar or a tuple
    # return the (length, shape) or (length, *shape)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def placeholder(dim=None):
    """
    Creates computational graph placeholder of with the shape (None, dim)
    """

    return tf.placeholder(dtype=tf.float32, shape=combined_shape(None,dim))

def placeholders(*args):
    """
    Creates a list of computational graph placeholders based on the arguments provided
    """

    return [placeholder(dim) for dim in args]

def placeholder_from_space(space):
    """
    Creates computational graph placeholder for a space. A space being an observation space or action space.
    Handles Box and Discrete space.
    """

    # checks if the space is a Box or Discrete space.
    # a Discrete space has a 1d shape so it can return a computational graph placeholder with a shape of (None,)
    # a Box space has multiple dimensional shape so it returns a computation graph placeholder with a shape of (None, dim)
    if isinstance(space, Box):
        return placeholder(space.shape)
    elif isinstance(space, Discrete):
        return tf.placeholder(dtype=tf.int32, shape=(None,))
    raise NotImplementedError

def placeholders_from_spaces(*args):
    """
    Create a list of computational graph placeholders for spaces.
    """

    return [placeholder_from_space(space) for space in args]

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    """
    Create a multilayer perceptron where the with an abstract number of units and an activation function
    Assume that they last layer has a differing number of units (number of actions), and activation function (linear)
    """

    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def get_vars(scope=''):
    """
    Gets all trainable variables from a given scope of the computational graph
    and returns them as a list
    """
    
    return [x for x in tf.trainable_variables() if scope in x.name]

def count_vars(scope=''):
    """
    Gives the number of trainable variables in a given scope.
    """

    # get variables from the scope
    v = get_vars(scope)

    # get the product of the number of variables
    # each variable has a shape because the trainable variables are stacked in layers of the mlp
    return sum([np.prod(var.shape.as_list()) for var in v])

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """

    # we want to calculate
    # C[i] = R[i] + discount * C[i+1]
    # it turns out we can use lfilter to do this
    # https://stackoverflow.com/questions/47970683/vectorize-a-numpy-discount-calculation    

    # first param is the numberator for a coefficient vector
    # second param is the denominator for a coefficient vector
    # third param is an N-dimensional input array
    # forth param is the axis to apply the linear filter
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


"""
Policies
"""

def mlp_categorical_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    """
    Uses the MLP to get logits from the current policy, then sample them, and get 
    log probabilities
    pi: samples actions from the given policy given states
    logp: log probability according to the policy, of taking actions a_ph in states x_ph
    logp_pi: log probablity accoring to the policy, of the action sampled by pi
    """

    # number of actions
    act_dim = action_space.n

    # get value estimate logits from the mlp. Give a list of units for the layers, the last layer should be for actions
    logits = mlp(x, list(hidden_sizes)+[act_dim], activation, None)

    # apply a softmax to the logits to get values between 0 and 1
    logp_all = tf.nn.log_softmax(logits)

    # sample the logits and remove 1s from axis 1
    pi = tf.squeeze(tf.multinomial(logits,1), axis=1)

    # get the log probability of a
    # one hot encode a, then apply the softmax'd logits, then adding all of these into 1 number
    logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
    
    # get the log probability of pi
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
    
    return pi, logp, logp_pi


"""
Actor-Critics
"""
def mlp_actor_critic(x, a, hidden_sizes=(64,64), activation=tf.tanh, 
                     output_activation=None, policy=None, action_space=None):

    # only for discrete action space
    policy = mlp_categorical_policy

    # create a variable scope for pi
    with tf.variable_scope('pi'):
        # use the policy to get pi, logp, and logp_pi
        pi, logp, logp_pi = policy(x, a, hidden_sizes, activation, output_activation, action_space)
    
    # create a variable scope for pi
    with tf.variable_scope('v'):
        # use the mlp to get value estimates for states, then remove 1s in axis 1
        v = tf.squeeze(mlp(x, list(hidden_sizes)+[1], activation, None), axis=1)
    
    return pi, logp, logp_pi, v