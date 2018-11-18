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

    # initialize assume shape is None.
    result = (length, )

    # otherwise, handle shape as a scalar or a tuple
    if np.isscalar(shape):
        result = (length, shape)
    else:
        result = (length, *shape)

    return result

     
def placeholder(dim=None):
    """
    Creates computational graph placeholder of with the shape (None, dim)
    """

    # setup the shape for the placeholder
    shape = (None, dim)

    # create the placeholder, assume the datatype is float32
    # is there a scope issue?
    graph_placeholder = tf.placeholder(dtype=float32, shape=shape)

    return graph_placeholder

def placeholders(*args):
    """
    Creates a list of computational graph placeholders based on the arguments 
    provided
    """

    # create the placeholders
    graph_placeholders = [placeholder(dim) for dim in args]

    return graph_placeholders

def placeholder_from_space(space):
    """
    Creates computational graph placeholder for a space. A space being an 
    observation space or action space.
    Handles Box and Discrete space.
    """

    # setup a placeholder for Discrete space. This only has 1d 
    space_placeholder = tf.placeholder(dtype=float32, shape=(None,))

    # handle Box space
    if isinstance(Box):
        space_placeholder = placeholder(space.shape)

    return space_placeholder

def placeholders_from_spaces(*args):
    """
    Create a list of computational graph placeholders for spaces.
    """

    # create space placeholders
    space_placeholders = [placeholder_from_space(space) for space in args]

    return space_placeholders

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    """
    Create a multilayer perceptron where the with an abstract number of units and an activation function
    Assume that they last layer has a differing number of units (number of actions), and activation function (linear)
    """

    # create a later for all hidden sizes but the last one
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)

    # get the hidden sizes for the last layer which should be action dim
    last_h = hidden_sizes[-1]
    nn = tf.layers.dense(x, units=last_h, activation=output_activation) 

    return nn

def get_vars(scope=''):
    """
    Gets all trainable variables from a given scope of the computational graph
    and returns them as a list
    """
    
    # get trainable variables from a given scope
    vars = [var for var in tf.trainable_variables() if scope in var.name]

    return vars

def count_vars(scope=''):
    """
    Gives the number of trainable variables in a given scope.
    """

    # get vars in the scope
    vars = get_vars(scope)

    # get the number of variables in the scope
    vars_list = [np.prod(var.shape.as_list()) for var in vars]]

    # add them all up
    var_count = sum(vars_list)

    return var_count

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

    # use rllab to calculate discounted cumulative sums
    # C[i] = R[i] + discount * C[i+1]
    # it turns out we can use lfilter to do this
    # https://stackoverflow.com/questions/47970683/vectorize-a-numpy-discount-calculation    
    C = scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    return C

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

    

"""
Actor-Critics
"""
def mlp_actor_critic(x, a, hidden_sizes=(64,64), activation=tf.tanh, 
                     output_activation=None, policy=None, action_space=None):
