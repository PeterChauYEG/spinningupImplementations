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

    # check if shape is None.
    # if so, we cant check if it's scalar
    if shape is None:
        result = (length, )
        return result

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
    shape = combined_shape(None, dim)

    # create the placeholder, assume the datatype is float32
    # is there a scope issue?
    graph_placeholder = tf.placeholder(dtype=tf.float32, shape=shape)

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

    # handle Box space
    if isinstance(space, Box):
        space_placeholder = placeholder(space.shape)
    elif isinstance(space, Discrete):
        # setup a placeholder for Discrete space. This only has 1d 
        # get datatype as an int
        space_placeholder = tf.placeholder(dtype=tf.int32, shape=(None,))

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
    vars_list = [np.prod(var.shape.as_list()) for var in vars]

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
    log probabilities.
    https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#key-concepts-and-terminology
    https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html

    pi: samples actions from the given policy given states
    logp: log probability according to the policy, of taking actions a_ph in states x_ph
    logp_pi: log probablity accoring to the policy, of the action sampled by pi
    """

    # get the number of actions in action space
    action_dim = action_space.n

    # construct the hidden sizes for getting hidden sizes. 
    # the last layer needs to have an output matching then dims of the action space
    logit_hidden_sizes = list(hidden_sizes) + [action_dim]

    # get logits using x
    logits = mlp(x, 
        hidden_sizes=logit_hidden_sizes, 
        activation=activation, 
        output_activation=output_activation)

    # run the logits through a softmax to squash values to be between 0 and 1
    # this give probabilities of the action
    # this is P_theta(s)
    logits_squashed = tf.nn.log_softmax(logits)

    # sample the actions from the policy
    # feed it the logits and the number of samples we want
    # remove 1s on axis 1
    pi = tf.squeeze(tf.multinomial(logits_squashed, 1), axis=1)

    # get the log prob of the policy with a
    # a needs to be hot-one encoded to turn the actions into indices
    # this is the log-likelihood log_pi_theta(a|s)
    logp = tf.reduce_sum(tf.one_hot(a, depth=action_dim) * logits_squashed, axis=1)

    # get the log prob of the policy with ph
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=action_dim) * logits_squashed, axis=1)

    return pi, logp, logp_pi


"""
Actor-Critics
"""
def mlp_actor_critic(x, a, hidden_sizes=(64,64), activation=tf.tanh, 
                     output_activation=None, policy=None, action_space=None):
    """
    This is for Discrete action space.
    Takes in placeholder symbols for state, ``x_ph``, and action, ``a_ph``, 
    and returns the main outputs from the agent's Tensorflow computation graph:

    ===========  ================  ======================================
    Symbol       Shape             Description
    ===========  ================  ======================================
    ``pi``       (batch, act_dim)  | Samples actions from policy given 
                                    | states.
    ``logp``     (batch,)          | Gives log probability, according to
                                    | the policy, of taking actions ``a_ph``
                                    | in states ``x_ph``. Used to calculate pi loss.
    ``logp_pi``  (batch,)          | Gives log probability, according to
                                    | the policy, of the action sampled by
                                    | ``pi``.
    ``v``        (batch,)          | Gives the value estimate for states
                                    | in ``x_ph``. (Critical: make sure 
                                    | to flatten this!) Used to calculate pi loss.
    ===========  ================  ======================================
    """

    policy = mlp_categorical_policy

    # ensure we set the variable scope to not polute calculation of pi
    with tf.variable_scope('pi'):
        # get pi, logp, logp_pi
        pi, logp, logp_pi = policy(x, 
                                   a, 
                                   hidden_sizes, 
                                   activation, 
                                   output_activation, 
                                   action_space)

    # ensure we set the variable scope to not polute calculation of a
    with tf.variable_scope('v'):
        # generate hidden_sizes for v
        # we only want 1 output
        v_hidden_sizes = list(hidden_sizes) + [1]

        # get v
        # remove 1s from axis 1
        logits = mlp(x, 
            hidden_sizes=v_hidden_sizes, 
            activation=activation, 
            output_activation=output_activation)

        v = tf.squeeze(logits, axis=1)

    return pi, logp, logp_pi, v
