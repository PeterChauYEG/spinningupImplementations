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
    
def placeholder(dim=None):
    """
    Creates computational graph placeholder of with the shape (None, dim)
    """

def placeholders(*args):
    """
    Creates a list of computational graph placeholders based on the arguments provided
    """

def placeholder_from_space(space):
    """
    Creates computational graph placeholder for a space. A space being an observation space or action space.
    Handles Box and Discrete space.
    """

def placeholders_from_spaces(*args):
    """
    Create a list of computational graph placeholders for spaces.
    """

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    """
    Create a multilayer perceptron where the with an abstract number of units and an activation function
    Assume that they last layer has a differing number of units (number of actions), and activation function (linear)
    """

def get_vars(scope=''):
    """
    Gets all trainable variables from a given scope of the computational graph
    and returns them as a list
    """
    
def count_vars(scope=''):
    """
    Gives the number of trainable variables in a given scope.
    """

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