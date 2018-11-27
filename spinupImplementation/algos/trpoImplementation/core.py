# REFERENCE IMPLEMENTATION

import numpy as np
import tensorflow as tf
import scipy.signal
from gym.spaces import Box, Discrete

EPS = 1e-8

def combined_shape(length, shape=None):
    """
    Takes a length and shape and returns a them as a tuple.
    Should handle when the shape is None, a scalar, or a tuple
    """

    

def keys_as_sorted_list(dict):
    """
    Given a dictionary, get the keys from it, and convert it into a list,
    then sort the list of keys
    """

    

def values_as_sorted_list(dict):
    """
    Given a dictionary, create a sorted list of keys, then extract the values of 
    those keys. Return them as a list
    """

    

def placeholder(dim=None):
    """
    Creates a computation graph placeholder given some dimensions, of datatype: float32
    """

    

def placeholders(*args):
    """
    Creates a list of computation graph placeholders given a set of dimensions
    """

    

def placeholder_from_space(space):
    """
    Given a space, Box or Discrete, create a computation placeholder. Box should
    have a datatype of int32
    """

    

def placeholders_from_spaces(*args):
    """
    Given a set of spaces, create a list of computation graph placesholders for them
    """

    

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    """
    Create a multilayer perceptron with a number of layers based on the hidden sizes.
    Assume the last layer will have different output_activation (linear)
    """

    

def get_vars(scope=''):
    """
    Given a computation graph variable scope, get the trainable variables within
    that scope. Return them as a list.
    """

    

def count_vars(scope=''):
    """
    Given a computation graph variable scope, get the trainable variables from it. 
    Then calculate the total amount of them. Take into account that there are many layers.
    """

    

def gaussian_likelihood(x, mu, log_std):
    """
    Get the log likelihood of a diagonal gaussian policy.
    https://spinningup.openai.com/en/latest/spinningup/rl_intro.html

    Returns: The log likelihood an action
    """

    

def diagonal_gaussian_kl(mu0, log_std0, mu1, log_std1):
    """
    tf symbol for mean KL divergence between two batches of diagonal gaussian distributions,
    where distributions are specified by means and log stds.
    (https://en.wikipedia.org/wiki/Kullback-Leibler_divergence#Multivariate_normal_distributions)
    https://spinningup.openai.com/en/latest/algorithms/trpo.html
    """
    
    

def categorical_kl(logp0, logp1):
    """
    tf symbol for mean KL divergence between two batches of categorical probability distributions,
    where the distributions are input as log probs.
    """
    
    

def flat_concat(xs):
    """
    Given a variable, flatten it, then combines them along the same axis
    https://www.tensorflow.org/api_docs/python/tf/concat
    https://www.tensorflow.org/api_docs/python/tf/reshape
    """



def flat_grad(f, params):
    """
    Calculate gradients and combine them along the same axis
    https://www.tensorflow.org/api_docs/python/tf/gradients
    """
    


def hessian_vector_product(f, params):
    """
    Compute the hessian vector product
    https://spinningup.openai.com/en/latest/algorithms/trpo.html

    use flat_grad to get g
    Create a placeholder for x

    return x and the hessian vector produce
    """

    # for H = grad**2 f, compute Hx


def assign_params_from_flat(x, params):
    """

    Create a function to get the flat size of the input
    split x based on params's flat size
    https://www.tensorflow.org/api_docs/python/tf/split

    Create new params using the split sizes and params

    Return the new group of assigned variables
    https://www.tensorflow.org/api_docs/python/tf/group
    https://www.tensorflow.org/api_docs/python/tf/assign
    """

    # the 'int' is important for scalars
    

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
    
    # https://stackoverflow.com/questions/47970683/vectorize-a-numpy-discount-calculation    
    """
    

"""
Policies
"""

def mlp_categorical_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    """
    Runs the MLP for categorial policy, used for Discrete spaces

    Uses categorical kl

    Returns:
        pi, logp, logp_pi, info, info_phs, d_kl
    """
    
    


def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    """
    Runs the MLP for gaussian policy, Box spaces

    Uses diagonal gaussial likelihood

    returns pi, logp, logp_ph, info, info_phs, d_kl
    """
    
    


"""
Actor-Critics
"""
def mlp_actor_critic(x, a, hidden_sizes=(64,64), activation=tf.tanh, 
                     output_activation=None, policy=None, action_space=None):

    """
    Runs the policy based on Action space, Box or Discrete.
    Returns:
        pi: sample actions from policy given states
        logp: log probability according to policy taking actions a_ph
        logp_pi: log probability according to policy taking actions pi
        info: dictionary of intermediate quantities for analytically computing KL divergence
        info_phs: dictionary of placeholders for old values of the entries in info
        d_kl: symbol for computing mean KL divergence between current and old policy over batch of statges
        v: value estimates for states (flatten this)
    """

    # default policy builder depends on action space
    