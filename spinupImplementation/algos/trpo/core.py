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

    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def keys_as_sorted_list(dict):
    """
    Given a dictionary, get the keys from it, and convert it into a list,
    then sort the list of keys
    """

    return sorted(list(dict.keys()))

def values_as_sorted_list(dict):
    """
    Given a dictionary, create a sorted list of keys, then extract the values of 
    those keys. Return them as a list
    """

    return [dict[k] for k in keys_as_sorted_list(dict)]

def placeholder(dim=None):
    """
    Creates a computation graph placeholder given some dimensions, of datatype: float32
    """

    return tf.placeholder(dtype=tf.float32, shape=combined_shape(None,dim))

def placeholders(*args):
    """
    Creates a list of computation graph placeholders given a set of dimensions
    """

    return [placeholder(dim) for dim in args]

def placeholder_from_space(space):
    """
    Given a space, Box or Discrete, create a computation placeholder. Box should
    have a datatype of int32
    """

    if isinstance(space, Box):
        return placeholder(space.shape)
    elif isinstance(space, Discrete):
        return tf.placeholder(dtype=tf.int32, shape=(None,))
    raise NotImplementedError

def placeholders_from_spaces(*args):
    """
    Given a set of spaces, create a list of computation graph placesholders for them
    """

    return [placeholder_from_space(space) for space in args]

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    """
    Create a multilayer perceptron with a number of layers based on the hidden sizes.
    Assume the last layer will have different output_activation (linear)
    """

    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def get_vars(scope=''):
    """
    Given a computation graph variable scope, get the trainable variables within
    that scope. Return them as a list.
    """

    return [x for x in tf.trainable_variables() if scope in x.name]

def count_vars(scope=''):
    """
    Given a computation graph variable scope, get the trainable variables from it. 
    Then calculate the total amount of them. Take into account that there are many layers.
    """

    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def gaussian_likelihood(x, mu, log_std):
    """
    Get the log likelihood of a diagonal gaussian policy.
    https://spinningup.openai.com/en/latest/spinningup/rl_intro.html

    Returns: The log likelihood an action
    """

    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def diagonal_gaussian_kl(mu0, log_std0, mu1, log_std1):
    """
    tf symbol for mean KL divergence between two batches of diagonal gaussian distributions,
    where distributions are specified by means and log stds.
    (https://en.wikipedia.org/wiki/Kullback-Leibler_divergence#Multivariate_normal_distributions)
    https://spinningup.openai.com/en/latest/algorithms/trpo.html
    """
    var0, var1 = tf.exp(2 * log_std0), tf.exp(2 * log_std1)
    pre_sum = 0.5*(((mu1- mu0)**2 + var0)/(var1 + EPS) - 1) +  log_std1 - log_std0
    all_kls = tf.reduce_sum(pre_sum, axis=1)
    return tf.reduce_mean(all_kls)

def categorical_kl(logp0, logp1):
    """
    tf symbol for mean KL divergence between two batches of categorical probability distributions,
    where the distributions are input as log probs.
    """
    all_kls = tf.reduce_sum(tf.exp(logp1) * (logp1 - logp0), axis=1)
    return tf.reduce_mean(all_kls)

def flat_concat(xs):
    """
    Given a variable, flatten it, then combines them along the same axis
    https://www.tensorflow.org/api_docs/python/tf/concat
    https://www.tensorflow.org/api_docs/python/tf/reshape
    """

    return tf.concat([tf.reshape(x,(-1,)) for x in xs], axis=0)

def flat_grad(f, params):
    """
    Calculate gradients and combine them along the same axis
    https://www.tensorflow.org/api_docs/python/tf/gradients
    """
    
    return flat_concat(tf.gradients(xs=params, ys=f))

def hessian_vector_product(f, params):
    """
    Compute the hessian vector product
    https://spinningup.openai.com/en/latest/algorithms/trpo.html

    use flat_grad to get g
    Create a placeholder for x

    return x and the hessian vector produce
    """

    # for H = grad**2 f, compute Hx
    g = flat_grad(f, params)
    x = tf.placeholder(tf.float32, shape=g.shape)
    return x, flat_grad(tf.reduce_sum(g*x), params)

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

    flat_size = lambda p : int(np.prod(p.shape.as_list())) # the 'int' is important for scalars
    splits = tf.split(x, [flat_size(p) for p in params])
    new_params = [tf.reshape(p_new, p.shape) for p, p_new in zip(params, splits)]
    return tf.group([tf.assign(p, p_new) for p, p_new in zip(params, new_params)])

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
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

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
    
    act_dim = action_space.n
    logits = mlp(x, list(hidden_sizes)+[act_dim], activation, None)
    logp_all = tf.nn.log_softmax(logits)
    pi = tf.squeeze(tf.multinomial(logits,1), axis=1)
    logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)

    old_logp_all = placeholder(act_dim)
    d_kl = categorical_kl(logp_all, old_logp_all)

    info = {'logp_all': logp_all}
    info_phs = {'logp_all': old_logp_all}

    return pi, logp, logp_pi, info, info_phs, d_kl


def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    """
    Runs the MLP for gaussian policy, Box spaces

    Uses diagonal gaussial likelihood

    returns pi, logp, logp_ph, info, info_phs, d_kl
    """
    
    act_dim = a.shape.as_list()[-1]
    mu = mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
    log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(act_dim, dtype=np.float32))
    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp = gaussian_likelihood(a, mu, log_std)
    logp_pi = gaussian_likelihood(pi, mu, log_std)

    old_mu_ph, old_log_std_ph = placeholders(act_dim, act_dim)
    d_kl = diagonal_gaussian_kl(mu, log_std, old_mu_ph, old_log_std_ph)

    info = {'mu': mu, 'log_std': log_std}
    info_phs = {'mu': old_mu_ph, 'log_std': old_log_std_ph}

    return pi, logp, logp_pi, info, info_phs, d_kl


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
    if policy is None and isinstance(action_space, Box):
        policy = mlp_gaussian_policy
    elif policy is None and isinstance(action_space, Discrete):
        policy = mlp_categorical_policy

    with tf.variable_scope('pi'):
        policy_outs = policy(x, a, hidden_sizes, activation, output_activation, action_space)
        pi, logp, logp_pi, info, info_phs, d_kl = policy_outs
    with tf.variable_scope('v'):
        v = tf.squeeze(mlp(x, list(hidden_sizes)+[1], activation, None), axis=1)
    return pi, logp, logp_pi, info, info_phs, d_kl, v