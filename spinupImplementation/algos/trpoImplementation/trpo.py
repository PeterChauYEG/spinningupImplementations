import numpy as np
import tensorflow as tf
import gym
import time
import spinup.algos.trpo.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


EPS = 1e-8

class GAEBuffer:
    """
    A buffer for storing trajectories experienced by a TRPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, info_shapes, gamma=0.99, lam=0.95):
        """
        Initialize properties:

        Environment:
        observations, actions, rewards, total expected rewards

        Computed:
        advantages, values, logps, infos,

        Training:
        gamma, lam

        Store:
        path trajectory, path start index, max size of store

        Get sorted info keys
        """

        

    def store(self, obs, act, rew, val, logp, info):
        """
        Append one timestep of agent-environment interaction to the buffer.

        Increment the path trajectory counter
        """
        # buffer has to have room so you can store
        

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        # get current path trajectory indices
        

        # get current path trajectory rewards and values
        
        # the next two lines implement GAE-Lambda advantage calculation
        
        # the next line computes rewards-to-go, to be targets for the value function
        
        
        # update the part trajectory start of the next trajectory
        

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        
        # buffer has to be full before you can get

        # reset the path trajectory counter and start index
        

        # the next two lines implement the advantage normalization trick
        
        # return obs_buf, act_buf, adv_buf, ret_buf, logp_buf, and info_bUf (sorted) as a list
        

"""

Trust Region Policy Optimization 

(with support for Natural Policy Gradient)

"""
def trpo(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0, 
         steps_per_epoch=4000, epochs=50, gamma=0.99, delta=0.01, vf_lr=1e-3,
         train_v_iters=80, damping_coeff=0.1, cg_iters=10, backtrack_iters=10, 
         backtrack_coeff=0.8, lam=0.97, max_ep_len=1000, logger_kwargs=dict(), 
         save_freq=10, algo='trpo'):
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:

            ============  ================  ========================================
            Symbol        Shape             Description
            ============  ================  ========================================
            ``pi``        (batch, act_dim)  | Samples actions from policy given 
                                            | states.
            ``logp``      (batch,)          | Gives log probability, according to
                                            | the policy, of taking actions ``a_ph``
                                            | in states ``x_ph``.
            ``logp_pi``   (batch,)          | Gives log probability, according to
                                            | the policy, of the action sampled by
                                            | ``pi``.
            ``info``      N/A               | A dict of any intermediate quantities
                                            | (from calculating the policy or log 
                                            | probabilities) which are needed for
                                            | analytically computing KL divergence.
                                            | (eg sufficient statistics of the
                                            | distributions)
            ``info_phs``  N/A               | A dict of placeholders for old values
                                            | of the entries in ``info``.
            ``d_kl``      ()                | A symbol for computing the mean KL
                                            | divergence between the current policy
                                            | (``pi``) and the old policy (as 
                                            | specified by the inputs to 
                                            | ``info_phs``) over the batch of 
                                            | states given in ``x_ph``.
            ``v``         (batch,)          | Gives the value estimate for states
                                            | in ``x_ph``. (Critical: make sure 
                                            | to flatten this!)
            ============  ================  ========================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to TRPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        delta (float): KL-divergence limit for TRPO / NPG update. 
            (Should be small for stability. Values like 0.01, 0.05.)

        vf_lr (float): Learning rate for value function optimizer.

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        damping_coeff (float): Artifact for numerical stability, should be 
            smallish. Adjusts Hessian-vector product calculation:
            
            .. math:: Hv \\rightarrow (\\alpha I + H)v

            where :math:`\\alpha` is the damping coefficient. 
            Probably don't play with this hyperparameter.

        cg_iters (int): Number of iterations of conjugate gradient to perform. 
            Increasing this will lead to a more accurate approximation
            to :math:`H^{-1} g`, and possibly slightly-improved performance,
            but at the cost of slowing things down. 

            Also probably don't play with this hyperparameter.

        backtrack_iters (int): Maximum number of steps allowed in the 
            backtracking line search. Since the line search usually doesn't 
            backtrack, and usually only steps back once when it does, this
            hyperparameter doesn't often matter.

        backtrack_coeff (float): How far back to step during backtracking line
            search. (Always between 0 and 1, usually above 0.5.)

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        algo: Either 'trpo' or 'npg': this code supports both, since they are 
            almost the same.

    """

    # initialize logger and save it
    

    # initialize seed, and set tf and np
    
    # get the env function, observation dimensions, and action dimensions
    
    # Share information about action space with policy architecture
    
    # Inputs to computation graph
    
    # Main outputs from computation graph, plus placeholders for old pdist (for KL)
    
    # Need all placeholders in *this* order later (to zip with data from buffer)
    
    # Every step, get: action, value, logprob, & info for pdist (for computing kl div)
    
    # Experience buffer
    # calculate the number of steps per epoch per process
    

    # get the info shapes
    

    # initialize the bugger
    

    # Count variables
    
    # TRPO losses
    # ratio of pi / pi_old
    # pi loss
    # v loss
    
    # Optimizer for value function
    

    # Symbols needed for CG solver
    # pi params
    # gradient
    # v_ph and hvp
    
    # check if the damping coeff is needed
    # if so, update hvp (damping_coeff * v_ph)
    
    # Symbols for getting and setting params
    # get pi params
    # set pi params
    
    # create a tf session and initialize it's variables
    
    # Sync params across processes
    

    # Setup model saving
    

    def cg(Ax, b):
        """
        Conjugate gradient algorithm
        (see https://en.wikipedia.org/wiki/Conjugate_gradient_method)
        """

        # initialize x as 0s of shape b
    

        # Note: should be 'b - Ax(x)', but for x=0, Ax(x)=0. Change if doing warm start.
        # make a copy of b and r as r and p
    
        # calculate r dot old (r dot r)
    

        # for cg_iterations
    

            # calc z as Ax(p)
    

            # calculate alpha 
            

            # increment x
      

            # decrement r

            # calculate r dot new (r dot r)

            # calculate p

            # update r dot old with r dot new

    def update():
        # Prepare hessian func, gradient eval
        # get inputs as a dictionary, all phs and buffer

        # calculate Hx

        # get g, pi_l_old, v_l_old

        # get g and pi_l_old averages

        # Core calculations for TRPO or NPG
        # get x

        # get alpha

        # get old paramers

        def set_and_eval(step):
            # set pi params with v_ph
            # old_params - alpha * x * step

            # return average of d_kl and pi_loss operation

        # handle npg
        
            # npg has no backtracking or hard kl constraint enforcement

        # handle trpo
        
            # trpo augments npg with backtracking line search, hard kl
            # for backtrack iterations
        
        # Value function updates
        # for train_v_iterations
        
        # update v_l_new with v_loss operation
        

        # Log changes from update
        
    # Update start time

    # reset variables
    # o, r, d, ep_ret, ep_len

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            # get agent outputs

            # decontruct the above to a, v_t, logp_t, info_t

            # save and log


            # take an action

            # update ep rewards and length
          

            # check if the episode is done

            # check if terminal or at max t for local epoch
            
                # if trajectory didn't reach terminal state, bootstrap value target

                # add the finish path to buffer

                    # only save EpRet / EpLen if trajectory finished

                # reset environment variables
                # o, r, d, ep_ret, ep_len

        # Save model
      

        # Perform TRPO or NPG update!

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('KL', average_only=True)
        if algo=='trpo':
            logger.log_tabular('BacktrackIters', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='trpo')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    trpo(lambda : gym.make(args.env), actor_critic=core.mlp_actor_critic,
         ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
         seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
         logger_kwargs=logger_kwargs)