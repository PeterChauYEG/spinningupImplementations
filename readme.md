# Spinning up implemenation
=========================

This is my implemenations of the algorithms from the OpenAI spinning up module. Everything is from that repo. I've gutted it and am trying to understand the algorithms better by reimplementing them.

## Running VPG:

For Discrete action space'd environments such as CartPole-v1:
`$ python -m spinupImplementation.run vpgImplementation --exp_name test_vpgImplementation --env CartPole-v1`

For Box action space'd environments such as HalfCheetah-v2:
`$ python -m spinupImplementation.run vpgImplementation --exp_name vpgHalfCheetah --env HalfCheetah-v2`

## Running TRPO:

For Discrete action space'd environments such as CartPole-v1:
`$ python -m spinupImplementation.run trpoImplementation --exp_name test_trpoImplementation --env CartPole-v1`

For Box action space'd environments such as HalfCheetah-v2:
`$ python -m spinupImplementation.run trpoImplementation --exp_name trpoHalfCheetah --env HalfCheetah-v2`


====================================================================================

Welcome to Spinning Up in Deep RL! 
==================================

This is an educational resource produced by OpenAI that makes it easier to learn about deep reinforcement learning (deep RL).

For the unfamiliar: [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) (RL) is a machine learning approach for teaching agents how to solve tasks by trial and error. Deep RL refers to the combination of RL with [deep learning](http://ufldl.stanford.edu/tutorial/).

This module contains a variety of helpful resources, including:

- a short [introduction](http://spinningup.openai.com/en/latest/spinningup/rl_intro.html) to RL terminology, kinds of algorithms, and basic theory,
- an [essay](http://spinningup.openai.com/en/latest/spinningup/spinningup.html) about how to grow into an RL research role,
- a [curated list](http://spinningup.openai.com/en/latest/spinningup/keypapers.html) of important papers organized by topic,
- a well-documented [code repo](https://github.com/openai/spinningup) of short, standalone implementations of key algorithms,
- and a few [exercises](http://spinningup.openai.com/en/latest/spinningup/exercises.html) to serve as warm-ups.

Get started at [spinningup.openai.com](http://spinningup.openai.com)!