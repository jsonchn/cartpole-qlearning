# CartPole Q-learning 

This project implements Q-learning, a type of reinforcement learning algorithm, to solve the classic CartPole problem using the OpenAI Gym library. 

## Problem Description

The CartPole problem is a classic reinforcement learning problem where a pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart, and the goal is to prevent the pole from falling over and to keep the cart within the boundaries of the environment. 

The original problem has a continuous state space, but in this implementation, I discretize the state space to use tabular Q-learning.

## Implementation

The implementation includes an epsilon-greedy exploration strategy and a Q-value update rule as part of the Q-learning algorithm. The epsilon-greedy strategy ensures that the agent balances exploration and exploitation throughout its learning process, while the Q-value update rule allows the agent to incrementally improve its policy.
