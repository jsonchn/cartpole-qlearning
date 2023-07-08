# CartPole Q-learning 

This project implements Q-learning, a type of reinforcement learning algorithm, to solve the classic CartPole problem using the OpenAI Gym library. 

## Problem Description

The CartPole problem is a classic reinforcement learning problem where a pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart, and the goal is to prevent the pole from falling over and to keep the cart within the boundaries of the environment. 

The original problem has a continuous state space, but in this implementation, I discretize the state space to use tabular Q-learning.

## Implementation

The implementation includes an epsilon-greedy exploration strategy and a Q-value update rule as part of the Q-learning algorithm. The epsilon-greedy strategy ensures that the agent balances exploration and exploitation throughout its learning process, while the Q-value update rule allows the agent to incrementally improve its policy.


### Prerequisites

Before running this code, you will need to install the required libraries. You can do this by running 
`pip install -r requirements.txt`. 

### Usage

You can run the `qlearning.py` script with the following command-line options:

* `-h`: Display a help message.
* `--algorithm {random, qlearning}`: Choose the algorithm to use. Use `random` for a baseline policy that just chooses a random action at each timestep, and `qlearning` to use the Q-learning algorithm.
* `--num-episodes NUM_EPISODES`: The number of episodes to run. 
* `--epsilon EPSILON`: The epsilon value for epsilon-greedy exploration in the Q-learning algorithm.

Some example uses:
* Run the baseline policy that just chooses a random action at each timestep:
`python3 qlearning.py cartpole --algorithm random`
* Run Q-learning with 10,000 episodes and an ϵ of 0.1 for ϵ-greedy exploration:
`python3 qlearning.py cartpole --algorithm qlearning --num-episodes 10000 --epsilon 0.1`
* Run Q-learning with 10,000 episodes without ϵ-greedy, i.e. ϵ = 0:
`python3 qlearning.py cartpole --algorithm qlearning --num-episodes 10000 --epsilon 0.0`

## Acknowledgments

* Thanks to [OpenAI](https://openai.com) for the [Gym library](https://gym.openai.com). The CartPole environment used in this project can be found [here](https://gymnasium.farama.org/environments/classic_control/cart_pole/).
