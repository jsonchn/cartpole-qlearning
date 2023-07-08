import argparse
import sys

import gymnasium as gym
import matplotlib.pyplot as plt
import random
import numpy as np
import time

ENV_DICTS = {
        'cartpole': {
                'name': 'CartPole-v1',
                'short_name': 'cartpole',
                'num_actions': 2,
                'num_bins': 10,
                'num_dims': 4,
                'bounds': [[-3.0, 3.0], [-3.0, 3.0], [-.3, .3], [-.3, .3]]
        }
}
for e in ENV_DICTS.values():
    e['num_states'] = (e['num_bins'] + 1)**e['num_dims']
    e['bins'] = [np.linspace(x[0], x[1], e['num_bins']) for x in e['bounds']]

OPTS = None

def get_discrete_state(env_dict, state):
    idx = 0
    for i in range(env_dict['num_dims']):
        idx = (env_dict['num_bins'] + 1) * idx + np.digitize(state[i], env_dict['bins'][i])
    return idx

def run_qlearning(env, env_dict, num_episodes=10000, max_steps=500, window_size=100, epsilon=0.1, lr=0.1, discount=0.99):
    q_values = np.zeros((env_dict['num_states'], env_dict['num_actions']), dtype=np.float64)
    cur_window_rewards = 0.0
    reward_curve = []

    for episode in range(num_episodes):
        episode_reward = 0.0
        continuous_state, info = env.reset()  # Reset environment, get starting state
        state = get_discrete_state(env_dict, continuous_state)

        for t in range(max_steps):
            # TODO: Implement epsilon-greedy exploration
            # With probability epsilon, choose a random action
            # Otherwise, use q_values to choose the best action at the current state
            ### BEGIN_SOLUTION 2b
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_values[state])
            ### END_SOLUTION 2b

            # Take the chosen action, discretize the new state, record the reward
            new_continuous_state, reward, terminated, truncated, info = env.step(action)
            new_state = get_discrete_state(env_dict, new_continuous_state)
            episode_reward += reward

            # TODO: Do a Q-learning update to q_values
            ### BEGIN_SOLUTION 2b
            q_values[state, action] += lr * (reward + discount * np.max(q_values[new_state]) - q_values[state, action])
            ### END_SOLUTION 2b

            # Check if episode has terminated
            if terminated or truncated:
                break

            # Update state
            state = new_state

        cur_window_rewards += episode_reward
        # Print the average reward from every |window_size| episodes
        if (episode + 1) % window_size == 0:
            cur_avg_reward = cur_window_rewards / window_size
            reward_curve.append(cur_avg_reward)
            print(f'Episode {episode+1}: average_reward={cur_avg_reward}')
            cur_window_rewards = 0.0

    # Plot the reward over time
    plt.plot(np.arange(len(reward_curve)) * window_size, reward_curve)
    plt.xlabel('Number of Episodes Elapsed')
    plt.ylabel('Average Reward')
    plt.savefig(f'reward_{env_dict["short_name"]}_eps{epsilon:.2f}.png')

    return q_values

def run_test(env, env_dict, algorithm, q_values, num_episodes=20, max_steps=500):
    all_rewards = []

    for episode in range(num_episodes):
        episode_reward = 0.0
        continuous_state, info = env.reset()  # Reset environment, get starting state
        state = get_discrete_state(env_dict, continuous_state)

        for t in range(max_steps):
            if algorithm == 'random':
                action = env.action_space.sample()
            else:
                ### BEGIN_SOLUTION 2b
                action = np.argmax(q_values[state])
                ### END_SOLUTION 2b

            new_continuous_state, reward, terminated, truncated, info = env.step(action)
            new_state = get_discrete_state(env_dict, new_continuous_state)
            episode_reward += reward

            if terminated or truncated:
                break
            state = new_state

        all_rewards.append(episode_reward)
        print(f'Episode {episode}: reward={episode_reward}')

    print(f'Average reward: {np.mean(all_rewards):.2f}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('environment', choices=ENV_DICTS.keys())
    parser.add_argument('--algorithm', '-a', choices=['random', 'qlearning'])
    parser.add_argument('--num-episodes', '-n', type=int, default=10000)
    parser.add_argument('--epsilon', '-e', type=float, default=0.0)
    parser.add_argument('--learning-rate', '-r', type=float, default=0.1)
    return parser.parse_args()

def main():
    random.seed(0)
    env_dict = ENV_DICTS[OPTS.environment]
    q_values = None
    if OPTS.algorithm == 'qlearning':
        env = gym.make(env_dict['name'])  # No render during training
        q_values = run_qlearning(env, env_dict, num_episodes=OPTS.num_episodes,
                                 epsilon=OPTS.epsilon, lr=OPTS.learning_rate)
    env = gym.make(env_dict['name'], render_mode='human')
    run_test(env, env_dict, OPTS.algorithm, q_values=q_values)

if __name__ == '__main__':
    OPTS = parse_args()
    main()

