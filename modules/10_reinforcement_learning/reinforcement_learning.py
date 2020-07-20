import itertools
import os

import numpy as np
from gym import wrappers
from gym.envs import toy_text

import test_reinforcement_learning


WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'q_values.npy')

def try_gym():
    # Jump to the FrozenLakeEnv definition
    # (https://github.com/openai/gym/blob/345c65973fc7160d8be374745a60c36869d8accc/gym/envs/toy_text/frozen_lake.py#L71)
    # to see its descrition
    env = wrappers.TimeLimit(toy_text.FrozenLakeEnv(map_name='8x8'), max_episode_steps=1000)
    print('action_space size:', env.action_space.n)
    print('random action sample:', env.action_space.sample())
    print('observation_space size:', env.observation_space.n)
    for episode_id in range(1):
        score = 0.0
        observation = env.reset()
        print('observation:', observation)
        # If render() prints colorcodes instead of coloring a simbol on Windows and you are a perfectionist, you can use
        # https://aka.ms/terminal if your Windows version is high enough
        env.render()
        for step_id in itertools.count():
            observation, reward, done, diagnostic_info = env.step(env.action_space.sample())
            # env.render()
            score += reward
            if done: break
        print('score:', score)  # It will probaly be 0


class Agent:
    ALPHA = 0.1
    GAMMA = 0.999

    def __init__(self, observation_space_size, action_space_size):
        self.q_values = np.zeros((observation_space_size, action_space_size), np.float32)

    def load(self, fileName):
        self.q_values = np.load(fileName)

    def save(self, file_name):
        np.save(file_name, self.q_values)

    def best_action(self, observation):
        """Return best action given obesrvation according to current stored q_values"""
        print('best_action is not implemented')
        exit(1)

    def update_q_values(self, observation, action, reward, next_observation, done):
        """Update stored q_values with provided data following Bellman equation.

        The updated value is a weighted sum of the old value taken with 1.0 - ALPHA coefficient and the new one with
        ALPHA. The new value is a sum of reward recieved and what the agent thinks about the next obesrvation discounted
        by GAMMA like if it played the best action.
        
        If done==True the update doesn't care about next_observation.
        """
        print('update_q_values is not implemented')
        exit(1)


def check_agent_actions(agent):
    assert np.issubdtype(agent.best_action(0).dtype, np.integer)
    assert np.issubdtype(agent.best_action(15).dtype, np.integer)
    assert 0 <= agent.best_action(1) <= 3


def check_agent_update():
    agent = Agent(16, 4)
    assert (agent.q_values == 0).all()
    agent.update_q_values(0, 0, 9e9, 0, True)
    assert agent.q_values[0, 0] > 0


def train():
    UPDATE_PERIOD = 10000
    EPSILON_DECAY = 0.7
    env = wrappers.TimeLimit(toy_text.FrozenLakeEnv(map_name='8x8'), max_episode_steps=1000)

    print('agent is not created')
    exit(1)
    agent = None

    check_agent_actions(agent)
    check_agent_update()

    epsilon = 1.0
    last_scores_sum = 0.0
    for episode_id in itertools.count():
        score = 0.0
        observation = env.reset()
        while True:

            print('perform a random action with probability of epsilon. Request best action from agent otherwise')
            exit(1)
            action = None

            next_observation, reward, done, _ = env.step(action)
            agent.update_q_values(observation, action, reward, next_observation, done)
            score += reward
            if done:
                break
            observation = next_observation

        last_scores_sum += score
        if episode_id > 0 and episode_id % UPDATE_PERIOD == 0:
            epsilon *= EPSILON_DECAY
            last_mean_score = last_scores_sum / UPDATE_PERIOD
            print('last_mean_score:', last_mean_score)
            if last_mean_score > 0.999:
                print('You won!')
                break
            else:
                last_scores_sum = 0.0
    return agent


def main():
    try_gym()
    agent = train()
    agent.save(WEIGHTS_PATH)


if __name__ == '__main__':
    main()
