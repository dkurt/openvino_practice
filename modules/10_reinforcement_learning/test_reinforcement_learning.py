#!/usr/bin/env python3
from gym import wrappers
from gym.envs import toy_text

import reinforcement_learning


def test(agent):
    EPISODE_NUM = 1000
    env = wrappers.TimeLimit(toy_text.FrozenLakeEnv(map_name='8x8'), max_episode_steps=1000)
    score_sum = 0.0
    for _ in range(EPISODE_NUM):
        observation = env.reset()
        while True:
            observation, reward, done, _ = env.step(agent.best_action(observation))
            score_sum += reward
            if done:
                break
    assert score_sum / EPISODE_NUM >= 0.99


def main():
    agent = reinforcement_learning.Agent(0, 0)
    agent.load(reinforcement_learning.WEIGHTS_PATH)
    test(agent)


if __name__ == '__main__':
    main()
