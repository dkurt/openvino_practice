#!/usr/bin/env python3
import reinforcement_learning


def test():
    agent = reinforcement_learning.Agent(0, 0)
    agent.load(reinforcement_learning.WEIGHTS_PATH)
    reinforcement_learning.final_test(agent)


if __name__ == '__main__':
    test()
