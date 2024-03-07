import json
import random
import time

import numpy as np
import torch

from dqn.state_container import CacheContainer

import dqn.test_dqn_agent as net


from dqn.replay_buffer import ReplayBuffer

num_episodes = 5000
buffer_size = 1000000

minimal_size = 1000
batch_size = 512
update_interval = 8
# 1， 3， 5

class ENV:
    # 初始所有的state都是1，每3个state1组，当选中1个action时，对应state+（1~3），每个state都有自己的reward计算公式和终止条件
    # reward计算公式为sigma(state_i * wi)

    def __init__(self):
        super().__init__()
        self.state = [0 for i in range(30)]
        # 每次选择后state增加的值
        self.add = [random.randint(1,3) for i in range(10)]
        # 设置终止条件
        self.end = [random.randint(10, 100) for i in range(30)]
        self.weight = [random.randint(-10,10) for i in range(30)]
        self.last_reward = 0
        print(self.add)
        print(self.end)
        print(self.weight)

    def getReward(self, actions:[int]):
        sum = 0
        for action in actions:
            sum += self.weight[action*3 ] * self.add[int((actions[0] + actions[1] + actions[2]) / 3)]
            sum += self.weight[action*3 + 1] * self.add[int((actions[0] + actions[1] + actions[2]) / 3)]
            sum += self.weight[action*3 + 2] * self.add[int((actions[0] + actions[1] + actions[2]) / 3)]
        return sum

    def getTotalReward(self):
        sum = 0
        for i in range(30):
            sum += self.state[i] * self.weight[i]
        return sum

    def next(self, actions:[int]):
        for action in actions:
            self.state[action * 3] += self.add[int((actions[0] + actions[1] + actions[2]) / 3)]
            self.state[action * 3 + 1] += self.add[int((actions[0] + actions[1] + actions[2]) / 3)]
            self.state[action * 3 + 2] += self.add[int((actions[0] + actions[1] + actions[2]) / 3)]
        for i in range(30):
            if self.state[i] >= self.end[i]:
                return self.state.copy(), self.getReward(actions), 1
        return self.state.copy(), self.getReward(actions), 0

    def reset(self):
        self.state = [0 for i in range(30)]


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    total_step = 0

    replay_buffer = ReplayBuffer(buffer_size, batch_size)
    agent = net.DDQN(device, 30, 10, "_1_", epsilon=0.05)
    env = ENV()
    env.reset()
    results = []
    for epoch in range(num_episodes):
        while True:
            state = env.state.copy()
            actions = agent.take_action(state)
            next_state, reward, done = env.next(actions)
            replay_buffer.add(state, actions, reward, next_state, done)

            if done == 1:
                if replay_buffer.size(
                ) >= minimal_size and epoch % int(update_interval) == 0:
                    sample = replay_buffer.sample()
                    agent.update(sample)
                print(env.state)
                print(env.getTotalReward())
                results.append(env.getTotalReward())
                env.reset()
                break
    with open("test", "w") as file:
        file.write(json.dumps(results))

