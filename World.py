import pandas as pd
import numpy as np


class World:
    def __init__(self):
        self.n = 0  # number of states
        self.reward = {}
        self.states = []
        self.terminal_states = set()
        self.number_directions = 0
        self.directions_list = ['N', 'E', 'S', 'W']
        self.directions = {}
        self.m = 0  # number of edges
        self.world_map = {}
        self.gamma = 0  # discount
        self.start_state = ''

    def build_world(self, name):
        # read from text file
        file = open(name, "r")
        self.n = int(file.readline())
        for i in range(self.n):
            reward = file.readline()
            reward = reward.rstrip('\n')
            split = reward.split(' ')
            self.reward.update({split[0]: int(split[1])})
        terminal = file.readline()
        terminal = terminal.rstrip('\n')
        split = terminal.split(' ')
        for m in range(len(split)):
            self.terminal_states.add(split[m])
        self.number_directions = int(file.readline())
        for j in range(self.number_directions):
            direction = file.readline()
            direction = direction.rstrip('\n')
            split = direction.split(' ')
            self.directions.update({split[0]: [float(split[1])]})
            for k in range(2, self.number_directions + 1):
                self.directions[split[0]].append(float(split[k]))
        self.m = int(file.readline())
        for j in range(self.m):
            trans = file.readline()
            trans = trans.rstrip('\n')
            split = trans.split(' ')
            self.world_map.update({split[0] + split[1]: split[2]})
        self.gamma = float(file.readline())
        self.start_state = file.readline()
        file.close()
        for words in self.reward:
            if words not in self.terminal_states:
                self.states.append(words)
        return

    # return a reward for a given state
    def R(self, state):
        x = self.reward.get(state)
        return self.reward.get(state)

    def T(self, state, action):
        action_take = [self.world_map[state + 'N'], self.world_map[state + 'E'], self.world_map[state + 'S'], self.world_map[state + 'W']]
        return self.directions.get(action), action_take

    def actions(self, state):
        if state in self.terminal_states:
            return [None]
        else:
            return self.directions_list

def value_iteration(mdp, epsilon=0.001):
    U1 = mdp.reward.copy()
    D1 = {s: ' ' for s in mdp.reward}
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    eps_gam = epsilon * (1 - gamma) / gamma
    while True:
        U = U1.copy()
        D = D1.copy()
        delta = float(0)
        for s in mdp.states:
            temp = []
            for a in mdp.actions(s):
                p, sl = T(s,a)
                temp.append(p[0] * U[sl[0]] + p[1] * U[sl[1]] + p[2] * U[sl[2]] + p[3] * U[sl[3]])
            U1[s] = R(s) + gamma * max(temp)
            D1[s] = getDirection(temp.index(max(temp)))
            delta = max(delta, abs(U1[s] - U[s]))
        if delta <= eps_gam:
            return U,D

def getDirection(index):
    if index == 0:
        return 'N'
    elif index == 1:
        return 'E'
    elif index == 2:
        return 'S'
    elif index == 3:
        return 'W'
    else:
        return IndexError


def main():
    obsidian = World()
    obsidian.build_world('simple_g09_r0.txt')
    print(value_iteration(obsidian, 0.001))
    return
if __name__ == '__main__':
    main()
