import pandas as pd
import numpy as np


class World:
    def __init__(self):
        self.n = 0  # number of states
        self.rewards = {}
        self.states = {}
        self.terminal_states = set()
        self.number_directions = 0
        self.directions_list = ['N', 'S', 'E', 'W']
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
            state = file.readline()
            state = state.rstrip('\n')
            split = state.split(' ')
            self.states.update({split[0]: int(split[1])})
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
        return

    # return a reward for a given state
    def R(self, state):
        return self.rewards.get(state)

    def T(self, state, action):
        return self.directions[1], self.world_map[state + action]

    def actions(self, state):
        if state in self.terminal_states:
            return [None]
        else:
            return self.directions_list

def value_iteration(mdp, epsilon=0.001):
    U1 = {s: 0 for s in mdp.states}
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    while True:
        U = U1.copy()
        delta = 0
        for s in mdp.states:
            U1[s] = R(s) + gamma * max([sum([p * U[s1] for (p, s1) in T(s, a)]) for a in mdp.actions(s)])
            delta = max(delta, abs(U1[s] - U[s]))
        if delta < epsilon * (1 - gamma) / gamma:
            return U


def main():
    obsidian = World()
    obsidian.build_world('simple_g09_r0.txt')
    print(value_iteration(obsidian, 0.001))
    return
if __name__ == '__main__':
    main()
