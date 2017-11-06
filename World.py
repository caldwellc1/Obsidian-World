import pandas as pd
import numpy as np


class World:
    def __init__(self, start):
        self.n = 0  # number of states
        self.rewards = {}
        self.terminal_states = []
        self.number_directions = 0
        self.directions = {}
        self.m = 0  # number of edges
        self.world_map = {}
        self.gamma = 0
        self.start_state = start

    def build_world(self, name):
        df = pd.read_csv(name)
        return


def main():
    return
if __name__ == '__main__':
    main()
