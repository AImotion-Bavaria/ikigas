# testing a new kind of environment - a simple gas container
# able to control the inflow valve
import gym
import numpy as np
from gym import  spaces


import numpy as np

def labeling_function(x):
    """
    simply returns 50 as the best value and -1 for 0 and 100
    :param x:
    :return:
    """
    CENTER = 50.0
    return -1. + 2*(CENTER - np.abs(x-CENTER) ) / CENTER

class SimpleGasEnvironment(gym.Env):
    metadata = {'render.modes': ['console']}

    # Constants to make code readable
    UP = 0
    NOTHING = 1
    DOWN = 2

    INIT_VALVE = 0.7
    DELTA_VALVE = 0.01
    DT = 0.1

    MIN_LINEPACK = 0.0
    MAX_LINEPACK = 100.0

    def __init__(self, in_flow, out_flow):
        super(SimpleGasEnvironment, self).__init__()

        # define action and observation space for gym.spaces
        self.action_space = spaces.Discrete(3) # represents a nudge up or down
        self.observation_space = spaces.Box(low=np.array([self.MIN_LINEPACK, 0.0]),
                                            high=np.array([self.MAX_LINEPACK, 1.0]), shape=(2,), dtype=np.float32)

        # initially at the half
        self.linepack = 50.0

        # the valve is initially half open
        self.valve_position = self.INIT_VALVE

        # inflow is assumed to be constant and can be passed via constructor
        self.in_flow = in_flow

        # for now outflow is also constant, but this will be random function
        self.out_flow = out_flow

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        # "Linepack" refers to the volume of gas that can be "stored" in a gas pipeline.
        self.linepack = 60.0
        self.valve_position = self.INIT_VALVE

        # where am I will be a numpy array (1,) containing the position as float
        return np.array([self.linepack, self.valve_position]).astype(np.float32)

    def step(self, action):

        #print("Action picked: ", action)
        if action == self.DOWN:
            self.valve_position -= self.DELTA_VALVE
        elif action == self.NOTHING:
            pass
        elif action == self.UP:
            self.valve_position += self.DELTA_VALVE
        else:
            raise ValueError(f"Received action {action} which is not part of the action space")

        self.valve_position = np.clip(self.valve_position, 0.0, 1.0)

        # the actual dynamics go right here
        last_linepack = self.linepack
        self.linepack = self.linepack + self.DT * (self.in_flow * self.valve_position - self.out_flow )

        done = bool(self.linepack < self.MIN_LINEPACK or self.linepack > self.MAX_LINEPACK)

        # reward is issued according
        reward = labeling_function(self.linepack)

        # an empty set of info
        info = {}
        info["TimeLimit.truncated"] = True
        return np.array([self.linepack, self.valve_position]).astype(np.float32), reward, done, info

    def render(self, mode="console"):
        if mode != "console":
            raise NotImplementedError()
        # now just some pretty prints

        print("Line pack", self.linepack)

    def close(self):
        # nothing to do yet
        pass