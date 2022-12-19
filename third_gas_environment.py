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
    CENTER = 0.5
    res = -1 + 2*(CENTER - np.abs(x-CENTER)) / CENTER
    if x >= 0.45 and x <= 0.55:
        return 5
    else:
        return res

x = np.linspace(0, 1, 1000)
#plt.plot(x, labeling_function(x))

def out_flow_function (x):
    fq = 2
    y1 = 0.029 * np.sin(2*np.pi*fq*x) + 0.2
    y2 = 0.028 * np.sin(2*np.pi*2*fq*x)
    y = y1 + y2
    return y


class SimpleGasEnvironment(gym.Env):
    metadata = {'render.modes': ['console']}

    # Constants to make code readable
    DOWN = 0
    NOTHING = 1
    UP = 2

    MIN_VALVE = 0.0
    MAX_VALVE = 1.0
    INIT_VALVE = 0.5
    DELTA_VALVE = 0.2
    DT = 0.01

    MIN_LINEPACK = 0.0
    MAX_LINEPACK = 1.0
    INIT_LINEPACK = 0.5

    TIME = 0
    MIN_OUTFLOW = 0.0
    MAX_OUTFLOW = 1.0

    def __init__(self, in_flow, randomize=True):
        super(SimpleGasEnvironment, self).__init__()
        self.randomize = randomize
        # define action and observation space for gym.spaces
        self.action_space = spaces.Discrete(3) # represents a nudge up or down
        self.observation_space = spaces.Box(low=np.array([self.MIN_LINEPACK, self.MIN_VALVE, self.MIN_OUTFLOW]),
                                            high=np.array([self.MAX_LINEPACK, self.MAX_VALVE, self.MAX_OUTFLOW]), shape=(3,), dtype=np.float32)


        self.linepack = self.INIT_LINEPACK
        if randomize:
            self.linepack = np.clip(self.linepack + np.random.normal(0, 0.3), 0.0, 1.0)

        self.valve_position = self.INIT_VALVE
        if randomize:
            self.valve_position = np.clip(self.valve_position + np.random.normal(0, 0.3), 0.0, 1.0)

        # inflow is assumed to be constant and can be passed via constructor
        self.in_flow = in_flow

        # for now outflow is also constant, but this will be random function
        #self.out_flow = out_flow

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """

        # "Linepack" refers to the volume of gas that can be "stored" in a gas pipeline.
        self.linepack = self.INIT_LINEPACK
        if self.randomize:
            self.linepack = np.clip(self.linepack + np.random.normal(0, 0.1), 0.0, 1.0)

        self.valve_position = self.INIT_VALVE
        if self.randomize:
            self.valve_position = np.clip(self.valve_position + np.random.normal(0, 0.1), 0.0, 1.0)

        # out_flow is issued according
        self.out_flow = out_flow_function(self.TIME)
        if self.randomize:
            self.out_flow = np.clip(self.out_flow + np.random.normal(0, 0.1), 0.0, 1.0)

        # where am I will be a numpy array (1,) containing the position as float
        return np.array([self.linepack, self.valve_position, self.out_flow]).astype(np.float32)

    def step(self, action):
        last_valve_position = self.valve_position
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
        self.linepack = self.linepack + self.DT * (self.in_flow * self.valve_position - self.out_flow)

        done = bool(self.linepack < self.MIN_LINEPACK or self.linepack > self.MAX_LINEPACK)

        # reward is issued according
        reward = labeling_function(self.linepack)

        # out_flow is issued according
        self.TIME += self.DT
        self.out_flow = out_flow_function(self.TIME)


        # an empty set of info
        info = {}
        #info["TimeLimit.truncated"] = True

        # use the temporal differences
        lp_dt = (self.linepack - last_linepack) / self.DT
        vp_dt = (self.valve_position - last_valve_position) / self.DT
        return np.array([self.linepack, self.valve_position, self.out_flow]).astype(np.float32), reward, done, info

    def render(self, mode="console"):
        if mode != "console":
            raise NotImplementedError()
        # now just some pretty prints

        print("Line pack", self.linepack)

    def close(self):
        # nothing to do yet
        pass