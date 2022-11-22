# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import gym
import matplotlib.pyplot as plt
import moviepy.video.io.ImageSequenceClip
from stable_baselines3 import A2C
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoClip, ImageClip, concatenate_videoclips
from moviepy.video.io.bindings import mplfig_to_npimage

x = np.linspace(-2, 2, 200)
duration = .5


import gym
from gym import spaces

class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {"render.modes": ["human"]}

    def __init__(self, arg1, arg2, ...):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)

    def step(self, action):
        ...
        return observation, reward, done, info
    def reset(self):
        ...
        return observation  # reward, done, info can't be included
    def render(self, mode="human"):
        ...
    def close (self):
        ...

def make_env():
    env = gym.make("CartPole-v1")
    env.reset()
    return env

def basic_policy(obs):
    angle = obs[2]
    if angle < 0:
        return 0
    else:
        return 1

def do_naive(env):
    # Use a breakpoint in the code line below to debug your script.
    img = env.render()

    n_episodes = 1
    n_steps = 500

    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward_sum = 0
        imgs = []

        for step in range(n_steps):
            action = basic_policy(obs)
            obs, reward, done, info = env.step(action)
            img = env.render(mode="rgb_array")
            episode_reward_sum += reward
            if done:
                break
            imgs += [ImageClip(img, duration=duration)]
            plt.imshow(img)
            plt.show()

        concat_clip = concatenate_videoclips(imgs, method="compose")
        concat_clip.write_videofile(f"episode_{episode}.mp4", fps=24)
    i = 2

import tensorflow as tf
from tensorflow import keras 

# Press the green button in the gutter to run the script.
def train_stable_baselines(env):
    model_file = Path("a2c_cartpole.zip")
    if model_file.is_file():
        model = A2C.load("a2c_cartpole")
    else:
        model = A2C("MlpPolicy", env, verbose=0)
        print("Starting training ...")
        model.learn(total_timesteps=15_000)
        print("Training done ... ")
        model.save("a2c_cartpole")

    obs = env.reset()
    for i in range(500):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        img = env.render(mode="rgb_array")
        plt.imshow(img)
        plt.show()
        if done:
            print(f"We are done after {i} steps")
            obs = env.reset()
            break


if __name__ == '__main__':
    env = make_env()
    #do_naive(env)
    #train_stable_baselines(env)
    #policy_gradients(env)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
