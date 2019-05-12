
import os
import numpy as np
import gym
from gym import wrappers
import pybullet_envs
# theta 4x24, input 24x4, output 1x4
env = gym.make("BipedalWalker-v2")

num_deltas = 3
theta = np.random.random((env.action_space.shape[0], env.observation_space.shape[0]))
input = env.reset()

delta =  np.random.random(np.shape(theta))

deltas = [np.random.random(np.shape(theta)) for k in range(num_deltas)]

def evaluate(theta,delt, sign):
	if sign == "+":
		return (theta + delt).dot(input)
	elif sign == "-":
		return (theta + delt).dot(input)

rollout = {}
for i in range(num_deltas):
	rollout[i] = [0,0]
env.reset()
for k in range(num_deltas):
	state, reward, done, _ = env.step(evaluate(theta, deltas[k], "+"))
	rollout[k][0] = reward
	state, reward, done, _ = env.step(evaluate(theta, deltas[k], "-"))
	rollout[k][1] = reward
print(rollout)

