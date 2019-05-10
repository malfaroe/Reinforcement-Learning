import gym
import numpy as np 

env = gym.make("Pendulum-v0")
num_deltas = 3

print (env.observation_space, env.action_space)
theta = np.zeros((env.action_space.shape[0], env.observation_space.shape[0]))
deltas = [np.random.randn(*theta.shape) for k in range(num_deltas)]
print ("Hola a todos")
a = 56
for i in range(8):
	i += 2
print(deltas)
print(a)