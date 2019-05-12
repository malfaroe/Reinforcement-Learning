# My implementation of Augmented Random Search Algorithm on Openai BipedalWalker env

import os
import numpy as np
import gym
from gym import wrappers
import pybullet_envs

class Hp():
	#Definition of hyperparameters
	def __init__(self, nb_steps = 100, 
		episode_length = 2000,
		learning_rate = 0.02, num_deltas = 16, 
		num_best_deltas = 16, noise = 0.03, seed= 1 ,
		 env_name = "BipedalWalker-v2"):
		
		self.nb_steps = nb_steps
		self.episode_length = episode_length
		self.learning_rate = learning_rate
		self.num_deltas = num_deltas
		self.num_best_deltas = num_best_deltas
		self.noise = noise
		self.seed = seed
		self.env_name = env_name

# Environment
env = gym.make("BipedalWalker-v2")


class  Normalizer():
	# Normalizer of inputs:
	def __init__(self, nb_inputs):
		self.n = np.zeros(nb_inputs)
		self.mean = np.zeros(nb_inputs)
		self.mean_diff = np.zeros(nb_inputs)
		self.var = np.zeros(nb_inputs)


	def observe(self, x):
		self.n += 1.0
		last_mean = self.mean.copy()
		self.mean += (x - self.mean) / self.n 
		self.var = (self.mean_diff / self.n).clip(min = 1e-2)
		

	def normalize(self, inputs):
		obs_mean = self.mean
		obs_std = np.sqrt(self.var)
		return (inputs - obs_mean) / obs_std




class Trainer():
	def __init__(self,
				 hp=None,
				 normalizer=None, deltasWork = None):

		self.hp = hp or Hp()
		np.random.seed(self.hp.seed)
		self.env = gym.make(self.hp.env_name)
		self.normalizer = normalizer or Normalizer(env.observation_space.shape[0])
		self.theta = np.zeros((env.action_space.shape[0], env.observation_space.shape[0]))

	def run_episode(self, weights):
		state = env.reset()
		done = False
		acum_rewards = 0
		num_plays = 0
		while  not done and num_plays < self.hp.episode_length:
			self.normalizer.observe(state)
			state = self.normalizer.normalize(state)
			action = (weights).dot(state)
			state, reward, done,_ = env.step(action)
			reward = max(min(reward, 1), -1)
			acum_rewards += reward
			num_plays += 1
		return acum_rewards

	def train(self): #Main algorithm
		
		for i in range(self.hp.nb_steps):
			rew_acum = 0
			# Generates the deltas:
			deltas = [np.random.randn(*self.theta.shape) for k in range(self.hp.num_deltas)]
			# Processing deltas:
			pos_rewards = [0] * self.hp.num_deltas
			neg_rewards = [0] * self.hp.num_deltas
			for k in range(self.hp.num_deltas):
				pos_rewards[k] = self.run_episode(self.theta + self.hp.noise * deltas[k])
				neg_rewards[k] = self.run_episode(self.theta - self.hp.noise * deltas[k])
			sigma_rewards = np.std(pos_rewards + neg_rewards)
			scores = {k:max(r_pos, r_neg) for k,(r_pos,r_neg) in enumerate(zip(pos_rewards, neg_rewards))}
			order = sorted(scores.keys(), key = lambda x:scores[x], reverse = True)[:self.hp.num_deltas]
			sorted_rolls = [(pos_rewards[k], neg_rewards[k], deltas[k]) for k in order]
	  

		# Step for updating:
			step = np.zeros(self.theta.shape)
			for r_pos, neg_pos, deltas in sorted_rolls:
				step += (r_pos - neg_pos) * deltas
		   

		# Updates theta:
			self.theta += self.hp.learning_rate /(self.hp.num_deltas * sigma_rewards) * step
			rew_acum += self.run_episode(self.theta)
			print("Paso {} : {}").format(i, rew_acum)



# Main Ars code:
if __name__ == "__main__":
	hp = Hp(seed=1946)
	trainer = Trainer(hp = hp)
	trainer.train()




















	





				

		




		