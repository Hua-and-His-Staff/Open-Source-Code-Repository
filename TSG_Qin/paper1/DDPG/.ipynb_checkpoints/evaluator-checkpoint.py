
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

from util import *

class Evaluator(object):

    def __init__(self, num_episodes, num_values):
        self.num_episodes = num_episodes


    def __call__(self, env, policy, debug=False):

        self.is_training = False
        observation = None
        np.random.seed(100)
        result = np.zeros([self.num_episodes,17])

        for episode in range(self.num_episodes):

            # reset at the start of episode
            observation = env.reset()
            episode_steps = 0  
            
            assert observation is not None
            #print(observation)

            # start episode
            done = False
            while not done:
                # basic operation, action ,reward, blablabla ...
                action = policy(observation)
                observation, reward, done, info = env.step(action)
                #if self.max_episode_length and episode_steps >= self.max_episode_length -1:
                #    done = True
                # update
                result[episode]+=reward
                episode_steps += 1
            

        #result.shape: num_episodes*(2+5+10)
        return result

