from itertools import count
from env import Env
import os, sys, random
import numpy as np
import pandas as pd
from collections import deque
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random
from utils import *


class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size, batch_size, data):
        self.buffer = deque(maxlen=max_size)
        self.batch_size = batch_size
        self.data = data
    def push(self, data):
        self.buffer.append(data)

    def sample(self):
        sample = random.sample(self.buffer, self.batch_size)
        index, states, next_states, actions, rewards = map(np.asarray, zip(*sample))
        inputs = [self.data[ind:ind+24] for ind in index]
        next_inputs = [self.data[ind+1:ind+25] for ind in index]
        return np.stack(inputs,axis=0), np.stack(next_inputs,axis=0), states, next_states, actions, rewards.reshape(-1,1)



class Actor(nn.Module):
    def __init__(self, input_size, state_size, action_size, prediction_size):
        super(Actor, self).__init__()

        self.RNN = nn.Sequential(
                                nn.Linear(input_size,128),
                                nn.Tanh(),
                                nn.Linear(128,128),
                                nn.Tanh(),
                                nn.GRU(128, 64, batch_first=True))
        self.MLP = nn.Sequential(
                                 nn.Linear(72,128),
                                 nn.Tanh(),
                                 nn.Linear(128,128),
                                 nn.Tanh(),
                                 nn.Linear(128,64),
                                 nn.Tanh())
        self.output = nn.Sequential(
                                 nn.Linear(64+64 + state_size,256),
                                 nn.Tanh(),
                                 nn.Linear(256,128),
                                 nn.Tanh(),
                                 nn.Linear(128,action_size),
                                 nn.Tanh())
        self.apply(network_initializer)


    def forward(self, Input, state):
        #not provide hidden(default zero)
        predictions,_ = self.RNN(Input)
        
        x = torch.cat([predictions[:,-1], self.MLP(Input[:,:,-3:].reshape(len(Input),-1)), state], dim=1)
        return self.output(x)
            


class Critic(nn.Module):
    def __init__(self, input_size, state_size, action_size, prediction_size):
        super(Critic, self).__init__()

        self.RNN = nn.Sequential(
                                nn.Linear(input_size,128),
                                nn.Tanh(),
                                nn.Linear(128,128),
                                nn.Tanh(),
                                nn.GRU(128,64, batch_first=True))
        self.MLP = nn.Sequential(
                                 nn.Linear(72,128),
                                 nn.Tanh(),
                                 nn.Linear(128,128),
                                 nn.Tanh(),
                                 nn.Linear(128,64),
                                 nn.Tanh())
        self.output = nn.Sequential(
                                nn.Tanh(),
                                 nn.Linear(128+state_size+action_size,256),
                                 nn.Tanh(),
                                 nn.Linear(256,128),
                                 nn.Tanh(),
                                 nn.Linear(128,1))
        self.apply(network_initializer)


    def forward(self, inputs, x, u):
        predictions,_ = self.RNN(inputs)
        x = torch.cat([predictions[:,-1], self.MLP(inputs[:,:,-3:].reshape(len(inputs),-1)), x,u], dim=1)
        return self.output(x)


class DDPG(object):
    def __init__(self, input_size, state_size, action_size, prediction_size, model_path, device):
        self.actor = Actor(input_size, state_size, action_size, prediction_size).to(device)
        self.actor_target = Actor(input_size, state_size, action_size, prediction_size).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic = Critic(input_size, state_size, action_size, prediction_size).to(device)
        self.critic_target = Critic(input_size, state_size, action_size, prediction_size).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.model_path = model_path
        if not os.path.exists(model_path):
            os.makedirs(model_path) 
    def save(self):
        torch.save(self.actor.state_dict(),  self.model_path+'actor_now.pth')
        torch.save(self.critic.state_dict(), self.model_path+'critic_now.pth')
    def save_as_best(self):
        torch.save(self.actor.state_dict(),  self.model_path+'actor_best.pth')
        torch.save(self.critic.state_dict(), self.model_path+'critic_best.pth') 
    def load(self):
        self.actor.load_state_dict(torch.load('./model/best_actor.pth'))
        self.critic.load_state_dict(torch.load('./model/best_critic.pth'))
        
        
def RL_RNN_train(args):
    data=pd.read_csv('./data/data_full.csv',index_col=0)
    env = Env(data[['generation','load','price']].values.astype(float))
    env_valid = Env(data[['generation','load','price']].values.astype(float))
    data = pd.read_csv('./data/data_for_predict.csv',index_col=0).values.astype(float)
    replay_buffer = Replay_buffer(int(args.capacity), args.batch_size, data)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    writer = SummaryWriter('./log/RNN{}/'.format(args.forward_steps))
    if args.seed:
        env.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        random.seed(args.random_seed)
    _, input_size = data.shape
    state_size = 1
    action_size = 2
    agent = DDPG(input_size = input_size, state_size=state_size, action_size = action_size, prediction_size = 20*args.forward_steps, model_path='./model/RNN{}/'.format(args.forward_steps), device = device)
    if args.load: agent.load()
    total_steps = 0
    reward_best = -1e10
    train_rewards = 0
    for epoch in range(int(args.max_epoch)):
        train_rewards = 0
        start, state = env.reset()
        #noise = max(args.exploration_noise - args.noise_decay*epoch, args.noise_min)
        for t in count():
            Input = data[start+t:start+args.backward_steps+t]
            action = agent.actor(torch.FloatTensor(Input.reshape(1, args.backward_steps,-1)).to(device), torch.FloatTensor(state.reshape(1,-1)).to(device)).cpu().data.numpy().flatten()
            action = (action + np.random.normal(0, args.exploration_noise, size=action_size)).clip(-1, 1)
            next_state, reward, done, info = env.move(action)
            replay_buffer.push((start+t, state, next_state, action, reward))
            state = next_state
            total_steps += 1
            train_rewards += reward
            if t+1 == args.update_interval:
                writer.add_scalar('Train/reward', train_rewards/args.update_interval, global_step=total_steps)
                mean_critic_loss = 0
                mean_actor_loss = 0
                for it in range(args.update_iteration):
                    # Sample replay buffer
                    inputs, next_inputs, states, next_states, actions, rewards = replay_buffer.sample()
                    inputs = torch.FloatTensor(inputs).to(device)
                    next_inputs = torch.FloatTensor(next_inputs).to(device)
                    states = torch.FloatTensor(states).to(device)
                    next_states = torch.FloatTensor(next_states).to(device)
                    actions = torch.FloatTensor(actions).to(device)
                    rewards = torch.FloatTensor(rewards).to(device)

                    # Compute the target Q value
                    target_Q = agent.critic_target(next_inputs, next_states, agent.actor_target(next_inputs, next_states))
                    target_Q = rewards + args.gamma*target_Q.detach()
                    # Get current Q estimate
                    current_Q = agent.critic(inputs, states, actions)
                    # Compute critic loss
                    critic_loss = F.mse_loss(current_Q, target_Q)
                    mean_critic_loss += critic_loss

                    # Optimize the critic
                    agent.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    agent.critic_optimizer.step()

                    # Compute actor loss
                    actor_loss = -agent.critic(inputs, states, agent.actor(inputs,states)).mean()
                    mean_actor_loss += actor_loss


                    # Optimize the actor
                    agent.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    agent.actor_optimizer.step()

                    # Update the frozen target models
                    for param, target_param in zip(agent.critic.parameters(), agent.critic_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                    for param, target_param in zip(agent.actor.parameters(), agent.actor_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                writer.add_scalar('Train/critic_loss', mean_critic_loss/args.update_iteration, global_step=total_steps)
                writer.add_scalar('Train/actor_loss', mean_actor_loss/args.update_iteration, global_step=total_steps)
                break
        if (epoch+1) % args.save_interval == 0:
            agent.save()
            with torch.no_grad(): 
                rewards = []
                start,state = env_valid.valid()
                for i in count():
                    Input = data[start+i:start+i+args.backward_steps]
                    action = agent.actor(torch.FloatTensor(Input.reshape(1, args.backward_steps,-1)).to(device), torch.FloatTensor(state.reshape(1,-1)).to(device)).cpu().data.numpy().flatten()
                    next_state, reward, done, info = env_valid.move(action)
                    rewards.append(reward)
                    state = next_state
                    if i==11064:
                        break
            writer.add_scalar('Valid/reward', sum(rewards)/len(rewards), global_step=total_steps)
            if sum(rewards)>reward_best:
                reward_best = sum(rewards)
                agent.save_as_best()
            save_rewards = 0
                