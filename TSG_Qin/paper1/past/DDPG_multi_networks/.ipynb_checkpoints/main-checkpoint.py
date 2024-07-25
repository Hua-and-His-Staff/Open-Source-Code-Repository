#!/usr/bin/env python3 
import sys
import numpy as np
import argparse
from copy import deepcopy
import torch
#import gym
from torch.utils.tensorboard import SummaryWriter
#from normalized_env import NormalizedEnv
from env import EIEnv
from evaluator import Evaluator
from ddpg import DDPG
from util import *

#gym.undo_logger_setup()
def show_progress(ratio,info=''):
    i=int(max(0,min(1.0,ratio))*80)
    sys.stdout.write('\r')
    sys.stdout.write("[%-81s] %d%% %s" % ('='*i+'>', 100*ratio,info))
    sys.stdout.flush()
    
def train(num_iterations, agent, train_env, valid_env, evaluate, steps_update_interval, episodes_validate_interval, output, debug=False):
    agent.is_training = True
    total_steps = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    writer = SummaryWriter(log_dir='./runs')
    while total_steps < num_iterations:
        # [optional] evaluate         
        # reset if it is the start of episode
        if observation is None:
            observation = deepcopy(train_env.reset())
            agent.reset(observation)
        # agent pick action ...
        if total_steps <= args.warmup:
            action = agent.random_action()
        else:
            action = agent.select_action(observation)
        # env response with next_observation, reward, terminate_info
        observation2, reward, done, info = train_env.step(action)
        observation2 = deepcopy(observation2)


        # agent observe and update policy
        agent.observe(reward, observation2, done)
        
        if total_steps > args.warmup and total_steps % steps_update_interval == 0:
            agent.update_policy()
        info='total_steps: %d'%total_steps
        show_progress(total_steps/num_iterations,info)


        # [optional] save intermideate model

        # update 
        total_steps += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)

        if done: # end of episode
            agent.memory.append(
                observation,
                agent.select_action(observation),
                0., False
            )

            # reset
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1


            if evaluate is not None and episodes_validate_interval > 0 and episode % episodes_validate_interval == 0:
                policy = lambda x: agent.select_action(x, decay_epsilon=False)
                validate_costs = evaluate(valid_env, policy, debug=False)
                agent.save_model(output,validate_costs.mean())
                writer.add_scalar('Evaluate/gen_costs', validate_costs[:,0].mean(), total_steps)
                writer.add_scalar('Evaluate/bes_costs', validate_costs[:,1].mean(), total_steps)
                writer.add_scalar('Evaluate/EV_costs', validate_costs[:,2:7].mean(), total_steps)
                writer.add_scalar('Evaluate/AC_costs', validate_costs[:,-10:].mean(), total_steps)
                writer.add_scalar('Evaluate/costs', validate_costs.mean(), total_steps)
                

            
        

def test(num_episodes, agent, env, evaluate, model_path, visualize=True, debug=False):

    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()
    policy = lambda x: agent.select_action(x, decay_epsilon=False)

    for i in range(num_episodes):
        validate_reward = evaluate(env, policy, debug=debug, visualize=visualize, save=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--env', default='microgrid', type=str, help='')
    parser.add_argument('--pre_training', default= True, type=bool, help='')
    parser.add_argument('--rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--prate', default=2e-4, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--warmup', default=60*24*10, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.95, type=float, help='')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=600000, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.01, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.3, type=float, help='noise sigma') 
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu') 
    parser.add_argument('--validate_episodes', default=10, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--steps_update_interval', default=100, type=int, help='how many steps to update network')
    parser.add_argument('--episodes_validate_interval', default=100, type=int, help='how many train episodes to perform a validate experiment')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='') 
    parser.add_argument('--train_iter', default=3e8, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=1e8, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--cuda', default=True, type=bool, help='')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    # parser.add_argument('--l2norm', default=0.01, type=float, help='l2 weight decay') # TODO
    # parser.add_argument('--cuda', dest='cuda', action='store_true') # TODO

    args = parser.parse_args()
    #创建输出文件
    output = './output'
    if not os.path.exists(output):
        os.makedirs(output)
    if args.resume=='default':
        args.resume = output
    train_env = EIEnv()
    valid_env = EIEnv()
    #if args.seed > 0:
    #    np.random.seed(args.seed)
    #    env.seed(args.seed)

    nb_states = train_env.observation_space.shape[0]
    nb_actions = train_env.action_space.shape[0]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    agent = DDPG(nb_states, nb_actions, device, args)
    #agent.load_current_weights(args.resume)
    evaluate = Evaluator(args.validate_episodes, nb_actions+1)

    if args.mode == 'train':
        train(args.train_iter, agent, train_env, valid_env, evaluate, args.steps_update_interval,
            args.episodes_validate_interval, output, debug=args.debug )

    elif args.mode == 'test':
        test(1, agent, valid_env, evaluate, args.resume,
            visualize=True, debug=args.debug)

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
