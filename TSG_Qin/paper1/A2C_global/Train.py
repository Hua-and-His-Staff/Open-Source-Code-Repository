import sys
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
from torch.utils.tensorboard import SummaryWriter
from os import path
from itertools import count

M = 10
N = 5


class A2C():
    def __init__(self, args, net, optimizer, device):
        self.args = args
        self.net = net
        self.optimizer = optimizer
        self.device = device

    def update(self,trajectory,dones):
        # bootstrap discounted returns with final value estimates
        _, _, _, last_values = trajectory[-1]
        returns = last_values.detach() * (~dones)
        out = [None] * (len(trajectory) - 1)
        # run Advantage Estimation, calculate returns, advantages
        for t in reversed(range(len(trajectory) - 1)):
            costs, actions, probs, values = trajectory[t]
            _, _, _, next_values = trajectory[t + 1]
            returns = costs + returns * self.args.gamma 
            advantages = costs + next_values.detach() * self.args.gamma - values.detach()
            log_probs = probs.log_prob(actions)
            out[t] = log_probs, probs.entropy(),values, returns, advantages    
            
        log_probs, normal_entropys, values, returns, advantages = map(lambda x: torch.cat(x, 0), zip(*out))
        policy_loss = (log_probs[:,0]*advantages[:,0]).mean()+(log_probs*advantages[:,1].unsqueeze(1)).mean() + M*(log_probs[:,-M:] * advantages[:,-M:]).mean() + N*(log_probs[:,1:1+N] * advantages[:,2:2+N]).mean() 
        value_loss = 0.1*(.5 * (values - returns) ** 2.).mean() 
        entropy_loss = normal_entropys.mean() 
        loss = policy_loss + value_loss - entropy_loss * self.args.entropy_coeff
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_norm_limit)          
        self.optimizer.step()
        # empty the gradient
        self.optimizer.zero_grad()
        return policy_loss, value_loss, entropy_loss

 
    def save_model(self, model_path, total_episodes, bestPerformance):
        torch.save({
                'total_episodes': total_episodes,
                'best_performance': bestPerformance,
                'model_state_dict': self.net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                }, model_path)
        
    def load_model(self,model_path):
        if path.exists(model_path):
            checkpoint = torch.load(model_path)
            self.net.load_state_dict(checkpoint['model_state_dict'],strict=False)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])        
            total_episodes = checkpoint['total_episodes']
            best_performance = checkpoint['best_performance']
            #steps=0
            self.net.train()
        else:
            total_episodes=0
            best_performance = 1e10
        return total_episodes, best_performance

    
def train(args, policy, env, device):
    
    total_episodes, bestPerformance = policy.load_model(args.model_dir+'/current')
    writer = SummaryWriter(log_dir=args.log_dir)
    interrupted=False
    try:
        while total_episodes < args.total_episodes:
            trajectory = []
            obs_GRU,obs_MLP = env.reset()
            hidden = None
            policy_losses = 0 
            value_losses = 0
            entropy_losses = 0
            for episode_step in count():
                
                obs_GRU = torch.from_numpy(obs_GRU).to(torch.float).to(device)
                obs_MLP = torch.from_numpy(obs_MLP).to(torch.float).to(device)
                
                mu, sigma, values, hidden = policy.net(obs_GRU,obs_MLP,hidden)
                probs = dist.normal.Normal(mu,sigma+1e-1)
                #sigmoid_probs = dist.transformed_distribution.TransformedDistribution(normal_probs,dist.transforms.SigmoidTransform())
                actions = probs.sample()
                # gather env data, reset done envs and update their obs
                obs_GRU, obs_MLP, costs, dones, infos = env.move(actions.detach().cpu().numpy())
                # record the finished process          
                costs = torch.from_numpy(costs).float()
                costs = costs.to(device)
                trajectory.append(( costs, actions, probs, values))
                if (episode_step+1) % args.update_steps == 0:
                    final_obs_GRU = torch.from_numpy(obs_GRU).to(torch.float).to(device)
                    final_obs_MLP = torch.from_numpy(obs_MLP).to(torch.float).to(device)
                    _, _, final_values, _ = policy.net(final_obs_GRU, final_obs_MLP, hidden) 
                    trajectory.append((None, None, None, final_values))
                    policy_loss, value_loss, entropy_loss = policy.update(trajectory,torch.from_numpy(dones).unsqueeze(1).to(device))
                    policy_losses += policy_loss
                    value_losses += value_loss
                    entropy_losses += entropy_loss
                    trajectory = []
                    hidden = hidden.detach()
                if dones.any()==True:
                    break
                    
            total_episodes += args.num_workers
            writer.add_scalar('Loss/policy_loss', policy_losses, total_episodes)
            writer.add_scalar('Loss/value_loss', value_losses, total_episodes)
            writer.add_scalar('Loss/entropy_loss', entropy_losses, total_episodes)
            
            info='total_episodes: %d'%total_episodes
            show_progress(total_episodes/args.total_episodes,info)            

            # test the model periodically
            if total_episodes%args.save_interval==0:
                Values=np.zeros([args.test_runs,args.num_workers])
                #using the same seed
                np.random.seed(100)
                with torch.no_grad():
                    for i in range(args.test_runs):
                        obs_GRU,obs_MLP = env.reset()
                        costs_episode = np.zeros(args.num_workers)
                        hidden = None
                        for episode_step in count():
                            obs_GRU = torch.from_numpy(obs_GRU).to(torch.float).to(device)
                            obs_MLP = torch.from_numpy(obs_MLP).to(torch.float).to(device)
                            mu, sigma, value, hidden = policy.net(obs_GRU,obs_MLP,hidden)
                            probs = dist.normal.Normal(mu,sigma)
                            #sigmoid_probs = dist.transformed_distribution.TransformedDistribution(normal_probs,dist.transforms.SigmoidTransform())
                            actions = probs.sample()
                            # gather env data, reset done envs and update their obs
                            obs_GRU, obs_MLP, costs, dones, infos = env.move(actions.detach().cpu().numpy())
                            costs_episode += np.sum(costs,axis=1)  
                            if dones.any()==True:
                                break
                        Values[i]=costs_episode
                performance=np.mean(Values)
                writer.add_scalar('Test/costs', performance, total_episodes)
                
                if bestPerformance>performance:
                    bestPerformance=performance
                    policy.save_model(args.model_dir+'/best', total_episodes, bestPerformance)
                else:
                    policy.save_model(args.model_dir+'/current', total_episodes, bestPerformance)
            
    except KeyboardInterrupt:
        print('\nReceived KeyboardInterrupt')
        interrupted=True
    finally:
        print('\nTraining stopped, saving model...')
        writer.close()
        policy.save_model(args.model_dir+'/current', total_episodes, bestPerformance)
        env.close(interrupted)    
    
def show_progress(ratio,info=''):
    i=int(max(0,min(1.0,ratio))*80)
    sys.stdout.write('\r')
    sys.stdout.write("[%-81s] %d%% %s" % ('='*i+'>', 100*ratio,info))
    sys.stdout.flush()


    
    
    


   
