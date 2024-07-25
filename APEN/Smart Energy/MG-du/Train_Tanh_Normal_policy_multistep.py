import sys
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
from torch.utils.tensorboard import SummaryWriter
from os import path

def MultivariateNormalDiag(loc, scale_diag):
    if loc.dim() < 1:
        raise ValueError("loc must be at least one-dimensional.")
    return dist.Independent(dist.Normal(loc,scale_diag), 1)

def _clipped_tanh(x):
    finfo = torch.finfo(x.dtype)
    return torch.clamp(torch.tanh(x), min=-1. + finfo.eps, max=1. - finfo.eps)

class TanhTransform(torch.distributions.transforms.Transform):
    domain = torch.distributions.constraints.real
    codomain = torch.distributions.constraints.interval(-1,1)
    bijective = True
    sign = +1
    
    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return _clipped_tanh(x)

    def _inverse(self, y):
        finfo = torch.finfo(y.dtype)
        y = y.clamp(min=-1. + finfo.eps, max=1. - finfo.eps)
        return 0.5*torch.log(2/(1-y)-1)

    def log_abs_det_jacobian(self, x, y):
        return -2*torch.log(torch.cosh(x))
    
def MultivariateTanhNormalDiag(loc,scale_diag):
    MultivariateNormal_dist=MultivariateNormalDiag(loc,scale_diag)
    return torch.distributions.transformed_distribution.TransformedDistribution(MultivariateNormal_dist,TanhTransform())
    

def show_progress(ratio,info=''):
    i=int(max(0,min(1.0,ratio))*80)
    sys.stdout.write('\r')
    sys.stdout.write("[%-81s] %d%% %s" % ('='*i+'>', 100*ratio,info))
    sys.stdout.flush()
    
def save_model(args, net, optimizer, steps):
    torch.save({
            'steps': steps,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, args.model_dir)

def load_model(args, net, optimizer):
    PATH=args.model_dir
    if path.exists(PATH):
        checkpoint = torch.load(PATH, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'],strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        steps = checkpoint['steps']+1
        #steps=0
        net.train()
    else:
        steps=0
    return steps

def test(args, net,optimizer, env, device, writer,total_steps,bestPerformance,test_runs=50):
    Values=np.zeros(test_runs*args.num_workers)
    print('test!!')
    for i in range(test_runs):
        with torch.no_grad():
            observation = env.reset()
            rewards=[]
            for step in range(int(1440)):
                observation = torch.from_numpy(observation).to(torch.float32).to(device)
                # network forward pass
                policies_mu, policies_sigma, value_net= net(observation)

                actions=torch.tanh(policies_mu)
                # gather env data, reset done envs and update their obs
                observation, reward, done, info = env.step(actions.detach().cpu().numpy())
                rewards.append(reward)

        V_x=reward/(1-args.gamma)
        for reward in reversed(rewards):
            V_x=reward+V_x*args.gamma
        Values[i*args.num_workers:(i+1)*args.num_workers]=V_x
    writer.add_histogram('Test/Values', Values, total_steps)
    
    performance=np.mean(Values)
    ret=0
    if bestPerformance>performance:
        save_model(args, net, optimizer, total_steps)
        ret=performance
    else:
        ret=bestPerformance
    return ret
    
def train(args, net, optimizer, env, device):
    total_steps=load_model(args, net, optimizer)
    observations = env.reset()
    writer = SummaryWriter(log_dir=args.log_dir)
    #writer.add_graph(net,Variable(torch.from_numpy(observations).float()).cuda())
    steps = []
    masks=np.array([False]*args.num_workers)
    
    bestPerformance=1e10
    interrupted=False
    
    rolloutCount=0
    try:
        while total_steps < args.total_steps:
            # reset masks for done
            masks[:]=False
            
            for _ in range(args.rollout_steps):
                observations = torch.from_numpy(observations).to(torch.float32).to(device)
                # network forward pass
                policies_mu, policies_sigma, values= net(observations)
                
                policies_probs = MultivariateTanhNormalDiag(policies_mu,policies_sigma+1e-6)
                actions = policies_probs.sample()

                # gather env data, reset done envs and update their obs
                observations, rewards, dones, infos = env.step(actions.detach().cpu().numpy())
                # record the finished process
                masks=masks|dones

                total_steps += args.num_workers

                rewards = torch.from_numpy(rewards).float().unsqueeze(1)
                rewards = rewards.to(device)

                steps.append((rewards, actions, policies_probs, values))
            rolloutCount+=1 
            failed=[info['fail'] for info in infos]

            final_observations = torch.from_numpy(observations).to(torch.float32).to(device)
            _,_, final_values = net(final_observations)
            steps.append((None, None, None, final_values))
            actions, log_action_probs, entropys, values, returns, advantages = process_rollout(args, steps, device)

            policy_loss = (log_action_probs * advantages).mean()
            value_loss = (.5 * (values - returns) ** 2.).mean()
            entropy_loss = entropys.mean()

            loss = policy_loss* args.policy_coeff + value_loss * args.value_coeff - entropy_loss * args.entropy_coeff

            loss.backward()

            nn.utils.clip_grad_norm_(net.parameters(), args.grad_norm_limit)
            optimizer.step()

            writer.add_scalar('State/Failed', np.sum(failed), total_steps)
            writer.add_scalar('Policy/advantages', advantages.mean(), total_steps)
            writer.add_scalar('Policy/values', values.mean(), total_steps)
            writer.add_scalar('Policy/returns', returns.mean(), total_steps)
            writer.add_scalar('Loss/policy_loss', policy_loss, total_steps)
            writer.add_scalar('Loss/value_loss', value_loss, total_steps)
            writer.add_scalar('Loss/entropy_loss', entropy_loss, total_steps)
            writer.add_scalar('Loss/loss', loss, total_steps)

            steps = []
            
            # test the model periodically
            if rolloutCount%args.save_interval==0:
                bestPerformance=test(args, net, optimizer, env, device, writer,total_steps,bestPerformance)
                # reset the env after test
                masks[:]=True

            # reset the finished process, and update the observation
            observations=env.partial_reset(masks)
            
            # empty the gradient
            optimizer.zero_grad()
            
            info='total_steps: %d'%total_steps
            show_progress(total_steps/args.total_steps,info)
            
    except KeyboardInterrupt:
        print('\nReceived KeyboardInterrupt')
        interrupted=True
    finally:
        print('\nTraining stopped, saving model...')
        writer.close()
        save_model(args, net,optimizer, total_steps)
        env.close(interrupted)
    
    
    
def process_rollout(args, steps, device):
    # bootstrap discounted returns with final value estimates
    _, _, _, last_values = steps[-1]
    returns = last_values.detach()

    advantages = torch.zeros(args.num_workers, 1)
    advantages = advantages.to(device)

    out = [None] * (len(steps) - 1)
    
    # run Advantage Estimation, calculate returns, advantages
    for t in reversed(range(len(steps) - 1)):
        rewards, actions, policies_probs, values = steps[t]
        _, _, _, next_values = steps[t + 1]

        returns = rewards + returns * args.gamma
        advantages = rewards + next_values.detach() * args.gamma - values.detach()
        log_action_probs=policies_probs.log_prob(actions)
        entropys=policies_probs.base_dist.entropy()
        out[t] = actions, log_action_probs, entropys, values, returns, advantages

    # return data as batched Tensors, Variables
    return map(lambda x: torch.cat(x, 0), zip(*out))

   
