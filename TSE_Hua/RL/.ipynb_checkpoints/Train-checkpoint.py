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

def _clipped_sigmoid(x):
    finfo = torch.finfo(x.dtype)
    return torch.clamp(torch.sigmoid(x), min=0. + finfo.eps, max=1. - finfo.eps)

#%class SigmoidTransform(torch.distributions.transforms.Transform):
#    domain = torch.distributions.constraints.real
#    codomain = torch.distributions.constraints.interval(0,1)
#    bijective = True
#    sign = +1
#    
#    def __eq__(self, other):
#        return isinstance(other, SigmoidTransform)
#
#    def _call(self, x):
#        return _clipped_sigmoid(x)
#
#    def _inverse(self, y):
#        finfo = torch.finfo(y.dtype)
#        y = y.clamp(min=-1. + finfo.eps, max=1. - finfo.eps)
#        return 0.5*torch.log(2/(1-y)-1)
#
#    def log_abs_det_jacobian(self, x, y):
#        return -2*torch.log(torch.cosh(x))
    
def MultivariateSigmoidNormalDiag(loc,scale_diag):
    MultivariateNormal_dist=MultivariateNormalDiag(loc,scale_diag)
    return torch.distributions.transformed_distribution.TransformedDistribution(MultivariateNormal_dist, dist.transforms.SigmoidTransform())
    

def show_progress(ratio,info=''):
    i=int(max(0,min(1.0,ratio))*80)
    sys.stdout.write('\r')
    sys.stdout.write("[%-81s] %d%% %s" % ('='*i+'>', 100*ratio,info))
    sys.stdout.flush()
    
def save_model(actor_net, critic_net, actor_optimizer, critic_optimizer, steps, name):
    torch.save({
            'steps': steps,
            'actor_model_state_dict': actor_net.state_dict(),
            'actor_optimizer_state_dict': actor_optimizer.state_dict(),
            'critic_model_state_dict': critic_net.state_dict(),
            'critic_optimizer_state_dict': critic_optimizer.state_dict(),        
            }, name)

def load_model(args, actor_net, critic_net, actor_optimizer, critic_optimizer):
    PATH=args.model_dir + '/best'
    if path.exists(PATH):
        checkpoint = torch.load(PATH)
        actor_net.load_state_dict(checkpoint['actor_model_state_dict'],strict=False)
        actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        actor_net.load_state_dict(checkpoint['actor_model_state_dict'],strict=False)
        actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])        
        steps = checkpoint['steps']+1
        #steps=0
        actor_net.train()
        critic_net.train()
    else:
        steps=0
    return steps

def test(args, actor_net, critic_net, actor_optimizer, critic_optimizer, env, device, writer,total_steps,bestPerformance,test_runs=10):
    Values=np.zeros(test_runs*args.num_workers)
    #using one seed
    np.random.seed(args.seed)
    for i in range(test_runs):
        with torch.no_grad():
            observation = env.reset()
            #rewards=[]
            reward_sum = np.zeros(args.num_workers)
            for step in range(int(1440)):
                observation = torch.from_numpy(observation).to(torch.float32).to(device)
                # network forward pass
                policies_mu, policies_sigma = actor_net(observation)

                actions=policies_mu
                # gather env data, reset done envs and update their obs
                observation, reward, done, info = env.step(actions.detach().cpu().numpy())
                #dones.append(done)
                #rewards.append(reward)
                reward_sum += reward
                #if done.sum()==args.num_workers:
                #    break

    #    V_x=reward
    #    for reward in reversed(rewards[:-1]):
    #        V_x=reward+V_x*args.gamma        
        Values[i*args.num_workers:(i+1)*args.num_workers]=reward_sum
    #writer.add_histogram('Test/Values', Values, total_steps)
    writer.add_scalar('Test/reward', Values.mean(), total_steps)
    save_model(actor_net, critic_net, actor_optimizer, critic_optimizer, total_steps, args.model_dir+'%d/now'%args.training_day)
    performance=np.mean(Values)
    ret=0
    if bestPerformance>performance:
        save_model(actor_net, critic_net, actor_optimizer, critic_optimizer, total_steps, args.model_dir+'%d/best'%args.training_day)
        ret=performance
    else:
        ret=bestPerformance
    return ret
    
def train(args, actor_net, critic_net, actor_optimizer, critic_optimizer, env, device):
    total_steps=load_model(args, actor_net, critic_net, actor_optimizer, critic_optimizer)
    total_steps = 0
    observations = env.reset()
    writer = SummaryWriter(log_dir=args.log_dir+'%d'%args.training_day)
    #writer.add_graph(net,Variable(torch.from_numpy(observations).float()).cuda())
    steps = []
    bestPerformance=1e10
    interrupted=False
    
    rolloutCount=0
    try:
        while total_steps < args.total_steps:

            
            for _ in range(args.rollout_steps):
                observations = torch.from_numpy(observations).to(torch.float32).to(device)
                # network forward pass
                policies_mu, policies_sigma = actor_net(observations)
                values = critic_net(observations)
                #policies_probs = MultivariateSigmoidNormalDiag(policies_mu,5*policies_sigma+1e-2)
                policies_probs = dist.normal.Normal(policies_mu,0.2*policies_sigma+1e-5)
                actions = policies_probs.sample()
                # gather env data, reset done envs and update their obs
                observations, rewards, dones, infos = env.step(actions.detach().cpu().numpy())
                # record the finished process

                #if done=True, repeat
                total_steps += args.num_workers
                


                rewards = torch.from_numpy(rewards).float().unsqueeze(1)
                rewards = rewards.to(device)

                steps.append(( rewards, actions, policies_probs, values))
                if dones.sum()>0:
                    break
            rolloutCount+=1 
            failed=[info['fail'] for info in infos]
            Dones = torch.from_numpy(dones).unsqueeze(1)
            Dones = Dones.to(device)
            final_observations = torch.from_numpy(observations).to(torch.float32).to(device)
            final_values = critic_net(final_observations) 
            steps.append((None, None, None, final_values))
            actions, log_action_probs, entropys, values, returns, advantages = process_rollout(args, steps, device, Dones)

            policy_loss = (log_action_probs * advantages).mean()
            value_loss = (.5 * (values - returns) ** 2.).mean()
            entropy_loss = entropys.mean()
            

            loss = policy_loss - entropy_loss * args.entropy_coeff
            if total_steps > args.training_start_steps:
                loss.backward()
            value_loss.backward()

            nn.utils.clip_grad_norm_(actor_net.parameters(), args.actor_grad_norm_limit)
            nn.utils.clip_grad_norm_(critic_net.parameters(), args.critic_grad_norm_limit)            
            actor_optimizer.step()
            critic_optimizer.step()

            #writer.add_scalar('State/Failed', np.sum(failed), total_steps)
            writer.add_scalar('Policy/advantages', advantages.mean(), total_steps)
            writer.add_scalar('Policy/values', values.mean(), total_steps)
            writer.add_scalar('Policy/returns', returns.mean(), total_steps)
            writer.add_scalar('Loss/policy_loss', policy_loss, total_steps)
            writer.add_scalar('Loss/value_loss', value_loss, total_steps)
            writer.add_scalar('Loss/entropy_loss', entropy_loss, total_steps)
            #writer.add_scalar('Loss/loss', loss, total_steps)

            steps = []
            
            # test the model periodically
            if rolloutCount%args.save_interval==0:
                bestPerformance=test(args, actor_net, critic_net, actor_optimizer, critic_optimizer, env, device, writer,total_steps, bestPerformance)
                # reset the env after test


            # reset the finished process, and update the observation
            observations=env.partial_reset(dones)
            
            # empty the gradient
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            info='total_steps: %d'%total_steps
            show_progress(total_steps/args.total_steps,info)
            
    except KeyboardInterrupt:
        print('\nReceived KeyboardInterrupt')
        interrupted=True
    finally:
        print('\nTraining stopped, saving model...')
        writer.close()
        save_model(actor_net, critic_net, actor_optimizer, critic_optimizer, total_steps, args.model_dir+'/now')
        env.close(interrupted)
    
    
    
def process_rollout(args, steps, device, dones):
    # bootstrap discounted returns with final value estimates
    _, _, _, last_values = steps[-1]
    returns = last_values.detach() * (~dones)

    advantages = torch.zeros(args.num_workers, 1)
    advantages = advantages.to(device)

    out = [None] * (len(steps) - 1)
    
    # run Advantage Estimation, calculate returns, advantages
    for t in reversed(range(len(steps) - 1)):
        rewards, actions, policies_probs, values = steps[t]
        _, _, _, next_values = steps[t + 1]
        #if done
        #returns = rewards
        returns = rewards + returns * args.gamma 
        advantages = rewards + next_values.detach() * args.gamma - values.detach()
        log_action_probs=policies_probs.log_prob(actions)
        #print(log_action_probs)
        #entropys=policies_probs.base_dist.entropy()
        entropys=policies_probs.entropy()
        out[t] = actions, log_action_probs, entropys, values, returns, advantages

    # return data as batched Tensors, Variables
    return map(lambda x: torch.cat(x, 0), zip(*out))

   
