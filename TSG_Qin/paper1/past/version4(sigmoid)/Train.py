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
    PATH=args.model_dir + '/now'
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
    Values=np.zeros(test_runs)
    #using one seed
    np.random.seed(100)
    for i in range(test_runs):
        with torch.no_grad():
            observation = env.reset()
            #rewards=[]
            reward_sum = 0
            for step in range(int(1438)):
                observation = torch.from_numpy(observation).to(torch.float32).to(device)
                # network forward pass
                EV_mu, EV_sigma, AC_prob = actor_net(observation)
                actions = torch.cat(( torch.sigmoid(EV_mu),(AC_prob>0.5).to(torch.float)),dim=1)
                # gather env data, reset done envs and update their obs
                observation, cost, dones, infos = env.step(actions.detach().cpu().numpy())
                reward_sum += np.mean(cost) 


    #    V_x=reward
    #    for reward in reversed(rewards[:-1]):
    #        V_x=reward+V_x*args.gamma        
        Values[i]=reward_sum
    #writer.add_histogram('Test/Values', Values, total_steps)
    
    save_model(actor_net, critic_net, actor_optimizer, critic_optimizer, total_steps, args.model_dir+'/now')
    performance=np.mean(Values)
    writer.add_scalar('Test/mean_loss', performance, total_steps)
    ret=0
    if bestPerformance>performance:
        save_model(actor_net, critic_net, actor_optimizer, critic_optimizer, total_steps, args.model_dir+'/best')
        ret=performance
    else:
        ret=bestPerformance
    return ret
    
def train(args, actor_net, critic_net, actor_optimizer, critic_optimizer, env, device):
    total_steps=load_model(args, actor_net, critic_net, actor_optimizer, critic_optimizer)
    observations = env.reset()
    writer = SummaryWriter(log_dir=args.log_dir)
    #writer.add_graph(net,Variable(torch.from_numpy(observations).float()).cuda())
    steps = []
    bestPerformance=1e10
    interrupted=False
    
    rolloutCount=0
    try:
        while total_steps < args.total_steps:

            
            for _ in range(args.rollout_steps):
                observations = torch.from_numpy(observations).to(torch.float32).to(device)
                #print(observations.shape)
                # network forward pass
                
                EV_mu, EV_sigma, AC_prob = actor_net(observations)
                value = critic_net(observations)
                EV_probs = MultivariateSigmoidNormalDiag(EV_mu,EV_sigma+1e-2)               
                #EV_probs = dist.normal.Normal(EV_mu,EV_sigma+1e-4)
                #压缩到0.2-0.8之间
                AC_prob = 0.2 + 0.6*AC_prob
                AC_probs = dist.bernoulli.Bernoulli(AC_prob)
                
                EV_actions = EV_probs.sample()
                AC_actions = AC_probs.sample()
                actions = torch.cat((EV_actions,AC_actions),dim=1)
                # gather env data, reset done envs and update their obs
                observations, cost, dones, infos = env.step(actions.detach().cpu().numpy())
                # record the finished process          
                total_steps += args.num_workers
                cost = torch.from_numpy(cost).float()

                cost = cost.to(device)
                steps.append(( cost, EV_actions, EV_probs, AC_actions, AC_probs, value))
                if dones.sum()>0:
                    break
                
            rolloutCount+=1 
            failed=[info['fail'] for info in infos]
            Dones = torch.from_numpy(dones).unsqueeze(1)
            Dones = Dones.to(device)
            final_observations = torch.from_numpy(observations).to(torch.float32).to(device)
            final_value = critic_net(final_observations) 
            steps.append((None, None, None, None, None, final_value))
            log_genEV_probs, log_AC_probs, EV_entropys, AC_entropys, values, returns, advantages = process_rollout(args, steps, device, Dones)
            #print(log_AC_probs.shape)
            #print(log_genEV_probs.shape)
            #print(advantages.shape)
            policy_loss = (30*(log_AC_probs * advantages).mean() + 21*(log_genEV_probs * advantages).mean())/51
            value_loss = (.5 * (values - returns) ** 2.).mean() 
            entropy_loss = EV_entropys.mean() + AC_entropys.mean()

            loss = policy_loss - entropy_loss * args.entropy_coeff

            loss.backward()
            value_loss.backward()

            nn.utils.clip_grad_norm_(actor_net.parameters(), args.actor_grad_norm_limit)
            nn.utils.clip_grad_norm_(critic_net.parameters(), args.critic_grad_norm_limit)            
            actor_optimizer.step()
            critic_optimizer.step()

            writer.add_scalar('State/Failed', np.sum(failed), total_steps)
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
    _, _, _, _, _, last_value = steps[-1]
    returns = last_value.detach() * (~dones)

    advantages = torch.zeros(args.num_workers, 1)
    advantages = advantages.to(device)
    out = [None] * (len(steps) - 1)
    # run Advantage Estimation, calculate returns, advantages
    for t in reversed(range(len(steps) - 1)):
        cost, EV_actions, EV_probs, AC_actions, AC_probs,value = steps[t]
        _, _, _, _, _, next_value = steps[t + 1]
        returns = cost + returns * args.gamma 
        advantages = cost + next_value.detach() * args.gamma - value.detach()
        log_genEV_probs = EV_probs.log_prob(EV_actions)
        log_AC_probs = AC_probs.log_prob(AC_actions)

        out[t] = log_genEV_probs, log_AC_probs, EV_probs.base_dist.entropy(), AC_probs.entropy(), value, returns, advantages
    # return data as batched Tensors, Variables
    return map(lambda x: torch.cat(x, 0), zip(*out))

   
