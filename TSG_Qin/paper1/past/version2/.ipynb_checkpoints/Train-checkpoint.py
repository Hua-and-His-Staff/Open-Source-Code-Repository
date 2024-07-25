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
    Values=np.zeros(test_runs*args.num_workers)
    #using one seed
    #env.seed(100)
    for i in range(test_runs):
        with torch.no_grad():
            observation = env.reset()
            #rewards=[]
            reward_sum = np.zeros(args.num_workers)
            for step in range(int(1438)):
                observation = torch.from_numpy(observation).to(torch.float32).to(device)
                # network forward pass
                EV_mu, EV_sigma, AC_prob = actor_net(observation)
                actions = torch.cat((EV_mu,(AC_prob>0.5).to(torch.float)),dim=1)
                # gather env data, reset done envs and update their obs
                observation, gen_cost, BES_cost, AC_costs, dones, infos = env.step(actions.detach().cpu().numpy())
                reward_sum += gen_cost + BES_cost[:,0] + np.sum(AC_costs, axis=1)


    #    V_x=reward
    #    for reward in reversed(rewards[:-1]):
    #        V_x=reward+V_x*args.gamma        
        Values[i*args.num_workers:(i+1)*args.num_workers]=reward_sum
    #writer.add_histogram('Test/Values', Values, total_steps)
    
    save_model(actor_net, critic_net, actor_optimizer, critic_optimizer, total_steps, args.model_dir+'/now')
    performance=np.mean(Values)
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
                gen_value, BES_value, AC_values = critic_net(observations)
                #policies_probs = MultivariateSigmoidNormalDiag(policies_mu,5*policies_sigma+1e-2)
                EV_probs = dist.normal.Normal(EV_mu,EV_sigma+1e-4)
                #greedy policy
                p = torch.rand(30,device='cuda:0')
                AC_probs = dist.bernoulli.Bernoulli(AC_prob*(p>0.3)+0.5*torch.ones(30,device='cuda:0')*(p<=0.3))
                EV_actions = EV_probs.sample()
                AC_actions = AC_probs.sample()
                actions = torch.cat((EV_actions,AC_actions),dim=1)
                # gather env data, reset done envs and update their obs
                observations, gen_cost, bes_cost, comfort_cost, dones, infos = env.step(actions.detach().cpu().numpy())
                # record the finished process          
                total_steps += args.num_workers
                gen_cost = torch.from_numpy(gen_cost).float()
                bes_cost = torch.from_numpy(bes_cost).float()
                comfort_cost = torch.from_numpy(comfort_cost).float()
                gen_cost = gen_cost.to(device)
                bes_cost = bes_cost.to(device)
                comfort_cost = comfort_cost.to(device)
                steps.append(( gen_cost, bes_cost, comfort_cost, EV_actions, EV_probs, AC_actions, AC_probs, gen_value, BES_value, AC_values))
                if dones.sum()>0:
                    break
                
            rolloutCount+=1 
            failed=[info['fail'] for info in infos]
            Dones = torch.from_numpy(dones).unsqueeze(1)
            Dones = Dones.to(device)
            final_observations = torch.from_numpy(observations).to(torch.float32).to(device)
            final_gen_value, final_BES_value, final_AC_values = critic_net(final_observations) 
            steps.append((None, None,None, None, None, None, None, final_gen_value, final_BES_value, final_AC_values))
            log_genEV_probs, log_AC_probs, EV_entropys, AC_entropys, gen_value, BES_value, AC_values, gen_returns,BES_returns, AC_returns, gen_advantages, BES_advantages, AC_advantages = process_rollout(args, steps, device, Dones)
            policy_loss = (log_AC_probs * AC_advantages).mean() + (log_genEV_probs[:,0].unsqueeze(1) * gen_advantages).mean() + (log_AC_probs * BES_advantages).mean() + (log_genEV_probs * BES_advantages).mean()
            value_loss = (.5 * (gen_value - gen_returns) ** 2.).mean() + (.5 * (BES_value - BES_returns) ** 2.).mean() +(.5 * (AC_values - AC_returns) ** 2.).mean()
            entropy_loss = EV_entropys.mean() + AC_entropys.mean()

            loss = policy_loss - entropy_loss * args.entropy_coeff

            loss.backward()
            value_loss.backward()

            nn.utils.clip_grad_norm_(actor_net.parameters(), args.actor_grad_norm_limit)
            nn.utils.clip_grad_norm_(critic_net.parameters(), args.critic_grad_norm_limit)            
            actor_optimizer.step()
            critic_optimizer.step()

            writer.add_scalar('State/Failed', np.sum(failed), total_steps)
            writer.add_scalar('Policy/AC_advantages', AC_advantages.mean(), total_steps)
            writer.add_scalar('Policy/AC_values', AC_values.mean(), total_steps)
            writer.add_scalar('Policy/AC_returns', AC_returns.mean(), total_steps)
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
    _, _, _, _, _, _, _, last_gen_value, last_BES_value, last_AC_values = steps[-1]
    gen_returns = last_gen_value.detach() * (~dones)
    BES_returns = last_BES_value.detach() * (~dones)
    AC_returns = last_AC_values.detach() * (~dones)

    gen_advantages = torch.zeros(args.num_workers, 1)
    BES_advantages = torch.zeros(args.num_workers, 1)
    AC_advantages = torch.zeros(args.num_workers, 30)
    gen_advantages = gen_advantages.to(device)
    BES_advantages = BES_advantages.to(device)
    AC_advantages = AC_advantages.to(device)
    out = [None] * (len(steps) - 1)
    # run Advantage Estimation, calculate returns, advantages
    for t in reversed(range(len(steps) - 1)):
        gen_cost, BES_cost, comfort_cost, EV_actions, EV_probs, AC_actions, AC_probs,gen_value, BES_value, AC_values = steps[t]
        _, _, _, _, _, _, _, next_gen_value,next_BES_value, next_AC_values = steps[t + 1]
        #if done
        #returns = rewards
        gen_returns = gen_cost + gen_returns * args.gamma 
        BES_returns = BES_cost + BES_returns * args.gamma
        AC_returns = comfort_cost + AC_returns * args.gamma
        gen_advantages = gen_cost.unsqueeze(1) + next_gen_value.detach() * args.gamma - gen_value.detach()
        BES_advantages = BES_cost + next_BES_value.detach() * args.gamma - BES_value.detach()
        AC_advantages = comfort_cost + next_AC_values.detach() * args.gamma - AC_values.detach()
        log_genEV_probs = EV_probs.log_prob(EV_actions)
        log_AC_probs = AC_probs.log_prob(AC_actions)

        out[t] = log_genEV_probs, log_AC_probs, EV_probs.entropy(), AC_probs.entropy(), gen_value, BES_value, AC_values, gen_returns,BES_returns, AC_returns, gen_advantages, BES_advantages, AC_advantages
    # return data as batched Tensors, Variables
    return map(lambda x: torch.cat(x, 0), zip(*out))

   
