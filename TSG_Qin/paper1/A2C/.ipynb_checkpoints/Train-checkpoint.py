import sys
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
from torch.utils.tensorboard import SummaryWriter
from os import path


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
    np.random.seed(100)
    for i in range(test_runs):
        with torch.no_grad():
            observations = env.reset()
            costs_mean = 0
            for step in range(int(1438)):
                observations = torch.from_numpy(observations).to(torch.float32).to(device)
                # network forward pass
                mu, sigma = actor_net(observations)
                values = critic_net(observations)
                normal_probs = dist.normal.Normal(mu,sigma)
                actions = normal_probs.sample()
                # gather env data, reset done envs and update their obs
                observations, costs, dones, infos = env.step(actions.detach().cpu().numpy())
                costs_mean += np.mean(costs)  
        Values[i]=costs_mean
    
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
                
                mu, sigma = actor_net(observations)
                values = critic_net(observations)
                normal_probs = dist.normal.Normal(mu,sigma+1e-1)
                actions = normal_probs.sample()
                # gather env data, reset done envs and update their obs
                observations, costs, dones, infos = env.step(actions.detach().cpu().numpy())
                # record the finished process          
                total_steps += args.num_workers
                costs = torch.from_numpy(costs).float()
                costs = costs.to(device)
                steps.append(( costs, actions, normal_probs, values))
                if dones.sum()>0:
                    break
                
            rolloutCount+=1 
            failed=[info['fail'] for info in infos]
            Dones = torch.from_numpy(dones).unsqueeze(1)
            Dones = Dones.to(device)
            final_observations = torch.from_numpy(observations).to(torch.float32).to(device)
            final_values = critic_net(final_observations) 
            steps.append((None, None, None, final_values))
            log_normal_probs, normal_entropys, values, returns, advantages = process_rollout(args, steps, device, Dones)
            policy_loss = (log_normal_probs[:,0]*advantages[:,0]).mean()+(log_normal_probs*advantages[:,1].unsqueeze(1)).mean() + args.AC_number*(log_normal_probs[:,-args.AC_number:] * advantages[:,-args.AC_number:]).mean() + args.EV_number*(log_normal_probs[:,1:1+args.EV_number] * advantages[:,2:2+args.EV_number]).mean() 

            value_loss = (.5 * (values - returns) ** 2.).mean() 
            entropy_loss = normal_entropys.mean() 

            loss = policy_loss - entropy_loss * args.entropy_coeff

            loss.backward()
            value_loss.backward()

            nn.utils.clip_grad_norm_(actor_net.parameters(), args.actor_grad_norm_limit)
            nn.utils.clip_grad_norm_(critic_net.parameters(), args.critic_grad_norm_limit)            
            actor_optimizer.step()
            critic_optimizer.step()

            writer.add_scalar('State/Failed', np.sum(failed), total_steps)
            writer.add_scalar('Policy/AC_advantages', advantages[:,-10:].mean(), total_steps)
            writer.add_scalar('Policy/AC_values', values[:,-10:].mean(), total_steps)
            writer.add_scalar('Policy/AC_returns', returns[:,-10:].mean(), total_steps)
            writer.add_scalar('Policy/gen_advantages', advantages[:,0].mean(), total_steps)
            writer.add_scalar('Policy/gen_values', values[:,0].mean(), total_steps)
            writer.add_scalar('Policy/gen_returns', returns[:,0].mean(), total_steps)
            writer.add_scalar('Policy/EV_advantages', advantages[:,2:7].mean(), total_steps)
            writer.add_scalar('Policy/EV_values', values[:,2:7].mean(), total_steps)
            writer.add_scalar('Policy/EV_returns', returns[:,2:7].mean(), total_steps)
            writer.add_scalar('Loss/policy_loss', policy_loss, total_steps)
            writer.add_scalar('Loss/value_loss', value_loss, total_steps)
            writer.add_scalar('Loss/entropy_loss', entropy_loss, total_steps)
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

    advantages = torch.zeros(args.num_workers, args.AC_number+args.EV_number+2)
    advantages = advantages.to(device)
    out = [None] * (len(steps) - 1)
    # run Advantage Estimation, calculate returns, advantages
    for t in reversed(range(len(steps) - 1)):
        costs, actions, normal_probs, values = steps[t]
        _, _, _, next_values = steps[t + 1]
        returns = costs + returns * args.gamma 
        advantages = costs + next_values.detach() * args.gamma - values.detach()
        log_normal_probs = normal_probs.log_prob(actions)

        out[t] = log_normal_probs, normal_probs.entropy(),values, returns, advantages
    # return data as batched Tensors, Variables
    return map(lambda x: torch.cat(x, 0), zip(*out))

   
