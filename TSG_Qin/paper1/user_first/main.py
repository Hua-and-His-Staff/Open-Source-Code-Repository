import copy
import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c.algo import A2C
from a2c.arguments import get_args
from a2c.envs import make_vec_envs
from a2c.model import Policy
from a2c.storage import RolloutStorage
from evaluation import evaluate
from torch.utils.tensorboard import SummaryWriter

def main():
    args = get_args()
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    log_dir = os.path.join(args.log_dir, 'user_first')
    save_dir = os.path.join(args.save_dir, 'user_first')        
    try:
        os.makedirs(log_dir)
    except OSError:
        pass    
    try:
        os.makedirs(save_dir)
    except OSError:
        pass
    #torch.set_num_threads(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    envs = make_vec_envs(args.seed, args.num_processes, args.env_settings)
    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space.shape,
        base_kwargs={'recurrent': args.recurrent_policy,'assign_credit': args.env_settings['assign_credit']})
    actor_critic.to(device)

    agent = A2C(
        actor_critic,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        max_grad_norm=args.max_grad_norm)


    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size,args.env_settings['assign_credit'])

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)



    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    writer = SummaryWriter(log_dir=log_dir)
    
    if os.path.exists(os.path.join(save_dir, 'current')):
        checkpoint = torch.load(os.path.join(save_dir, 'current'))
        actor_critic.base.load_state_dict(checkpoint['model_state_dict'],strict=False)
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])        
        j = checkpoint['num_updates']
        best_eval_reward = checkpoint['best_eval_reward']
        actor_critic.base.train()
    else:
        j=0
        best_eval_reward = -1e10
    policy_losses = 0
    value_losses = 0
    entropy_losses = 0
    interrupted=False
    try:
        while j < num_updates:
            if args.use_linear_lr_decay:
                # decrease learning rate linearly
                utils.update_linear_schedule(agent.optimizer, j, num_updates, args.lr)

            for step in range(args.num_steps):
                # Sample actions
                with torch.no_grad():
                    value, action, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])


                # Obser reward and next obs
                obs, reward, done, infos = envs.move(action.detach().cpu().numpy())
                #print(reward.shape)

                # If done then clean the history of observations.
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                rollouts.insert(obs, recurrent_hidden_states, action, reward, masks)
                
                obs = envs.partial_reset(done)

            with torch.no_grad():
                next_value = actor_critic.get_value(
                    rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1]).detach()


            rollouts.compute_returns(next_value, args.gamma)

            value_loss, policy_loss, dist_entropy = agent.update(rollouts)
            policy_losses+=policy_loss
            value_losses+=value_loss
            entropy_losses+=dist_entropy

            rollouts.after_update()
            
            j+=1
            
            if j % args.log_interval == 0:
                total_num_steps = (j + 1) * args.num_processes * args.num_steps
                writer.add_scalar('Loss/policy_loss', policy_losses/args.log_interval, total_num_steps)
                writer.add_scalar('Loss/value_loss', value_losses/args.log_interval, total_num_steps)
                writer.add_scalar('Loss/entropy_loss', entropy_losses/args.log_interval, total_num_steps)
                policy_losses = 0.
                value_losses = 0.
                entropy_losses = 0.
            # save for every interval-th episode or for the last epoch
            if (j % args.save_interval == 0 or j == num_updates - 1):
                total_num_steps = (j + 1) * args.num_processes * args.num_steps
                mean_eval_reward = evaluate(actor_critic, args.seed, args.num_processes, device, args.env_settings)
                writer.add_scalar('Evaluate/reward', mean_eval_reward, total_num_steps)
                if mean_eval_reward>best_eval_reward:
                    best_eval_reward = mean_eval_reward
                    torch.save({
                        'num_updates': j,
                        'best_eval_reward': best_eval_reward,
                        'model_state_dict': actor_critic.base.state_dict(),
                        'optimizer_state_dict': agent.optimizer.state_dict(),
                        }, os.path.join(save_dir, 'best'))
                else:
                    torch.save({
                        'num_updates': j,
                        'best_eval_reward': best_eval_reward,
                        'model_state_dict': actor_critic.base.state_dict(),
                        'optimizer_state_dict': agent.optimizer.state_dict(),
                        }, os.path.join(save_dir, 'current'))

    except KeyboardInterrupt:
        print('\nReceived KeyboardInterrupt')
        interrupted=True
    finally:
        print('\nTraining stopped, saving model...')
        writer.close()
        torch.save({
            'num_updates': j,
            'best_eval_reward': best_eval_reward,
            'model_state_dict': actor_critic.base.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            }, os.path.join(save_dir, 'current'))
        envs.close(interrupted)   


if __name__ == "__main__":
    main()
