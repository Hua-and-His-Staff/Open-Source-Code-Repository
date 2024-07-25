import numpy as np
import torch
from microgrid import MG_for_test
from env_compared import MG_compared_for_test
from a2c import utils
from a2c.envs import SubprocVecEnv

def make_vec_envs(seed,num_processes,env_settings):
    envs = [make_env(seed, i, env_settings) for i in range(num_processes)]
    envs = SubprocVecEnv(envs)
    return envs


def make_env(seed, rank, env_settings):
    def _thunk():
        if env_settings['compared'] == True:
            env = MG_compared_for_test(env_settings['assign_credit'])
        else:
            env = MG_for_test(env_settings['assign_credit'], env_settings['privacy_preserving'])
        env.seed(seed + rank)
        return env
    return _thunk

def evaluate(actor_critic, seed, num_processes, device, env_settings):
    eval_envs = make_vec_envs(seed, num_processes, env_settings)

    eval_episode_consumption = []
    eval_episode_loss = []
    costs = torch.zeros(16)
    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_loss) < 80:
        with torch.no_grad():
            _, action, eval_recurrent_hidden_states = actor_critic.act(
                obs.to(device,dtype=torch.float),
                eval_recurrent_hidden_states,
                eval_masks)
        # Obser reward and next obs
        obs, cost, done, infos = eval_envs.move(action.detach().cpu().numpy())
        costs += cost.sum(axis=0)
        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float,
            device=device)
        for info in infos:
            if info is not None:
                eval_episode_consumption.append(info['consumption'])
                eval_episode_loss.append(info['loss'])
                #eval_user_rewards.append(user_reward)
        obs = eval_envs.partial_reset(done)
    eval_envs.close()

    return np.mean(eval_episode_consumption), np.mean(eval_episode_loss)
