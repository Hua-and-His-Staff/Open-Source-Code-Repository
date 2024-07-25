import numpy as np
import torch

from a2c import utils
from a2c.envs import make_vec_envs


def evaluate(actor_critic, seed, num_processes, device, env_settings):
    eval_envs = make_vec_envs(seed + num_processes, num_processes, env_settings)

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_rewards) < 80:
        with torch.no_grad():
            _, action, eval_recurrent_hidden_states = actor_critic.act(
                obs.to(device,dtype=torch.float),
                eval_recurrent_hidden_states,
                eval_masks)
        # Obser reward and next obs
        obs, _, done, infos = eval_envs.move(action.detach().cpu().numpy())
        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float,
            device=device)
        for info in infos:
            if info is not None:
                eval_episode_rewards.append(info)
        obs = eval_envs.partial_reset(done)
    eval_envs.close()

    return np.mean(eval_episode_rewards)
