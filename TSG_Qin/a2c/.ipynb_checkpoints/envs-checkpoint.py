import os
from microgrid import MG
from env_compared import MG_compared
import gym
import numpy as np
import torch


from multiprocessing import Process
from multiprocessing.connection import Client,Listener


def make_vec_envs(seed,num_processes,env_settings):
    envs = [make_env(seed, i, env_settings) for i in range(num_processes)]
    envs = SubprocVecEnv(envs)
    return envs


def make_env(seed, rank, env_settings):
    def _thunk():
        if env_settings['compared'] == True:
            env = MG_compared(env_settings['assign_credit'])
        else:
            env = MG(env_settings['assign_credit'], env_settings['privacy_preserving'])
        env.seed(seed + rank)
        return env
    return _thunk


      
class SubprocVecEnv(object):
    def __init__(self, envs):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.closed = False
        nenvs = len(envs)
        self.listener=Listener()
        
        self.ps = [Process(target=worker, args=(self.listener.address, CloudpickleWrapper(env)))
            for env in envs]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        self.remotes=[self.listener.accept() for _ in range(nenvs)]
        self.listener.close()
        
        self.remotes[0].send(('get_spaces', None))
        self.action_space, self.observation_space = self.remotes[0].recv()


    def move(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('move', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, dones, infos = zip(*results)

        return torch.tensor(obs,dtype=torch.float), torch.tensor(rewards).view(self.num_envs,-1), dones, infos
    
    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return torch.tensor([remote.recv() for remote in self.remotes])

    def partial_reset(self,dones):
        for remote,done in zip(self.remotes,dones):
            if done:
                remote.send(('reset', None))
            else:
                remote.send(('observe',None))
        return torch.tensor([remote.recv() for remote in self.remotes])

    def close(self,interrupted=False):
        if self.closed:
            return

        for remote in self.remotes:
            if not interrupted:
                remote.send(('close',None))
            remote.close()
            
        for p in self.ps:
            p.join()
        self.closed = True

    @property
    def num_envs(self):
        return len(self.remotes)



def worker(remote_addr, env_wrapper):
    env = env_wrapper.x()
    local_conn=Client(remote_addr)
    try:
        while True:
            cmd, data = local_conn.recv()
            if cmd == 'move':
                obs, rewards, done, info = env.move(data)
                obs_norm=env.obs_scalar(obs)
                local_conn.send((obs_norm, rewards, done, info))
            #elif cmd == 'multistep':
            #    observation, rewards, done, info = env.multiple_step(*data)
            #    observation=env.observation_scalar(observation)
            #    local_conn.send((observation, rewards, done, info))             
            elif cmd == 'reset':
                obs = env.reset()
                obs_norm=env.obs_scalar(obs)
                local_conn.send(obs_norm)
            elif cmd == 'observe':
                obs = env._get_obs()
                obs_norm=env.obs_scalar(obs)
                local_conn.send(obs_norm)
            elif cmd == 'close':
                fd=local_conn.fileno()
                local_conn.close()
                print('\nlocal connection %s is closed' %fd)
                break
            elif cmd == 'get_spaces':
                local_conn.send((env.action_space, env.observation_space))
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        fd=local_conn.fileno()
        local_conn.close()
        print('\nSubprocVecEnv worker %s: KeyboardInterrupt received, and local connection is closed'%fd)
    finally:
        pass

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)
        
