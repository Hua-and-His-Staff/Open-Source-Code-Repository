import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from multiprocessing import Process
from multiprocessing.connection import Client,Listener



dt=1./60
eta_asterisk=1/dt
alpha=1.15
C=600
m=30
n=20
def ReLU(x):
    return x * (x > 0)

def OrsteinUhlenbeckStep(xt,theta,sigma,dw,dt=1./60):
    new_xt=xt*np.exp(-theta*dt)+sigma*np.sqrt((1-np.exp(-2*theta*dt))/(2*theta))*dw
    return new_xt

#equipment
#n EVs, m ACs.
    #overall: PV, G, BES
    #users: basic Load, EV, air conditioning
#state: t(state[0]), P_PV(state[1]), P_G(state[2]), SOC(state[3]), PLsum(state[4]), T_{out}(state[5]), E_EV*n(state[6:6+n]), T_EV*n(state[6+n:6+2*n]), T*m(state[6+2*n:]), 
#action: u_G, (P_EV)*n, (u_{AC})*m 
class EIEnv(gym.Env):
    def __init__(self,gamma=0.9):
        self.pvdata=np.load('./Data/PV_sum.npy')
        self.loaddata=np.load('./Data/load_sum.npy')
        self.tempdata=np.load('./Data/temp.npy')
        days, self.total_step=self.pvdata.shape
        self.train_days = int(days*0.95)
        self.gamma=gamma       
        self.dt=1./60
        self.n = 20
        self.m = 30
        self.index=0
        self.T_g=10./60
        self.max_g=20
        self.max_PBES = 20
        self.Q=1000
        self.eta_0=0.98
        self.eta_1=0.05
        self.T=self.total_step*self.dt        
        self.done=False
        self.fail=False
        self.price = 1
        self.AC_a = np.load('./Data/para_a.npy')
        self.AC_b = np.load('./Data/para_b.npy')
        self.Tupper = np.load('./Data/Tupper.npy')
        self.Tlower = np.load('./Data/Tlower.npy')
        self.mu = 3*np.load('./Data/mu.npy')
        self.mu_constant = self.mu*np.exp(5*self.mu)
        self.action_space=spaces.Box(low=np.zeros(n+m+1),high=np.ones(n+m+1),dtype=np.float32)
        self.PEVmax = np.concatenate((3.4*np.ones(5), 6.8*np.ones(10),2*np.ones(5)), axis=0)
        self.PACmax = np.concatenate((1*np.ones(10), 2*np.ones(10),3*np.ones(10)), axis=0)
        high = np.concatenate((np.array([self.T,0,self.max_g,1,0,0]), np.zeros(2*n), 30*np.ones(m)), axis=0)
        low =  np.concatenate((np.array([0,0,0,0,0,0]), np.zeros(2*n), 20*np.ones(m)),axis=0)
        self.observation_space=spaces.Box(low=low,high=high,dtype=np.float32)
        self.observation_scalar=lambda x:x/np.concatenate((np.array([24,100,self.max_g,1,20,10]), self.PEVmax, np.ones(n), 10*np.ones(m)),axis=0)
        self.seed()
        
    def seed(self,seed=None):
        self.np_random,seed = seeding.np_random(seed)
        return [seed]
    
    def step(self,action):
        # get current time
        if self.done == True:
            return self._get_obs(), 0., True, {'fail':False}
        index=self.index
        #state: t, P_PV, P_G, SOC, PLsum, T_{out}, (E_EV)*n,(T_EV)*n, (T_{in})*m
        #if AC out of control,T_{in} = T_{out}
        t = self.state[0]
        PV = self.state[1]
        PG = self.state[2]
        soc = self.state[3]
        PLsum = self.state[4]
        Tout = self.state[5]
        E_EV = self.state[6:6+n]
        T_EV = self.state[6+n:6+2*n]
        Tin = self.state[-self.m:] 
        #action: u_G, (P_EV)*n, (u_{AC})*m 
        u_g = action[0]
        #AC
        self.EV_sign = (self.start_charge_time<t) & (self.end_charge_time>t)
        self.AC_sign = (self.start_AC_time<t) & (self.end_AC_time>t)
        P_EV = self.EV_sign*action[1:1+n]*self.PEVmax


        for i in range(n):
            if self.EV_sign[i]==True:
                if E_EV[i]<=0:
                    P_EV[i] = 0
                elif E_EV[i]>=self.PEVmax[i]*T_EV[i]:
                    P_EV[i] = self.PEVmax[i]
        P_AC = self.AC_sign*action[1+n:]*self.PACmax

        P_BES = PV + PG - np.sum(P_EV) - np.sum(P_AC) - PLsum
        bes_cost = np.zeros(1)
        if (soc>=1) & (P_BES>0):#弃电免费
            new_soc = soc
            bes_cost += 10
            self.fail = True
        elif (soc<=0) & (P_BES<0):
            new_soc = soc
            bes_cost -= P_BES*self.price
            bes_cost += 10
            self.fail = True
        else:
            eta = self.eta_0 - self.eta_1*abs(P_BES/self.max_PBES)
            new_soc = soc + ((P_BES>0)*eta+(P_BES<0)/eta)*P_BES*dt/self.Q
            bes_cost += 1e-5*P_BES**2 + 10*(soc - 0.5)**2
            self.fail = False
        # put forward the time step
        
        gen_cost = 0.5*PG + 5e-3*PG**2 

        Tdiff = np.abs(Tin - np.clip(Tin, self.Tlower, self.Tupper))
        T5 = np.clip(Tdiff,a_min=None,a_max=5)#容忍范围最大不超过5度
        comfort_cost = (np.exp( self.mu*(T5)) + self.mu_constant*(Tdiff-T5) )*self.AC_sign
        
            
        # update the state
        self.state[0] +=self.dt
        self.index += 1
        self.state[1] = self.PV[self.index]
        self.state[2] += -1/self.T_g*(PG-u_g*self.max_g)*self.dt
        self.state[3] = new_soc
        self.state[4] = self.load[self.index]
        self.state[5] = self.temp[int(self.state[0])]
        #E_EV
        self.state[6:6+n] -= self.EV_sign*P_EV*self.dt
        self.state[6+n:6+2*n] -= self.EV_sign*self.dt
        self.state[-self.m:] -= (self.AC_a*(Tin-Tout) + self.AC_b*P_AC)*self.dt
        if self.index==self.total_step-1:
            self.done=True        
        #print(self.state)
        return self._get_obs(), 0.1*gen_cost, 0.1*bes_cost, 0.1*comfort_cost, self.done, {'fail':self.fail}
    

    
    def multiple_step(self,action,length=10):
        rewards=[]
        for inner_step in range(length):
            observation, reward, done, info = self.step(action)
            rewards.append(reward)
        total_reward=0.
        for reward in reversed(rewards):
            total_reward=total_reward*self.gamma+reward
        return observation,total_reward,done,info
    
    
#state: t, P_PV, P_G, SOC, PLsum, T_{out}, (E_EV)*n,(T_EV)*n, (T)*m,   
    def reset(self):
        self.index = 0
        day = np.random.randint(self.train_days)
        self.PV = self.pvdata[day]
        self.load = self.loaddata[day]
        self.temp = self.tempdata[np.random.randint(len(self.tempdata))]
        self.start_charge_time = 10+8*np.random.rand(n)
        self.min_charge_period = 2+np.random.rand(n)
        self.end_charge_time = np.clip(self.start_charge_time+self.min_charge_period+0.5+2*np.random.rand(n), 0,24)
        self.start_AC_time = 15+2*np.random.rand(m)
        self.end_AC_time = 18+2*np.random.rand(m)
        self.EV_sign = np.zeros(n, dtype=bool) 
        self.AC_sign = np.zeros(m, dtype=bool) 
        self.done=False
        self.fail=False
        self.state=np.zeros(6+m+2*n)
        self.state[1] = self.PV[0]
        self.state[2] = 0.5*self.max_g
        self.state[3] = 0.5
        self.state[4] = self.load[0]       
        self.state[5] = self.temp[0]
        self.state[6:6+n] = self.min_charge_period*self.PEVmax
        self.state[6+n:6+2*n] = self.end_charge_time-self.start_charge_time
        self.state[-m:] = self.temp[0]
        return self._get_obs()
    
    def preset(self,soc=0.5):
        self.done=False
        self.fail=False
        self.state=self.observation_space.sample()
        self.index=0
        self.state[0]=0
        self.state[-1]=soc
        return self._get_obs()
    
    def _get_obs(self):
        # get the system dynamics
        #t,e_pv,e_load,g0,g1,g2,soc=self.state
        return np.concatenate((self.state[:6], self.state[6:6+n]*self.EV_sign, self.state[6+n:6+2*n]*self.EV_sign, self.state[-m:]*self.AC_sign), axis=0)
        

def worker(remote_addr, env_wrapper):
    env = env_wrapper.x()
    local_conn=Client(remote_addr)
    try:
        while True:
            cmd, data = local_conn.recv()
            if cmd == 'step':
                observation, reward, done, info = env.step(data)
                observation=env.observation_scalar(observation)
                local_conn.send((observation, reward, done, info))
            elif cmd == 'multistep':
                observation, reward, done, info = env.multiple_step(*data)
                observation=env.observation_scalar(observation)
                local_conn.send((observation, reward, done, info))             
            elif cmd == 'reset':
                observation = env.reset()
                observation=env.observation_scalar(observation)
                local_conn.send(observation)
            elif cmd == 'observe':
                observation = env._get_obs()
                observation=env.observation_scalar(observation)
                local_conn.send(observation)
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
        
        
class SubprocVecEnv(object):
    def __init__(self, env_fns):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.closed = False
        nenvs = len(env_fns)
        self.listener=Listener()
        
        self.ps = [Process(target=worker, args=(self.listener.address, CloudpickleWrapper(env_fn)))
            for env_fn in env_fns]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        self.remotes=[self.listener.accept() for _ in range(nenvs)]
        self.listener.close()
        
        self.remotes[0].send(('get_spaces', None))
        self.action_space, self.observation_space = self.remotes[0].recv()


    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)

        return np.stack(obs), np.stack(rews), np.stack(dones), infos
    
    def multistep(self, actions,length=10):
        for remote, action in zip(self.remotes, actions):
            remote.send(('multistep', (action, length)))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)

        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def partial_reset(self,dones):
        for remote,done in zip(self.remotes,dones):
            if done:
                remote.send(('reset', None))
            else:
                remote.send(('observe',None))
        return np.stack([remote.recv() for remote in self.remotes])

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