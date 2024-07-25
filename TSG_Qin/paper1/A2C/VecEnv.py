import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from multiprocessing import Process
from multiprocessing.connection import Client,Listener



dt=1./60
N=10
M=5
def ReLU(x):
    return x * (x > 0)

def OrsteinUhlenbeckStep(xt,theta,sigma,dw,dt=1./60):
    new_xt=xt*np.exp(-theta*dt)+sigma*np.sqrt((1-np.exp(-2*theta*dt))/(2*theta))*dw
    return new_xt

#equipment
#M EVs, N ACs.
    #overall: PV, G, BES
    #users: basic Load, EV, air conditioning
#obs: t[0], P_PV-PLsum[1], P_G[2], SOC[3], T_out[4] E_EV[5:5+M], AC_open_time[-N:]
#actions: u_G, (P_EV)*M, (u_{AC})*N 
class EIEnv(gym.Env):
    def __init__(self,gamma=0.9):
        pvdata=np.load('./Data/PV_sum.npy')
        loaddata=np.load('./Data/load_sum.npy')
        tempdata=np.load('./Data/temp.npy')
        self.PV = 0.3*pvdata[0]
        self.load = loaddata[0]
        self.temp = tempdata[0]
        self.total_step=len(self.PV)
        self.gamma=gamma       
        self.dt=dt
        self.AC_number = N
        self.EV_number = M
        self.index=0
        self.T_g=10./60
        self.max_g=20
        self.max_PBES = 20
        self.Q=200
        self.eta_0=0.98
        self.eta_1=0.05
        self.T=self.total_step*self.dt        
        self.done=False
        self.fail=False
        self.price = 1
        self.AC_a = np.load('./Data/para_a.npy')[:N]
        self.AC_b = np.load('./Data/para_b.npy')[:N]
        self.Tupper = np.load('./Data/Tupper.npy')[:N]
        self.Tlower = np.load('./Data/Tlower.npy')[:N]
        self.mu = 4*np.load('./Data/mu.npy')[:N]
#obs: t[0], P_PV-PLsum[1], P_G[2], SOC[3], T_out[4] E_EV[5:5+M], AC_open_time[-N:]
#actions: u_G, (P_EV)*M, (u_{AC})*N 
        self.action_space=spaces.Box(low=np.zeros(M+N+1),high=np.ones(M+N+1),dtype=np.float32)
        self.PEVmax = np.concatenate((3.4*np.ones(2), 6.8*np.ones(2),2*np.ones(1)), axis=0)
        self.PACmax = np.concatenate((1*np.ones(2), 2*np.ones(5),3*np.ones(3)), axis=0)
        high = np.concatenate((np.array([self.T,0,self.max_g,1,0]), np.zeros(M), 30*np.ones(N)), axis=0)
        low =  np.concatenate((np.array([0,0,0,0,0]), np.zeros(M), np.zeros(N)),axis=0)
        self.observation_space=spaces.Box(low=low,high=high,dtype=np.float32)
        self.observation_scalar=lambda x:x/np.concatenate((np.array([24,10,self.max_g,1,10]), np.ones(M+N)),axis=0)
        
        
    def seed(self,seed=None):
        np.random.seed(seed)
    
    def step(self,action):
        # get current time
        if self.done == True:
            return self._get_obs(), 0., True, {'fail':False}
#obs: t[0], P_PV-PLsum[1], P_G[2], SOC[3], T_out[4] E_EV[5:5+M], AC_open_time[-N:]
#action: u_G, (P_EV)*M, (u_{AC})*N 
        t = self.dt*self.index
        soc = self.soc 
        Tout = self.temp[int(self.index/60)]
        action = np.clip(action,0,a_max=1)
        u_g = action[0]
        #EV
        #calculate EV costs
        EV_costs = np.zeros(M)
        self.P_EV = self.EV_sign*action[1:1+M]*self.PEVmax
        for i in range(M):
            if self.EV_sign[i]==True:
                if self.E_EV[i]<=0:
                    self.P_EV[i]=0
                if self.end_charge_time[i]<=t:
                    EV_costs[i] = 5*(ReLU(self.E_EV[i]))**2
        self.EV_sign = (self.start_charge_time<t) & (self.end_charge_time>t)


        #AC
        self.AC_sign = (self.start_AC_time<t) & (self.end_AC_time>t)
        AC_costs = np.zeros(N)
        AC_actions = np.zeros(N)
        for i in range(N):
            #若不在控制时间内，保持控制信号为0
            if self.AC_sign[i]==False:
                AC_costs[i] = 0.2*(action[1+M+i])**2
                AC_actions[i] = 0
            #若超过3度，接管控制，并惩罚与控制相反的控制信号
            elif self.T_in[i]>=self.Tupper[i]+3:
                AC_costs[i] += 2*(1-action[1+M+i]) + 0.5*(np.exp(3*self.mu[i]))
                AC_actions[i] = 1
            elif self.T_in[i]<=self.Tlower[i]-3:
                AC_costs[i] += 2*(action[1+M+i]) + 0.5*(np.exp(3*self.mu[i]))
                AC_actions[i] = 0
            elif self.T_in[i]>self.Tupper[i]:
                Tdiff = self.T_in[i]-self.Tupper[i]
                AC_costs[i] = (1-action[1+M+i])+ 0.5*(np.exp( self.mu[i]*Tdiff))
                AC_actions[i] = action[1+M+i]
            elif self.T_in[i]<self.Tlower[i]:
                Tdiff = self.Tlower[i]-self.T_in[i]
                AC_costs[i] = action[1+M+i]+0.5*(np.exp( self.mu[i]*Tdiff))
                AC_actions[i] = action[1+M+i]
                #温度在Tlower~Tupper之间没有损失

                
        self.P_AC = AC_actions*self.PACmax
        
        gen_cost = 0.1*(0.5*self.PG + 1e-2*self.PG**2) 
        
        P_BES = self.PV[self.index]- self.load[self.index] + self.PG - np.sum(self.P_EV) - np.sum(self.P_AC) 
        bes_cost = 0
        if (soc>=1) & (P_BES>0):#弃电免费
            new_soc = soc
            bes_cost += 5
            self.fail = True
        elif (soc<=0) & (P_BES<0):
            new_soc = soc
            bes_cost -= P_BES*self.price
            bes_cost += 5
            self.fail = True
            gen_cost+=5*(1-u_g)**2
        else:
            eta = self.eta_0 - self.eta_1*abs(P_BES/self.max_PBES)
            new_soc = soc + ((P_BES>0)*eta+(P_BES<0)/eta)*P_BES*dt/self.Q
            bes_cost += 1e-4*P_BES**2 + 20*(soc - 0.5)**2
            self.fail = False
            if soc<0.2:
                gen_cost+=5*(1-u_g)**2
        # put forward the time step
        
        

        
        
        #提前十分钟开始动作，之后计算损失
        #compute_cost_if = (self.start_AC_time<t-1/6) & (self.end_AC_time>t)
        
        # update the state
        self.index += 1
        self.PG += -1/self.T_g*(self.PG-u_g*self.max_g)*self.dt
        self.soc = new_soc
        #E_EV
        self.E_EV -= self.EV_sign*self.P_EV*self.dt
        self.T_in -= (self.AC_a*(self.T_in-Tout) + self.AC_b*self.P_AC)*self.dt
        if self.index==self.total_step-1:
            bes_cost += 1000*(0.5-new_soc)**2
            self.done=True        
        return self._get_obs(), np.concatenate((np.array([gen_cost, bes_cost]), EV_costs, AC_costs), axis=0), self.done, {'fail':self.fail}


    
    def multiple_step(self,action,length=10):
        rewards=[]
        for inner_step in range(length):
            observation, reward, done, info = self.step(action)
            rewards.append(reward)
        total_reward=0.
        for reward in reversed(rewards):
            total_reward=total_reward*self.gamma+reward
        return observation,total_reward,done,info
    
  
    def reset(self):
        self.index = 0
        self.PG = 0.5*self.max_g
        self.soc = 0.5
        self.T_in = self.temp[0]*np.ones(N)
        self.start_charge_time = 8+4*np.random.rand(M)
        self.min_charge_period = 2+2*np.random.rand(M)
        self.E_EV = self.min_charge_period*self.PEVmax
        #self.start_charge_time+self.min_charge_period
        self.end_charge_time = 16+4*np.random.rand(M)
        self.start_AC_time = 14+2*np.random.rand(N)
        self.end_AC_time = 20+2*np.random.rand(N)
        self.EV_sign = np.zeros(M, dtype=bool) 
        self.AC_sign = np.zeros(N, dtype=bool) 
        self.done=False
        self.fail=False    
        return self._get_obs()
    
    def preset(self,soc=0.5):
        self.done=False
        self.fail=False
        self.state=self.observation_space.sample()
        self.index=0
        self.state[0]=0
        self.state[-1]=soc
        return self._get_obs()
    
#obs: t[0], P_PV-PLsum[1], P_G[2], SOC[3], T_out[4] E_EV[5:5+M], AC_open_time[-N:]   
    def _get_obs(self):
        t = self.dt*self.index
        PV_L = self.PV[self.index]-self.load[self.index]
        state = np.array([t,PV_L,self.PG,self.soc,self.temp[int(self.index/60)]])
        AC_open_time = (t-self.start_AC_time)*self.AC_sign
        return np.concatenate((state, self.E_EV*self.EV_sign , AC_open_time), axis=0)
        

def worker(remote_addr, env_wrapper):
    env = env_wrapper.x()
    local_conn=Client(remote_addr)
    try:
        while True:
            cmd, data = local_conn.recv()
            if cmd == 'step':
                observation, costs, done, info = env.step(data)
                observation=env.observation_scalar(observation)
                local_conn.send((observation, costs, done, info))
            elif cmd == 'multistep':
                observation, costs, done, info = env.multiple_step(*data)
                observation=env.observation_scalar(observation)
                local_conn.send((observation, costs, done, info))             
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
        obs, costs, dones, infos = zip(*results)

        return np.stack(obs), np.stack(costs), np.stack(dones), infos
    
    def multistep(self, actions,length=10):
        for remote, action in zip(self.remotes, actions):
            remote.send(('multistep', (action, length)))
        results = [remote.recv() for remote in self.remotes]
        obs, costs, dones, infos = zip(*results)

        return np.stack(obs), np.stack(costs), np.stack(dones), infos

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