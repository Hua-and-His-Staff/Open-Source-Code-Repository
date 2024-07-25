import numpy as np
from multiprocessing import Process
from multiprocessing.connection import Client,Listener

dt=1./12
N=10
M=5
def ReLU(x):
    return x * (x > 0)

def ESloss(soc):
    return 0.13*soc-0.08*(ReLU(soc-0.5))**2

#equipment
#M EVs, N ACs.
    #overall: PV, DG, BES
    #users: basic Load, EV, air conditioning
#obs: 
#GRU: t[0], pure_gen[1], T_out[2], ACs_sign[3:3+N]
#MLP: t[0], P_G[1], SOC[2]  EVs_demand[2:2+M] 
#actions: u_G, (P_EV)*M, (u_{AC})*N 

class EIEnv():
    def __init__(self):
        self.pure_gen = np.load('./Data/pure_gen.npy')
        self.temp = np.load('./Data/temp.npy')
        self.ACs_parameter = np.load('./Data/ACs_parameter.npy')
        #a,b,mu,Tmax,Tmin,lambda,Pmax
        self.total_steps = int(24/dt)
        self.num_days = int(len(self.pure_gen)/self.total_steps)-10
        self.T_g=1./3
        self.max_g=20
        self.max_PBES = 20
        self.Q=50
        self.eta_charge=0.95
        self.eta_discharge=0.95       
        self.done=False
        #self.ACs_Pmax = np.concatenate((1*np.ones(2), 2*np.ones(5),3*np.ones(3)), axis=0)
        #self.EVs_Pmax = np.concatenate((3.4*np.ones(2), 6.8*np.ones(2),2*np.ones(1)), axis=0)
        self.EVs_parameter = np.load('./Data/EVs_parameter.npy')
        #obs_GRU: t[0], P_PV-PLsum[1], T_out[2] ACs_sign[3:3+N]
        #obs_MLP: t[0], P_G[1], SOC[2], EVs_demand[3:3+M]
        #actions: u_G, (P_EV)*M, (u_{AC})*N 
        self.obs_GRU_scalar=lambda x:(x-np.concatenate((np.array([0,0,20]), np.zeros(N)),axis=0))/np.concatenate((np.array([24,20,10]), np.ones(N)),axis=0)
        self.obs_MLP_scalar=lambda x:x/np.concatenate((np.array([24,self.max_g,1]), 10*np.ones(M)),axis=0)
        
        
    def seed(self,seed=None):
        np.random.seed(seed)
    
    def move(self,actions):
        # get current time
        if self.done == True:
            return self._get_obs(), 0., True, None
        costs = np.insert(0.2*(ReLU(actions-1)+ReLU(-actions)),obj=1,values=0)
        actions = np.clip(actions,0,a_max=1)
        u_g = actions[0]
        step = self.index % self.total_steps
        soc = self.soc 
        
        #EV
        #calculate EV costs
        EVs_power = actions[1:1+M]*self.EVs_parameter[2]
        costs[2:2+M] += (self.EVs_end==step) * (self.EVs_parameter[0]*self.EVs_demand + self.EVs_parameter[1]* (self.EVs_demand)**2)
        #仅在需求大于0时存在充电功率
        EVs_power = (self.EVs_demand>0)*EVs_power

        #AC
        ACs_action = actions[1+M:]
        self.ACs_sign = (self.ACs_start<=step) & (self.ACs_end>=step)
        #costs[-N:] += 
        costs[-N:] += self.ACs_sign*( (self.ACs_T>=self.ACs_parameter[3]+5)*(1-ACs_action) + (self.ACs_T<=self.ACs_parameter[4]-5)*ACs_action + self.ACs_parameter[5]* ((self.ACs_T>self.ACs_parameter[3])* np.exp(self.ACs_parameter[2]*np.clip(self.ACs_T-self.ACs_parameter[3],0,a_max=5)) +
         (self.ACs_T<self.ACs_parameter[4])*np.exp(self.ACs_parameter[2]*np.clip(self.ACs_parameter[4]-self.ACs_T,0,a_max=5))))*dt
                
        ACs_power = ACs_action*self.ACs_parameter[6]
        
        costs[0] += (0.5*self.DG_power + 0.0125*self.DG_power**2)*dt 
        
        BES_power = self.pure_gen[self.index] + self.DG_power - np.sum(EVs_power) - np.sum(ACs_power) 
        if (soc>=1) & (BES_power>0):#弃电免费
            new_soc = soc
        elif (soc<=0) & (BES_power<0):
            new_soc = soc
            costs[1] -= BES_power*2*dt
        else:
            new_soc = soc + ((BES_power>0)*self.eta_charge+(BES_power<0)/self.eta_discharge)*BES_power*dt/self.Q
            costs[1] += abs(ESloss(soc)-ESloss(new_soc))*self.Q 
            
        self.index += 1
        step+=1
        if step==self.total_steps:
            self.done=True
        else:
            # update the state
            
            self.DG_power += -1/self.T_g*(self.DG_power-u_g*self.max_g)*dt
            self.soc = new_soc
            #E_EV
            self.EVs_demand = ReLU(self.EVs_demand-EVs_power*dt)
            #newly generated and end by users
            self.EVs_demand = (self.EVs_start==step)*self.EVs_required + (self.EVs_end>=step)*self.EVs_demand
            self.ACs_T -= self.ACs_parameter[0]*(self.ACs_T- self.temp[self.index]) + self.ACs_parameter[1]*ACs_power    
            
        t = dt*step
        GRU = np.array([t,self.pure_gen[self.index],self.temp[self.index]])
        MLP = np.array([t,self.DG_power,self.soc])   
        
        return np.concatenate((GRU, self.ACs_sign), axis=0), np.concatenate((MLP, self.EVs_demand), axis=0), 5*costs, self.done, None
    
    def _get_obs(self):
        t = dt*(self.index%self.total_steps)
        GRU = np.array([t,self.pure_gen[self.index],self.temp[self.index]])
        MLP = np.array([t,self.DG_power,self.soc])
        return np.concatenate((GRU, self.ACs_sign), axis=0), np.concatenate((MLP, self.EVs_demand), axis=0)

 
  
    def reset(self):
        self.index = self.total_steps*np.random.randint(self.num_days)
        self.DG_power = 5
        self.soc = 0.1+0.8*np.random.rand()
        self.EVs_start = 9*12 + np.random.randint(3*12,size=M)
        self.EVs_end = self.EVs_start+2*12+np.random.randint(4*12,size=M)
        self.EVs_required = (self.EVs_end-self.EVs_start)*dt*self.EVs_parameter[2]*(0.3+0.5*np.random.rand(M))
        self.EVs_demand = np.zeros(M)
        self.ACs_start = 10*12+np.random.randint(2*12,size=N)
        self.ACs_end = self.ACs_start+2*12+2*np.random.randint(2*12,size=N)
        self.ACs_sign = np.zeros(N)
        self.ACs_T = self.temp[self.index]*np.ones(N)
        self.done=False   
        return self._get_obs()
    
    def reset_evaluation(self,soc=0.5):
        self.done=False
        return self._get_obs()
    

        

def worker(remote_addr, env_wrapper):
    env = env_wrapper.x()
    local_conn=Client(remote_addr)
    try:
        while True:
            cmd, data = local_conn.recv()
            if cmd == 'move':
                obs_GRU, obs_MLP, costs, done, info = env.move(data)
                obs_GRU=env.obs_GRU_scalar(obs_GRU)
                obs_MLP=env.obs_MLP_scalar(obs_MLP)
                local_conn.send((obs_GRU, obs_MLP, costs, done, info))
            #elif cmd == 'multistep':
            #    observation, costs, done, info = env.multiple_step(*data)
            #    observation=env.observation_scalar(observation)
            #    local_conn.send((observation, costs, done, info))             
            elif cmd == 'reset':
                obs_GRU, obs_MLP = env.reset()
                obs_GRU=env.obs_GRU_scalar(obs_GRU)
                obs_MLP=env.obs_MLP_scalar(obs_MLP)
                local_conn.send((obs_GRU, obs_MLP))
            elif cmd == 'observe':
                obs_GRU, obs_MLP = env._get_obs()
                obs_GRU=env.obs_GRU_scalar(obs_GRU)
                obs_MLP=env.obs_MLP_scalar(obs_MLP)
                local_conn.send((obs_GRU, obs_MLP))
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
        
        #self.remotes[0].send(('get_spaces', None))
        #self.action_space, self.observation_space = self.remotes[0].recv()


    def move(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('move', action))
        results = [remote.recv() for remote in self.remotes]
        obs_GRU, obs_MLP, costs, dones, infos = zip(*results)

        return np.stack(obs_GRU), np.stack(obs_MLP), np.stack(costs), np.stack(dones), infos
    
    #def multistep(self, actions,length=10):
    #    for remote, action in zip(self.remotes, actions):
    #        remote.send(('multistep', (action, length)))
    #    results = [remote.recv() for remote in self.remotes]
    #    obs, costs, dones, infos = zip(*results)
    #
    #    return np.stack(obs), np.stack(costs), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs_GRU, obs_MLP = zip(*results)
        return np.stack(obs_GRU), np.stack(obs_MLP)

    #def partial_reset(self,dones):
    #    for remote,done in zip(self.remotes,dones):
    #        if done:
    #            remote.send(('reset', None))
    #        else:
    #           remote.send(('observe',None))
    #    return np.stack([remote.recv() for remote in self.remotes])

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