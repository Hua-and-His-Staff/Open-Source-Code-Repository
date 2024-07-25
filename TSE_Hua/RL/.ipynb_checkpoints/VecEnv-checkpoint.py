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
        
def ReLU(x):
    return x * (x > 0)

def OrsteinUhlenbeckStep(xt,theta,sigma,dw,dt=1./60):
    new_xt=xt*np.exp(-theta*dt)+sigma*np.sqrt((1-np.exp(-2*theta*dt))/(2*theta))*dw
    return new_xt

#def costU(u):
#    finfo=np.finfo(u.dtype)
#    u=np.clip(u,-1+finfo.eps,1-finfo.eps)
#    return 0.5*np.log(1-u**2)+u*np.arctanh(u)


#def dcostBES(P_BES,DOD):
#    dod=ReLU(DOD)
#    return alpha/C*(P_BES<0)*np.power(dod,alpha-1)*dDOD(P_BES,dod)

class EIEnv(gym.Env):
    def __init__(self,env_id=0,gamma=0.99):
        pvdata=np.load('./Data/PV_%d.npz'%env_id)['arr_0']
        loaddata=np.load('./Data/Load_%d.npz'%env_id)['arr_0']
        self.total_step=len(pvdata)
        self.gamma=gamma
        self.gammas = [gamma**i for i in range(self.total_step)]        
        self.dt=1./60
        self.T=self.total_step*self.dt
        self.index=0
        
        self.pv_est=pvdata[:,0]
        self.p_sun=pvdata[:,1]
        self.pv_theta=pvdata[:,2]
        self.pv_sigma=pvdata[:,3]
        
        self.load_est=loaddata[:,0]
        self.load_theta=loaddata[:,1]
        self.load_sigma=loaddata[:,2]
        #g0: MT, g1: FC, g2: DEG
        self.T_g0=20./60
        self.T_g1=25./60
        self.T_g2=30./60
        self.max_g0=200
        self.max_g1=300
        self.max_g2=400
        self.max_PBES = 200
        self.Q=1000
        self.eta_0=0.98
        self.eta_1=0.05
        self.xiG=16
        self.xiBES=1.2e5
        
        self.done=False
        self.fail=False
        self.action_space=spaces.Box(low=np.array([0,0,0]),high=np.array([1,1,1]),dtype=np.float32)
        # t, E_PV, E_Load, D_G0, D_G1, D_G2, SOC
        high = np.array([self.T-10*self.dt,0,0,self.max_g0,self.max_g1,self.max_g2,  0.7])
        low =  np.array([0,0,0,50,50,50,  0.3])
        self.observation_space=spaces.Box(low=low,high=high,dtype=np.float32)
        self.observation_scaler=lambda x:x/np.array([24,1000,1000,self.max_g0,self.max_g1,self.max_g2,1,200])
        self.seed()
        self.price = 5.
        
    def seed(self,seed=None):
        self.np_random,seed = seeding.np_random(seed)
        return [seed]
    
    def step(self,action):
        # get current time
        if self.done == True:
            return self._get_obs(), 0., True, {'fail':False}
        
        
        index=self.index
        
        # get current state
        t,e_pv,e_load,g0,g1,g2,soc=self.state
        
        # get current action
        u=np.clip(action,self.action_space.low,self.action_space.high)
        u_g0, u_g1, u_g2 = u
        
        # calculate the current power input/output of BES
        P_BES=self.pv_est[index]+self.p_sun[index]*e_pv-self.load_est[index]-e_load+g0+g1+g2
        
        
        # calculate new states based on current state and action
        new_e_pv=OrsteinUhlenbeckStep(e_pv,self.pv_theta[index],self.pv_sigma[index],self.np_random.randn())
        new_e_load=OrsteinUhlenbeckStep(e_load,self.load_theta[index],self.load_sigma[index],self.np_random.randn())
        
        # calculate new generator output 
        new_g0=g0-1/self.T_g0*(g0-u_g0*self.max_g0)*self.dt
        new_g1=g1-1/self.T_g1*(g1-u_g1*self.max_g1)*self.dt
        new_g2=g2-1/self.T_g2*(g2-u_g2*self.max_g2)*self.dt
        
        cost = 0.0
        #弃电免费
        if (soc>=1) & (P_BES>0):
            new_soc = soc
            cost+= 15
            self.fail = True
        elif (soc<=0) & (P_BES<0):
            new_soc = soc
            cost -= P_BES*self.price*self.dt
            cost += 15
            self.fail = True
        else:
            eta = self.eta_0 - self.eta_1*abs(P_BES/self.max_PBES)
            new_soc = soc + ((P_BES>0)*eta+(P_BES<0)/eta)*P_BES*dt/self.Q
            cost+= 5e-5*P_BES**2 + 10*(soc - 0.5)**2
            self.fail = False
        # put forward the time step
        new_t=t+self.dt
        self.index+=1
        
        cost+=(0.5*g0 + 5e-3*g0**2 + g1 + 3e-3*g1**2 + 1.5*g2 + 2e-3*g2**2)*self.dt

        if self.index==self.total_step-1:
            self.done=True
            
        # update the state
        self.state=np.array([new_t,new_e_pv,new_e_load,new_g0,new_g1,new_g2,new_soc])
        return self._get_obs(), 0.1*cost, self.done, {'fail':self.fail}
    
    def multiple_step(self,action,length=10):
        rewards=[]
        for inner_step in range(length):
            observation, reward, done, info = self.step(action)
            rewards.append(reward)
        total_reward=0.
        for reward in reversed(rewards):
            total_reward=total_reward*self.gamma+reward
        return observation,total_reward,done,info
    
    
    def reset(self,soc=0.5):
        self.done=False
        self.fail=False
        self.state=self.observation_space.sample()
        self.index=0
        self.state[0]=0
        self.state[-1]=soc
        return self._get_obs()
    
    def _get_obs(self):
        # get the system dynamics
        t,e_pv,e_load,g0,g1,g2,soc=self.state
        index=self.index
        e_pv=min(e_pv,0.786)
        pv=self.pv_est[index]+self.p_sun[index]*e_pv
        load=self.load_est[index]+e_load
        return np.array([t,ReLU(pv),ReLU(load),g0,g1,g2,soc,g0+g1+g2+ReLU(pv)-ReLU(load)])
        

def worker(remote_addr, env_wrapper):
    env = env_wrapper.x()
    local_conn=Client(remote_addr)
    try:
        while True:
            cmd, data = local_conn.recv()
            if cmd == 'step':
                observation, reward, done, info = env.step(data)
                observation=env.observation_scaler(observation)
                local_conn.send((observation, reward, done, info))
            elif cmd == 'multistep':
                observation, reward, done, info = env.multiple_step(*data)
                observation=env.observation_scaler(observation)
                local_conn.send((observation, reward, done, info))             
            elif cmd == 'reset':
                observation = env.reset()
                observation=env.observation_scaler(observation)
                local_conn.send(observation)
            elif cmd == 'observe':
                observation = env._get_obs()
                observation=env.observation_scaler(observation)
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