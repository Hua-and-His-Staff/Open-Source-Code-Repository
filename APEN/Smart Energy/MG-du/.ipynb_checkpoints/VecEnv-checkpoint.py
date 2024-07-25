import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from multiprocessing import Process
from multiprocessing.connection import Client,Listener

Cap=2000
dt=1./60
eta_plus=0.98/Cap
eta_minus=1/0.97/Cap
eta_asterisk=1/dt
alpha=1.15
C=600
        
def ReLU(x):
    return x * (x > 0)

def OrsteinUhlenbeckStep(xt,theta,sigma,dw,dt=1./60):
    new_xt=xt*np.exp(-theta*dt)+sigma*np.sqrt((1-np.exp(-2*theta*dt))/(2*theta))*dw
    return new_xt

def costU(u):
    finfo=np.finfo(u.dtype)
    u=np.clip(u,-1+finfo.eps,1-finfo.eps)
    return 0.5*np.log(1-u**2)+u*np.arctanh(u)

def costG(P_G_ratio):
    return 0.5+0.8*P_G_ratio+0.5*P_G_ratio**2

# P_BES>0 : charging
def dSOC(P_BES):
    return ((P_BES>0)*P_BES*eta_plus+(P_BES<0)*P_BES*eta_minus)*dt

def dDOD(P_BES,DOD):
    return -((P_BES>0)*(DOD)*eta_asterisk+(P_BES<0)*P_BES*eta_minus)*dt

def dcostBES(P_BES,DOD):
    dod=ReLU(DOD)
    return alpha/C*(P_BES<0)*np.power(dod,alpha-1)*dDOD(P_BES,dod)

class EIEnv(gym.Env):
    def __init__(self,env_id=0,gamma=0.99):
        pvdata=np.load('./Data/PV_%d.npz'%env_id)['arr_0']
        loaddata=np.load('./Data/Load_%d.npz'%env_id)['arr_0']
        self.gamma=gamma
        self.total_step=len(pvdata)
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
        
        self.T_g0=10./60
        self.T_g1=20./60
        self.T_g2=10./60
        self.Amp_u0=50
        self.Amp_u1=50
        self.Amp_u2=50
        self.max_g0=500
        self.max_g1=500
        self.max_g2=500
        
        self.u0=0
        self.u1=0
        self.u2=0
        
        self.xiG=16
        self.xiBES=1.2e5
        
        self.done=False
        self.fail=False
        self.action_space=spaces.Box(low=np.array([-1,-1,-1]),high=np.array([1,1,1]),dtype=np.float32)
        # t, E_PV, E_Load, D_G0, D_G1, D_G2, DOD, SOC
        high = np.array([self.T-self.dt,0,0,300,300,300,0,  0.7])
        low =  np.array([0,0,0,100,100,100,0,  0.3])
        self.observation_space=spaces.Box(low=low,high=high,dtype=np.float32)
        self.observation_scaler=lambda x:x/np.array([24,1000,1000,500,500,500,1,1])
        self.seed()
        
    def seed(self,seed=None):
        self.np_random,seed = seeding.np_random(seed)
        return [seed]
    
    def step(self,action):
        # get current time
        index=self.index
        
        # get current state
        t,e_pv,e_load,g0,g1,g2,dod,soc=self.state
        
        # get current action
        #u=np.clip(action,self.action_space.low,self.action_space.high)
        u_g0,u_g1,u_g2=action
        
        # calculate the current power input/output of BES
        P_BES=self.pv_est[index]+self.p_sun[index]*e_pv-self.load_est[index]-e_load+g0+g1+g2
        
        # if the total process ended 
        if self.done:
            if self.fail:
                # if the control failed, add an extra penalty
                cost=300*self.dt
            else:
                # if the control successully finished, return final cost
                cost=((soc-0.5)**2)*100*self.dt
            # skip the calculation for new state, return 
            return self._get_obs(), cost, self.done, {'fail':self.fail}
        
        # calculate new states based on current state and action
        new_e_pv=OrsteinUhlenbeckStep(e_pv,self.pv_theta[index],self.pv_sigma[index],self.np_random.randn())
        new_e_load=OrsteinUhlenbeckStep(e_load,self.load_theta[index],self.load_sigma[index],self.np_random.randn())
        
        # update control input for generators
        self.u0+=self.Amp_u0*u_g0
        self.u1+=self.Amp_u1*u_g1
        self.u2+=self.Amp_u2*u_g2
        
        self.u0=np.clip(self.u0,0,self.max_g0)
        self.u1=np.clip(self.u1,0,self.max_g1)
        self.u2=np.clip(self.u2,0,self.max_g2)
        # calculate new generator output 
        new_g0=g0-1/self.T_g0*(g0-self.u0)*self.dt
        new_g1=g1-1/self.T_g1*(g1-self.u1)*self.dt
        new_g2=g2-1/self.T_g2*(g2-self.u2)*self.dt

        
        new_dod=dod+dDOD(P_BES,dod)
        new_soc=soc+dSOC(P_BES)
        # put forward the time step
        new_t=t+self.dt
        self.index+=1
        
        # calculate the cost regard the current state and action
        cost=0.0
        # state related costs
        # 1. price for power generation:
        # 0.5 ~ 1.8 * 3 * xiG * dt
        # 5 ~ 18 * dt
        cost+=(costG(g0/self.max_g0)+costG(g1/self.max_g1)+costG(g2/self.max_g0))*self.xiG*self.dt
        # 2. cost introduced by discharging of BES:
        # 0 ~ alpha/C*(P_BES<0)*np.power(dod,alpha-1)*|P_BES|*eta_minus)*xiBES*dt
        # 0 ~ 22 * dt
        cost+=dcostBES(P_BES,dod)*self.xiBES
        # 3. cost for power balance (if possible, we hope to maintain the power balance directly):
        # 0 ~ 1* 20 * dt
        #cost+=(abs(P_BES/Cap))*20*self.dt
        
        
        # penalty for action constraints:
        # 0 ~ 120+ * dt
        cost += costU(u_g0)*40*self.dt
        cost += costU(u_g1)*40*self.dt
        cost += costU(u_g2)*40*self.dt
        
        # penalty for constraints:
        # 0 ~ 60 * dt
        cost+=(ReLU(soc-0.8)+ReLU(0.2-soc))*300*self.dt
        
        # check the constraints, if any one of them is violated, set the flag done and fail to be True
        if new_soc >1 or new_soc<0:
            self.fail=True
            self.done=True
        
        # check the time step, if the whole period stops, set flag done to be True
        if self.index==self.total_step:
            self.done=True
            self.index=self.total_step-1
            
        # update the state
        self.state=np.array([new_t,new_e_pv,new_e_load,new_g0,new_g1,new_g2,new_dod,new_soc])
        return self._get_obs(), cost, self.done, {'fail':self.fail}
    
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
        self.done=False
        self.fail=False
        self.state=self.observation_space.sample()
        t,e_pv,e_load,g0,g1,g2,dod,soc=self.state
        self.index=int(t/self.dt)
        self.u0=g0
        self.u1=g1
        self.u2=g2
        return self._get_obs()
    
    def preset(self,soc=0.5):
        self.done=False
        self.fail=False
        self.state=self.observation_space.sample()
        _,_,_,g0,g1,g2,_,_=self.state
        self.index=0
        self.u0=g0
        self.u1=g1
        self.u2=g2
        self.state[0]=0
        self.state[-1]=soc
        return self._get_obs()
    
    def _get_obs(self):
        # get the system dynamics
        t,e_pv,e_load,g0,g1,g2,dod,soc=self.state
        index=self.index
        e_pv=min(e_pv,0.786)
        pv=self.pv_est[index]+self.p_sun[index]*e_pv
        load=self.load_est[index]+e_load
        return np.array([t,ReLU(pv),ReLU(load),g0,g1,g2,dod,soc])
        

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

    def partial_reset(self,masks):
        for remote,mask in zip(self.remotes,masks):
            if mask:
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