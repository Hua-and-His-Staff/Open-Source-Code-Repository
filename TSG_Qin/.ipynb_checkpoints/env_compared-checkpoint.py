import numpy as np
from gym.spaces.box import Box

dt=1./12
M=5
N=10

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

class MG_compared():
    def __init__(self,assign_credit):
        self.assign_credit = assign_credit
        self.pure_gen = np.load('./Data/pure_gen.npy')
        self.temp = np.load('./Data/temp.npy')
        self.ACs_parameter = np.load('./Data/ACs_parameter.npy')
        #a,b,mu,Tmax,Tmin,lambda,Pmax
        self.EVs_parameter = np.load('./Data/EVs_parameter.npy')
        #lambda1, lambda2, Pmax
        self.total_steps = int(24/dt)
        self.num_days = int(len(self.pure_gen)/self.total_steps)-10
        self.T_g=1./3
        self.max_g=20
        self.max_PBES = 20
        self.Q=50
        self.eta_charge=0.95
        self.eta_discharge=0.95       
        self.done=False
        #obs:   t, Ppure, PG, SOC, Tout, EVdemand, ACsign  
        #index:[0], [1], [2], [3], [4], [5:5+M],  [-N:]
        #actions: u_G, (P_EV)*M, (T^AC_ref)*N 
        self.observation_space = Box(low=np.float32(np.zeros(5+M+N)),high=np.float32(np.ones(5+M+N)))
        self.action_space = Box(low=np.float32(np.zeros(1+M+N)), high=np.float32(np.ones(1+M+N)))
        self.obs_scalar=lambda x:(x-np.concatenate((np.array([0,0,0,0,20]), np.zeros(M+N)),axis=0))/np.concatenate((np.array([24,20,self.max_g,1,10]),10*np.ones(M), np.ones(N)),axis=0)
        
        
    def seed(self,seed=None):
        np.random.seed(seed)
    
    def move(self,actions):
        # get current time
        if self.done == True:
            return self._get_obs(), 0., True, None
        fines = 0.2*(ReLU(actions-1)+ReLU(-actions))
        actions = np.clip(actions,0,a_max=1)
        u_g = actions[0]
        step = self.index % self.total_steps
        soc = self.soc 
        costs = np.zeros(1+M+N)
        #EV
        #calculate EV costs
        EVs_power = actions[1:1+M]*self.EVs_parameter[2]
        costs[1:1+M] += (self.EVs_end==step) * (self.EVs_parameter[0]*self.EVs_demand + self.EVs_parameter[1]* (self.EVs_demand)**2)
        #仅在需求大于0时存在充电功率
        EVs_power = (self.EVs_demand>0)*EVs_power

        #AC
        ACs_ref = actions[-N:]*20+20
        self.ACs_sign = (self.ACs_start<=step) & (self.ACs_end>=step)
        costs[-N:] += self.ACs_sign*( self.ACs_parameter[5]* ((self.ACs_T>self.ACs_parameter[3])* np.exp(self.ACs_parameter[2]*np.clip(self.ACs_T-self.ACs_parameter[3],0,a_max=5)) +
         (self.ACs_T<self.ACs_parameter[4])*np.exp(self.ACs_parameter[2]*np.clip(self.ACs_parameter[4]-self.ACs_T,0,a_max=5))))*dt
                
        ACs_power = (self.ACs_T>ACs_ref)*self.ACs_parameter[6]
        
        costs[0] += (0.5*self.DG_power + 0.0125*self.DG_power**2)*dt 
        BES_power = self.pure_gen[self.index] + self.DG_power - np.sum(EVs_power) - np.sum(ACs_power) 
        if BES_power>=0:
            new_soc = soc + (BES_power>0)*self.eta_charge*BES_power*dt/self.Q
            new_soc = np.clip(new_soc,0,a_max=1)
            BES_cost = (ESloss(new_soc)-ESloss(soc))*self.Q 
            #potential = (self.EVs_parameter[2]-EVs_power).sum() + (self.ACs_parameter[6]-ACs_power).sum()
            #costs[1:1+M] += BES_cost*(self.EVs_parameter[2]-EVs_power)/potential
            #costs[-N:] += BES_cost*(self.ACs_parameter[6]-ACs_power)/potential
        else:
            new_soc = soc + 1/self.eta_discharge*BES_power*dt/self.Q
            new_soc = np.clip(new_soc,0,a_max=1)
            BES_cost = (ESloss(soc)-ESloss(new_soc))*self.Q
            BES_cost -= (new_soc==0)*(soc*self.Q*self.eta_discharge+BES_power*dt)*5
            #potential = (EVs_power).sum() + (ACs_power).sum()
            #costs[1:1+M] += BES_cost*EVs_power/potential
            #costs[-N:] += BES_cost*ACs_power/potential
        
        #credit assignment
        #costs[0] += BES_cost
        
        
        self.index += 1
        step+=1
        self.episode_reward += -costs.sum()
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
        
        return self._get_obs(), -costs-fines-BES_cost if self.assign_credit==True else (-costs-fines).sum()-BES_cost, self.done, self.episode_reward if self.done==True else None
    
    def _get_obs(self):
        t = dt*(self.index%self.total_steps)
        obs = np.array([t,self.pure_gen[self.index],self.DG_power,self.soc,self.temp[self.index]])                 
        return np.concatenate((obs, self.ACs_sign, self.EVs_demand), axis=0)
            
    
 
  
    def reset(self):
        self.index = self.total_steps*np.random.randint(self.num_days) #+ np.random.randint(3*12)
        self.DG_power = 5
        self.soc = 0.1+0.8*np.random.rand()
        self.EVs_start = self.EVs_parameter[3] + np.random.randint(12,size=M)
        self.EVs_end = self.EVs_start+np.random.randint(low=3*12,high=4*12,size=M)
        self.EVs_required = (self.EVs_end-self.EVs_start)*dt*self.EVs_parameter[2]*(0.3+0.5*np.random.rand(M))
        self.EVs_demand = np.zeros(M)
        self.ACs_start = self.ACs_parameter[7]+np.random.randint(12,size=N)
        self.ACs_end = self.ACs_start+np.random.randint(low=2*12,high=3*12,size=N)
        self.ACs_sign = np.zeros(N)
        self.ACs_T = self.temp[self.index]*np.ones(N)
        self.done=False   
        self.episode_reward = 0.
        return self._get_obs()
    
    
class MG_compared_for_test():
    def __init__(self,assign_credit):
        self.assign_credit = assign_credit
        self.pure_gen = np.load('./Data/pure_gen.npy')
        self.temp = np.load('./Data/temp.npy')
        self.ACs_parameter = np.load('./Data/ACs_parameter.npy')
        #a,b,mu,Tmax,Tmin,lambda,Pmax
        self.EVs_parameter = np.load('./Data/EVs_parameter.npy')
        #lambda1, lambda2, Pmax
        self.total_steps = int(24/dt)
        self.num_days = int(len(self.pure_gen)/self.total_steps)-10
        self.T_g=1./3
        self.max_g=20
        self.max_PBES = 20
        self.Q=50
        self.eta_charge=0.95
        self.eta_discharge=0.95       
        self.done=False
        self.index = self.total_steps*(self.num_days)
        self.DG_power = 5
        self.soc = 0.1+0.8*np.random.rand()
        #obs:   t, Ppure, PG, SOC, Tout, EVdemand, ACsign  
        #index:[0], [1], [2], [3], [4], [5:5+M],  [-N:]
        #actions: u_G, (P_EV)*M, (T^AC_ref)*N 
        self.observation_space = Box(low=np.float32(np.zeros(5+M+N)),high=np.float32(np.ones(5+M+N)))
        self.action_space = Box(low=np.float32(np.zeros(1+M+N)), high=np.float32(np.ones(1+M+N)))
        self.obs_scalar=lambda x:(x-np.concatenate((np.array([0,0,0,0,20]), np.zeros(M+N)),axis=0))/np.concatenate((np.array([24,20,self.max_g,1,10]),10*np.ones(M), np.ones(N)),axis=0)
        
        
    def seed(self,seed=None):
        np.random.seed(seed)
    
    def move(self,actions):
        # get current time
        if self.done == True:
            return self._get_obs(), 0., True, None
        fines = 0.2*(ReLU(actions-1)+ReLU(-actions))
        actions = np.clip(actions,0,a_max=1)
        u_g = actions[0]
        step = self.index % self.total_steps
        soc = self.soc 
        costs = np.zeros(1+M+N)
        #EV
        #calculate EV costs
        EVs_power = actions[1:1+M]*self.EVs_parameter[2]
        costs[1:1+M] += (self.EVs_end==step) * (self.EVs_parameter[0]*self.EVs_demand + self.EVs_parameter[1]* (self.EVs_demand)**2)
        #仅在需求大于0时存在充电功率
        EVs_power = (self.EVs_demand>0)*EVs_power

        #AC
        ACs_ref = actions[-N:]*20+20
        self.ACs_sign = (self.ACs_start<=step) & (self.ACs_end>=step)
        costs[-N:] += self.ACs_sign*( self.ACs_parameter[5]* ((self.ACs_T>self.ACs_parameter[3])* np.exp(self.ACs_parameter[2]*np.clip(self.ACs_T-self.ACs_parameter[3],0,a_max=5)) +
         (self.ACs_T<self.ACs_parameter[4])*np.exp(self.ACs_parameter[2]*np.clip(self.ACs_parameter[4]-self.ACs_T,0,a_max=5))))*dt
                
        ACs_power = (self.ACs_T>ACs_ref)*self.ACs_parameter[6]
        
        costs[0] += (0.5*self.DG_power + 0.0125*self.DG_power**2)*dt 
        BES_power = self.pure_gen[self.index] + self.DG_power - np.sum(EVs_power) - np.sum(ACs_power) 
        if BES_power>=0:
            new_soc = soc + (BES_power>0)*self.eta_charge*BES_power*dt/self.Q
            new_soc = np.clip(new_soc,0,a_max=1)
            BES_cost = (ESloss(new_soc)-ESloss(soc))*self.Q 
            #potential = (self.EVs_parameter[2]-EVs_power).sum() + (self.ACs_parameter[6]-ACs_power).sum()
            #costs[1:1+M] += BES_cost*(self.EVs_parameter[2]-EVs_power)/potential
            #costs[-N:] += BES_cost*(self.ACs_parameter[6]-ACs_power)/potential
        else:
            new_soc = soc + 1/self.eta_discharge*BES_power*dt/self.Q
            new_soc = np.clip(new_soc,0,a_max=1)
            BES_cost = (ESloss(soc)-ESloss(new_soc))*self.Q
            BES_cost -= (new_soc==0)*(soc*self.Q*self.eta_discharge+BES_power*dt)*5
            #potential = (EVs_power).sum() + (ACs_power).sum()
            #costs[1:1+M] += BES_cost*EVs_power/potential
            #costs[-N:] += BES_cost*ACs_power/potential
        
        #credit assignment
        costs[0] += BES_cost
        self.AC_energy += ACs_power.sum()/12
        self.AC_loss += costs[-N:].sum()        
        
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
        
        return self._get_obs(), costs, self.done,  {'consumption':self.AC_energy,'loss':self.AC_loss} if self.done==True else None
    
    def _get_obs(self):
        t = dt*(self.index%self.total_steps)
        obs = np.array([t,self.pure_gen[self.index],self.DG_power,self.soc,self.temp[self.index]])                 
        return np.concatenate((obs, self.ACs_sign, self.EVs_demand), axis=0)
            
    
 
  
    def reset(self):
        self.DG_power = 5
        self.soc = 0.1+0.8*np.random.rand()
        self.EVs_start = self.EVs_parameter[3] + np.random.randint(12,size=M)
        self.EVs_end = self.EVs_start+np.random.randint(low=3*12,high=4*12,size=M)
        self.EVs_required = (self.EVs_end-self.EVs_start)*dt*self.EVs_parameter[2]*(0.3+0.5*np.random.rand(M))
        self.EVs_demand = np.zeros(M)
        self.ACs_start = self.ACs_parameter[7]+np.random.randint(12,size=N)
        self.ACs_end = self.ACs_start+np.random.randint(low=2*12,high=3*12,size=N)
        self.ACs_sign = np.zeros(N)
        self.ACs_T = self.temp[self.index]*np.ones(N)
        self.done=False   
        self.AC_energy = 0.
        self.AC_loss = 0.
        return self._get_obs()
    
    

        