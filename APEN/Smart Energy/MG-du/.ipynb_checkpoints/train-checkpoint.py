import multiprocessing
from threading import Event
import threading
import tensorflow as tf
import numpy as np
import os
import signal
import sys
import shutil
import time
from env.grid import Grid
from grid_description import n_bus,linkage,power_locs
from NN import ACNet,GLOBAL_NET_SCOPE

# 状态空间维数，bus+bes+line+ext_grid
N_S = n_bus+power_locs['bes'].size+len(linkage.keys())+1
# 动作空间维数，MT+FC+DEG+BES
N_A = power_locs['mt'].size+power_locs['fc'].size+power_locs['deg'].size+power_locs['bes'].size

zero_action={}
zero_action['mt']=np.zeros(power_locs['mt'].size)
zero_action['deg']=np.zeros(power_locs['deg'].size)
zero_action['fc']=np.zeros(power_locs['fc'].size)
zero_action['bes']=np.zeros(power_locs['bes'].size)

start=8*60
end=20*60



save_interval=60*15


OUTPUT_GRAPH = True
LOG_DIR = './log'
N_WORKERS = multiprocessing.cpu_count()
MAX_EP_STEP = end-start
MAX_GLOBAL_EP = 100000
GAMMA=0.8
UPDATE_GLOBAL_ITER = 17
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0
T_till_fail=0

class Worker(object):
    def __init__(self, name, SESS,COORD, globalAC):
        # 创建仿真环境
        self.env = Grid(n_bus,linkage,power_locs,data_source='./env/power_data.npz',start=start,end=end,step_len=1.0/60)
        self.name = name
        self.AC = ACNet(name, SESS, N_S,N_A, global_net=globalAC)
        self.COORD=COORD

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        global T_till_fail
        buffer_s, buffer_a, buffer_r = [], [], []
        while not self.COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            flag,r_time,r_real,info = self.env.simulate(zero_action)

            s=np.concatenate([info['power'],info['bes'],list(info['line_load'].values()),info['ext']])
            ep_r = 0
            rnn_state = SESS.run(self.AC.init_state)  # zero rnn state at beginning
            keep_state = rnn_state.copy()  # keep rnn state for updating global net
            for ep_t in range(1,MAX_EP_STEP):
                if  self.COORD.should_stop():
                    break
                a, rnn_state_ = self.AC.choose_action(s, rnn_state)  # get the action and next rnn state
                # 将action包装为环境接受的形式
                action={}
                action['mt']=a[:power_locs['mt'].size]
                action['deg']=a[power_locs['mt'].size:power_locs['mt'].size+power_locs['deg'].size]
                action['fc']=a[power_locs['mt'].size+power_locs['deg'].size:power_locs['mt'].size+power_locs['deg'].size+power_locs['fc'].size]
                action['bes']=a[power_locs['mt'].size+power_locs['deg'].size+power_locs['fc'].size:]
                # 获得环境反馈
                flag,r_time,r_optimal,info = self.env.simulate(action)
                s_=np.concatenate([info['power'],info['bes'],list(info['line_load'].values()),info['ext']])
                r=-r_time-r_optimal
                done=False
                if flag==0:
                    # 运行结束
                    done=True
                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)  # normalize

                if ep_t % UPDATE_GLOBAL_ITER == 0 or flag==-1 or done:  # update global and assign to local net
                    if done:
                        v_s_ = 0  # terminal
                    else:
                        v_s_ = SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :], self.AC.init_state: rnn_state_})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:  # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(
                        buffer_v_target)

                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                        self.AC.init_state: keep_state,
                    }

                    self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()
                    keep_state = rnn_state_.copy()  # replace the keep_state as the new initial rnn state_
                    # print('update parameter,worker %s, worker step %d, current reward %f'%(self.name,ep_t,r))
                s = s_
                rnn_state = rnn_state_  # renew rnn state
                if flag==-1:
                    T_till_fail=0.5*T_till_fail+ep_t+1
                    break
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.7 * GLOBAL_RUNNING_R[-1] + 0.3 * ep_r)
                    print('one ep finished,',self.name,"Ep:",GLOBAL_EP,"| Ep_r: %f" % GLOBAL_RUNNING_R[-1])
                    GLOBAL_EP += 1
                    break
        self.COORD.request_stop()
    
            

def save_network(exit,sess,parameters,save_path,save_interval):
    while not exit.is_set():
        exit.wait(save_interval)
        if not exit.is_set():
            saver = tf.train.Saver(var_list=parameters)
            save_path = saver.save(sess, 'rnn_model/global_net')
            print('mean step till fail:%d, model parameters are saved at %s'%(T_till_fail,time.ctime()))
        

def sigint_handler(exit,COORD):
    print('\nCtrl+C pressed, stop training...\n')
    COORD.request_stop()
    print('stop model saving process...')
    exit.set()
        
if __name__ == "__main__":
    SESS = tf.Session()
    GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE, SESS, N_S,N_A)  # we only need its params
    COORD = tf.train.Coordinator()
    
    workers = []
    # Create worker
    for i in range(N_WORKERS):
        i_name = 'NET_%i' % i  # worker name
        workers.append(Worker(i_name, SESS, COORD, GLOBAL_AC))

    
    SESS.run(tf.global_variables_initializer())

    ACNet_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=GLOBAL_NET_SCOPE)
    save_path = tf.train.latest_checkpoint('rnn_model/')
    if save_path:
        saver = tf.train.Saver(ACNet_vars)
        saver.restore(SESS, save_path)

    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, SESS.graph)

    worker_threads = []
    
    exit = Event()
    # 创建保存线程
    save_thread=threading.Thread(target=lambda :save_network(exit,SESS,ACNet_vars,save_path,save_interval))
    save_thread.start()
    for worker in workers:
        if save_path:
            worker.AC.pull_global()
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    print('binding interrupt handler to SIGINT...')
    signal.signal(signal.SIGINT, lambda signal, frame: sigint_handler(exit,COORD))  
    COORD.join(worker_threads)
    save_thread.join()
    print('save parameters...')
    saver = tf.train.Saver(var_list=ACNet_vars)
    save_path = saver.save(SESS, 'rnn_model/global_net')
    print("model parameters are saved in path: %s" % save_path)


np.save('GLOBAL_RUNNING_R', GLOBAL_RUNNING_R)
