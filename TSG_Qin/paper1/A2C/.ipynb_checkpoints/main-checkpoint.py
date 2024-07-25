import argparse
import torch
import torch.optim as optim


from Model import Actor,Critic
from VecEnv import SubprocVecEnv,EIEnv
from Train import train



parser = argparse.ArgumentParser(description='A2C (Advantage Actor-Critic)')
parser.add_argument('--no-cuda', action='store_true', help='use to disable available CUDA')
parser.add_argument('--num-workers', type=int, default=8, help='number of parallel workers')
parser.add_argument('--rollout-steps', type=int, default=20, help='steps per rollout')
parser.add_argument('--total-steps', type=int, default=int(3e8), help='total number of steps to train for')
parser.add_argument('--save-interval', type=int, default=int(3e3), help='interval for model saving')
parser.add_argument('--log-dir', type=str, default='runs', help='directory for log files')
parser.add_argument('--model-dir', type=str, default='./models', help='directory for trained model')
parser.add_argument('--actor-lr', type=float, default=2e-4, help='learning rate')
parser.add_argument('--critic-lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.95, help='gamma parameter for GAE')
parser.add_argument('--EV-number', type=int, default=5, help='number of EVs')
parser.add_argument('--AC-number', type=int, default=10, help='number of ACs')
#parser.add_argument('--value-coeff', type=float, default=0.1, help='value loss coeffecient')
parser.add_argument('--entropy-coeff', type=float, default=0.001, help='entropy loss coeffecient')
parser.add_argument('--weight-decay', type=float, default=1e-5, help='L2 regularization coeffecient')
parser.add_argument('--actor-grad-norm-limit', type=float, default=20., help='gradient norm clipping threshold')
parser.add_argument('--critic-grad-norm-limit', type=float, default=20., help='gradient norm clipping threshold')
parser.add_argument('--seed', type=int, default=31, help='random seed')

args = parser.parse_args()


env_fns=[]
for i in range(args.num_workers):
    env_fns.append(lambda: EIEnv(gamma=args.gamma))
venv=SubprocVecEnv(env_fns)
actor_net = Actor(venv.observation_space.shape[0],args.EV_number+1+args.AC_number)
critic_net = Critic(venv.observation_space.shape[0],args.AC_number+args.EV_number+2)
actor_optimizer = optim.Adam(actor_net.parameters(), lr=args.actor_lr,weight_decay=args.weight_decay)
critic_optimizer = optim.Adam(critic_net.parameters(), lr=args.critic_lr,weight_decay=args.weight_decay)
device = torch.device('cuda:0' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

actor_net = actor_net.to(device)
critic_net = critic_net.to(device)
train(args, actor_net, critic_net, actor_optimizer, critic_optimizer, venv, device)





