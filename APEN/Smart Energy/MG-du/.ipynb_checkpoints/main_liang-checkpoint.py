import argparse
import torch
import torch.optim as optim


from Model import GridMLP
from VecEnv import SubprocVecEnv,EIEnv
from Train import train



parser = argparse.ArgumentParser(description='A2C (Advantage Actor-Critic)')
parser.add_argument('--no-cuda', action='store_true', help='use to disable available CUDA')
parser.add_argument('--num-workers', type=int, default=8, help='number of parallel workers')
parser.add_argument('--rollout-steps', type=int, default=17, help='steps per rollout')
parser.add_argument('--total-steps', type=int, default=int(1e7), help='total number of steps to train for')
parser.add_argument('--save-interval', type=int, default=int(1e4), help='interval for model saving based on rollout')
parser.add_argument('--log-dir', type=str, default='runs', help='directory for log files')
parser.add_argument('--model-dir', type=str, default='model', help='directory for trained model')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.93, help='gamma parameter for GAE')
parser.add_argument('--policy-coeff', type=float, default=1.0, help='policy loss coeffecient')
parser.add_argument('--value-coeff', type=float, default=0.05, help='value loss coeffecient')
parser.add_argument('--entropy-coeff', type=float, default=0.01, help='entropy loss coeffecient')
parser.add_argument('--weight-decay', type=float, default=1e-4, help='L2 regularization coeffecient')
parser.add_argument('--grad-norm-limit', type=float, default=20., help='gradient norm clipping threshold')
parser.add_argument('--seed', type=int, default=23, help='random seed')

args = parser.parse_args()


env_fns=[]
for i in range(args.num_workers):
    env_fns.append(lambda: EIEnv())
venv=SubprocVecEnv(env_fns)

net = GridMLP(venv.observation_space.shape[0],venv.action_space.shape[0])
optimizer = optim.Adam(net.parameters(), lr=args.lr,weight_decay=args.weight_decay)

device = torch.device('cuda:0' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

net = net.to(device)

train(args, net, optimizer, venv, device)




