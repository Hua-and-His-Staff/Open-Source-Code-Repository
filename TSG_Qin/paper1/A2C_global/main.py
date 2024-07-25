import argparse
import torch
import torch.optim as optim
from Model import Network
from VecEnv import SubprocVecEnv,EIEnv
from Train import train, A2C



parser = argparse.ArgumentParser(description='A2C (Advantage Actor-Critic)')
parser.add_argument('--no-cuda', action='store_true', help='use to disable available CUDA')
parser.add_argument('--num-workers', type=int, default=8, help='number of parallel workers')
parser.add_argument('--update-steps', type=int, default=24, help='steps per rollout')
parser.add_argument('--total-episodes', type=int, default=int(1e6), help='total number of episodes to train for')
parser.add_argument('--save-interval', type=int, default=int(800), help='interval for model saving')
parser.add_argument('--test-runs', type=int, default=int(10), help='the number of episodes for test')
parser.add_argument('--log-dir', type=str, default='runs', help='directory for log files')
parser.add_argument('--model-dir', type=str, default='./models', help='directory for trained model')
parser.add_argument('--net-lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.95, help='gamma parameter for GAE')
parser.add_argument('--EV-number', type=int, default=5, help='number of EVs')
parser.add_argument('--AC-number', type=int, default=10, help='number of ACs')
#parser.add_argument('--value-coeff', type=float, default=0.1, help='value loss coeffecient')
parser.add_argument('--entropy-coeff', type=float, default=0.001, help='entropy loss coeffecient')
parser.add_argument('--weight-decay', type=float, default=1e-5, help='L2 regularization coeffecient')
parser.add_argument('--grad-norm-limit', type=float, default=20., help='gradient norm clipping threshold')
parser.add_argument('--seed', type=int, default=31, help='random seed')

args = parser.parse_args()


env_fns=[]
for i in range(args.num_workers):
    env_fns.append(lambda: EIEnv())
venv=SubprocVecEnv(env_fns)
net = Network(GRU_input_size=13, GRU_hidden_size=64, MLP_input_size=8, num_actions=16, num_values=17)
optimizer = optim.Adam(net.parameters(), lr=args.net_lr,weight_decay=args.weight_decay)
device = torch.device('cuda:0' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
net = net.to(device)
policy = A2C(args, net, optimizer, device)
train(args, policy, venv, device)





