import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument(
        '--use-linear-lr-decay',
        default=False,
        help='whether use linear learning rate decay (default: False)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.95,
        help='discount factor for rewards (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=10,
        help='max norm of gradients (default: 10)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--num-processes',
        type=int,
        default=8,
        help='how many training CPU processes to use (default: 8)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=24,
        help='number of forward steps in A2C (default: 24)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1000,
        help='save interval, one save per n updates (default: 1000)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=None,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=10e7,
        help='number of environment steps to train (default: 10e7)')
    parser.add_argument(
        '--env-name',
        default='microgrid',
        help='environment to train on (default: microgrid)')
    parser.add_argument(
        '--log-dir',
        default='./log/',
        help='directory to save agent logs (default: /log)')
    parser.add_argument(
        '--save-dir',
        default='./models/',
        help='directory to save agent logs (default: ./models/)')
    parser.add_argument(
        '--recurrent-policy',
        default= True,
        help='use a recurrent policy')
    parser.add_argument(
        '--env-settings',
        default={'privacy_preserving':True,'assign_credit':True,'compared':False},
        help='use credit assignment')
    args = parser.parse_args()

    return args
