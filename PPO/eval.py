import argparse
import os
import yaml
import sys

from tf_agents.environments import TFPyEnvironment


sys.path.append('.')
from DQN.utils import reduce_gpu_memory_usage
from DQN.agent import DQN_Agent
from DQN.Env import py_GpuSelectEnv
import tensorflow as tf

reduce_gpu_memory_usage()
num_iterations = 20000 # @param {type:"integer"}

initial_collect_steps = 100  # @param {type:"integer"}




def main(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    train_env = py_GpuSelectEnv()
    test_env = py_GpuSelectEnv(test_mode=True)
    agent = DQN_Agent(train_env, test_env,eval_mode=True, **config)

    if args.model:
        agent.load(args.model)
    agent.initialize()
    agent.eval()



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="DQN Tensorflow")
    parser.add_argument('--model', type=str, default="", help='Pretrained model')
    parser.add_argument('--config', type=str, default=os.path.join('DQN', 'config.yaml'))
    parser.add_argument('--num_iterations', type=int, default=1e6, help='Number of training steps')

    main(parser.parse_args())