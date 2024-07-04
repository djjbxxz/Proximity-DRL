import sys
sys.path.append('.')
import argparse
import os
import yaml
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.environments.batched_py_environment import BatchedPyEnvironment

from agent import PPO_Agent

from Env import py_GpuSelectEnv
from utils import isDebug, reduce_gpu_memory_usage
reduce_gpu_memory_usage()
from PPO.utils import HyperParamsWatcher

def main(args):
    with open(args.agent_config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    train_env = TFPyEnvironment(BatchedPyEnvironment([py_GpuSelectEnv(allow_idle=False,config_filepath=args.env_config)for _ in range(config.pop('env_batch_size'))]))
    test_env = py_GpuSelectEnv(allow_idle=False,config_filepath=args.env_config,test_mode=True)
    agent = PPO_Agent(train_env, test_env, **config, 
                      debug_summaries=args.debug_summaries, summarize_grads_and_vars=args.summarize_grads_and_vars, name=args.name)
    if not isDebug():
        agent.logger.save_config([args.agent_config, args.env_config])
        params_mapping = {each: '_' + each for each in ['value_pred_loss_coef',
                                                        'entropy_regularization',
                                                        'policy_l2_reg',
                                                        'value_function_l2_reg',
                                                        'shared_vars_l2_reg',
                                                        'learning_rate',]}
        watcher = HyperParamsWatcher(agent.logger.target_file_list[0], params_mapping, agent)
        watcher.start()

    if args.model:
        agent.load(args.model)
    agent.initialize()
    if not isDebug():
        agent.useGraphOptimization()
    agent.run(int(args.num_iterations))



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="PPO Tensorflow")
    parser.add_argument('--model', type=str, default="", help='Pretrained model')
    parser.add_argument('--agent_config', type=str, default=os.path.join('PPO', 'config.yaml'))
    parser.add_argument('--env_config', type=str, default=os.path.join('environment', 'config.yaml'))
    parser.add_argument('--num_iterations', type=int, default=1e8, help='Number of training steps')
    parser.add_argument('--debug_summaries',  action='store_true',help='enable debug info')
    parser.add_argument('--summarize_grads_and_vars', action='store_true',help='grads_and_vars')
    parser.add_argument('--name', type=str, default="", help='name of the run')

    main(parser.parse_args())