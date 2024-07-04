from env_components.job import Job
from ..base import GpuSelector
from environment.env import Env
import numpy as np
class PPOGpuSelector(GpuSelector):
    def __init__(self,weight_path=None):
        import sys
        sys.path.append('PPO')
        from PPO.agent import PPO_Agent
        from PPO.Env import py_GpuSelectEnv
        from tf_agents.environments.tf_py_environment import TFPyEnvironment
        import os
        import yaml
        #delay import
        super().__init__('PPO RL')
        self.weight_path = weight_path if weight_path is not None else 'runs/ppo/2023-10-31 19:56:05_continueWitLlr_5e-6_noEntropyReg'
        self.name+=f' weight path: {self.weight_path}'
        train_env = TFPyEnvironment(py_GpuSelectEnv())
        test_env = py_GpuSelectEnv(test_mode=True)
        with open(os.path.join('PPO', 'config.yaml')) as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        config.pop('env_batch_size')
        self.agent = PPO_Agent(train_env,test_env,**config, eval_mode=True)
        self.agent.load(self.weight_path)
        train_env.close()
        self.test_env = test_env

    def select(self, job: Job,jobqueue: list, resources: list,env:Env) -> tuple[list,list,list]:
        observation = self.observe(env, job)
        gpus = self.agent.infer(env, job, observation,allow_idle=False)
        if env.resources.size in gpus:# idle action
            return ([],[],[])
        actions = self.unflatten_action(env,gpus)
        
        return actions

    def observe(self, env:Env, job:Job) -> np.ndarray:
        '''
        Observation include:\n
        `1. Gpu availability`\n
        `2. Job length`\n
        `3. Gpu num requested`\n
        '''
        job_len, gpu_request = (0, 0) if job is None else (
            job.job_len, job.gpu_request)

        count_down = np.zeros(shape=env.resources.shape)
        for job in env.running_queue:
            count_down[job.gpus] = job.ts_togo

        # Normalization
        count_down = np.clip(count_down/50, 0, 1)
        job_len = job_len/env.max_job_len
        gpu_request = gpu_request / env.max_gpu_request

        # Assemble
        ob = np.zeros((count_down.size*2+2))
        ob[-count_down.size:] = count_down.flatten()
        ob[count_down.size] = job_len
        ob[count_down.size+1] = gpu_request
        return ob


    def _get_Gpu_available(self,env: Env) -> np.ndarray:
        '''
        Return a flatten array indicates the availability of gpus.
        1 means available, 0 means unavailable
        '''
        resourses = env.resources.flatten()
        availablity = np.zeros_like(resourses,dtype=int)
        availablity[resourses == -1] = 1
        return availablity
    
    def unflatten_action(self,env:Env, action: np.ndarray):
        '''
        Convert the action from 1d to 2d
        '''
        action = np.array(action,dtype=int)
        f_action = np.zeros(shape=env.resources.size,dtype=int)
        f_action[action] = 1
        f_action = f_action.reshape(env.resources.shape)
        return np.where(f_action == 1)
