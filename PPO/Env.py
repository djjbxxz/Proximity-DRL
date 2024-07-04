from collections import namedtuple
import sys
import tensorflow as tf
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.trajectories import time_step as ts
import numpy as np
from tf_agents.specs import array_spec, tensor_spec
from tf_agents.specs import BoundedTensorSpec,TensorSpec
from tf_agents.typing import types
sys.path.append('.')
from environment.gpu_select_env import Action_type, GpuSelectEnv
from tf_agents.environments.tf_py_environment import TFPyEnvironment

observation_struct = namedtuple('observation_struct',['observation','action_mask'])

class py_GpuSelectEnv(PyEnvironment, GpuSelectEnv):
    def __init__(self, allow_idle = False, test_mode=False,config_filepath:str=None):
        PyEnvironment.__init__(self)
        GpuSelectEnv.__init__(self, allow_idle = allow_idle, config_filepath = config_filepath, test_mode = test_mode)
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=int, minimum=0, maximum=self.num_actions-1, name='action')
        self._observation_spec = observation_struct(
            array_spec.ArraySpec(shape=(self.input_channel,), dtype=float, name='observation'), 
            array_spec.BoundedArraySpec(shape=(self.num_actions,), dtype=int, minimum=0, maximum=1, name='action_mask'))


    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _step(self, action):
        action = int(action)
        if self.done:
            return self.reset()
        if self.test:
            ob, reward, done, action_type = GpuSelectEnv.step(self, action)
            self.action_type_count[action_type.value]+=1
        else:
            ob, reward, done = GpuSelectEnv.step(self, action)
        action_mask = self.get_valid_mask()
        observation = observation_struct(observation=ob, action_mask=action_mask)

        #data check
        # if observation.observation[-1] ==0:
        #     r=[0]*32
        #     r.append(1)
        #     assert all(observation.action_mask== r)
        # else:
        #     for gpu in range(32):#check every gpu, if it is available, the action mask should be 1, otherwise, it should be 0
        #         if observation.action_mask[gpu]==1:
        #             assert observation.observation[gpu] == 1
        #         else:
        #             assert observation.observation[gpu] in [0,-1]
        #         if observation.observation[gpu]==-1:
        #             assert observation.action_mask[gpu]==0


        if done:
            return ts.termination(observation=observation, reward=reward)
        else:
            return ts.transition(observation=observation, reward=reward, discount=0.99)
    
    def _reset(self):
        if self.test:
            self.action_type_count = np.array([0 for _ in range(4)],dtype=int)
        ob = GpuSelectEnv.reset(self)
        action_mask = self.get_valid_mask()
        observation = observation_struct(observation=ob, action_mask=action_mask)
        return ts.restart(observation=observation)
    

# def wrapperFor_pyEnv(env_class:callable):
#     class env(PyEnvironment, env_class):
#         def __init__(self, pa=None, test_mode=False):
#             PyEnvironment.__init__(self)
#             env_class.__init__(self, pa, test_mode)
#             self._action_spec = array_spec.BoundedArraySpec(
#                 shape=(), dtype=int, minimum=0, maximum=self.num_actions-1, name='action')
#             self._observation_spec = observation_struct(
#                 array_spec.ArraySpec(shape=(self.input_channel,), dtype=float, name='observation'), 
#                 array_spec.BoundedArraySpec(shape=(self.num_actions,), dtype=int, minimum=0, maximum=1, name='action_mask'))


#         def action_spec(self):
#             return self._action_spec

#         def observation_spec(self):
#             return self._observation_spec

#         def _step(self, action):
#             action = int(action)
#             if self.done:
#                 return self.reset()
#             if self.test:
#                 ob, reward, done, action_type = env_class.step(self, action)
#                 self.action_type_count[action_type.value]+=1
#             else:
#                 ob, reward, done = env_class.step(self, action)
#             action_mask = self.get_valid_mask()
#             observation = observation_struct(observation=ob, action_mask=action_mask)

#             #data check
#             # if observation.observation[-1] ==0:
#             #     r=[0]*32
#             #     r.append(1)
#             #     assert all(observation.action_mask== r)
#             # else:
#             #     for gpu in range(32):#check every gpu, if it is available, the action mask should be 1, otherwise, it should be 0
#             #         if observation.action_mask[gpu]==1:
#             #             assert observation.observation[gpu] == 1
#             #         else:
#             #             assert observation.observation[gpu] in [0,-1]
#             #         if observation.observation[gpu]==-1:
#             #             assert observation.action_mask[gpu]==0


#             if done:
#                 return ts.termination(observation=observation, reward=reward)
#             else:
#                 return ts.transition(observation=observation, reward=reward, discount=0.99)
        
#         def _reset(self):
#             if self.test:
#                 self.action_type_count = np.array([0 for _ in range(4)],dtype=int)
#             ob = env_class.reset(self)
#             action_mask = self.get_valid_mask()
#             observation = observation_struct(observation=ob, action_mask=action_mask)
#             return ts.restart(observation=observation)
#     return env

if __name__=="__main__":
    py_GpuSelectEnv(pa=None, test_mode=False)
    pass