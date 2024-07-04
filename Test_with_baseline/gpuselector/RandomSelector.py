from env_components.job import Job
from ..base import GpuSelector
from environment.env import Env
import numpy as np

class RandomWithoutIdleGpuSelector(GpuSelector):
    def __init__(self):
        super().__init__('random without idle')

    def select(self, job: Job,jobqueue: list, resources: list,env:Env) -> tuple[list,list,list]:
        avl_gpus = env.get_avl_gpus()
        num_avl_gpus = avl_gpus[0].size
        if num_avl_gpus <job.gpu_request: return ([],[],[])

        choices = np.random.choice(range(avl_gpus[0].size),job.gpu_request,replace=False)
        selectedgpus = np.array(avl_gpus)[:,choices]
        
        return (selectedgpus[0],selectedgpus[1],selectedgpus[2])

class RandomWithIdleGpuSelector(GpuSelector):
    def __init__(self):
        super().__init__('random with idle')

    def select(self, job: Job,jobqueue: list, resources: list,env:Env) -> tuple[list,list,list]:
        avl_gpus = env.get_avl_gpus()
        num_avl_gpus = avl_gpus[0].size
        if num_avl_gpus <job.gpu_request: return ([],[],[])
        choices = np.random.choice(range(num_avl_gpus+1),job.gpu_request,replace=False)
        if 0 in choices: return ([],[],[]) # idle action
        choices-=1
        selectedgpus = np.array(avl_gpus)[:,choices]
        
        return (selectedgpus[0],selectedgpus[1],selectedgpus[2])
