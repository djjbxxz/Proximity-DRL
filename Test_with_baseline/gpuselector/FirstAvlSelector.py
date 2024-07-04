from env_components.job import Job
from ..base import GpuSelector
from environment.env import Env
import numpy as np

class FirstAvlGpuSelector(GpuSelector):
    def __init__(self):
        super().__init__('FirstAvaibleGpu')

    def select(self, job: Job,jobqueue: list, resources: list,env:Env) -> tuple[list,list,list]:
        avl_gpus = env.get_avl_gpus()
        num_avl_gpus = avl_gpus[0].size
        if num_avl_gpus <job.gpu_request: return self.empty_action

        num_gpu_requested = job.gpu_request

        
        return (avl_gpus[0][:num_gpu_requested],avl_gpus[1][:num_gpu_requested],avl_gpus[2][:num_gpu_requested])
