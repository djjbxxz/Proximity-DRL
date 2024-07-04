from env_components.job import Job
from ..base import GpuSelector
from environment.env import Env
import numpy as np
from ..make_tree_cluster import make_cluster,Node,Node_grow,find_gpus_on_node,Gpu
class LoadBalanceGpuSelector(GpuSelector):
    '''
    `LoadBalanceGpuSelector`: Load Balance Gpu Selector

    Select Machine or Rack with least number of GPUs

    The selection process is bottom-up, from Machine to Rack to Cluster.

    Check if there are enough in a machine, if not, check if there are enough in a rack, if not, check if there are enough in a cluster, if not, return empty action.
    '''

    def __init__(self):
        super().__init__('LoadBalanceGpuSelector')

    def select(self, job: Job, jobqueue: list, resources: list, env: Env) -> tuple[list, list, list]:
        selection = np.zeros(env.num_gpus_per_machine*env.num_machines_per_rack*env.num_racks_per_cluster,dtype=int)
        cluster = make_cluster(env)
        
        #Top down recursive search
        def grow_priority(node:Node):
            return node.available_gpu_num
        def stop_criteria(node:Node):
            return type(node) is Gpu or node.available_gpu_num < job.gpu_request
        node = Node_grow(cluster,grow_priority,stop_criteria,reverse=True)

        if node.available_gpu_num < job.gpu_request:
            return self.empty_action
        selected_gpus = find_gpus_on_node(node,grow_priority,job.gpu_request,env,reverse=True)[:job.gpu_request]
        selected_gpus_index = [gpu.global_id(env) for gpu in selected_gpus]
        selection[selected_gpus_index] = 1
        return self.unflatten_action(selection, env)
    