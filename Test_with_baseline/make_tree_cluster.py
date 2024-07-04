import numpy as np
from environment.env import Env


#Create a tree structure to represent the cluster


class Node:
    def __init__(self) -> None:
        self.id = None
        self.children = []
        self.parent = None

    @property
    def usage(self):
        return np.mean([child.usage for child in self.children])
    
class Gpu(Node):
    def __init__(self,id,parent) -> None:
        super().__init__()
        self.children = []
        self.parent = parent
        self.id = id
        self._usage = 0

    def global_id(self,env:Env):
        rack_id = self.parent.parent.id
        machine_id = self.parent.id
        gpu_id = self.id
        return rack_id*env.num_machines_per_rack*env.num_gpus_per_machine+machine_id*env.num_gpus_per_machine+gpu_id

    def set_usage(self,usage):
        self._usage = usage

    @property
    def usage(self):
        return self._usage

    @property
    def is_available(self):
        return self.usage == 0

class Machine(Node):
    def __init__(self,children,parent,id) -> None:
        super().__init__()
        self.children:list[Gpu] = children
        self.parent = parent
        self.id = id
    
    def global_id(self,env:Env):
        rack_id = self.parent.id
        machine_id = self.id
        return rack_id*env.num_machines_per_rack+machine_id

    @property
    def available_gpu_num(self):
        return len(self.get_available_gpu())

    def get_available_gpu(self):
        return [gpu for gpu in self.children if gpu.is_available]
    
    @property
    def is_available(self):
        return self.available_gpu_num > 0
    
class Rack(Node):
    def __init__(self,children,parent,id) -> None:
        super().__init__()
        self.children:list[Machine] = children
        self.parent = parent
        self.id = id

    def global_id(self,env:Env):
        rack_id = self.id
        return rack_id
    
    @property
    def available_gpu_num(self):
        return len(self.get_available_gpu())
    
    def get_available_gpu(self):
        return [gpu for machine in self.children for gpu in machine.get_available_gpu()]

    @property
    def is_available(self):
        return self.available_gpu_num > 0

class Cluster(Node):
    def __init__(self,children,id) -> None:
        super().__init__()
        self.children = children
        self.id = id

    @property
    def available_gpu_num(self):
        return len(self.get_available_gpu())
    
    def get_available_gpu(self):
        available_gpu=[]
        for rack in self.children:
            available_gpu.extend(rack.get_available_gpu())
        return available_gpu
    
    @property
    def is_available(self):
        return self.available_gpu_num > 0

def make_cluster(env:Env):
    cluster_children = []
    for rack_id in range(env.num_racks_per_cluster):
        rack_children = []
        for machine_id in range(env.num_machines_per_rack):
            gpu_children = []
            for gpu_id in range(env.num_gpus_per_machine):
                gpu_children.append(Gpu(gpu_id,None))
            rack_children.append(Machine(gpu_children,None,machine_id))
        cluster_children.append(Rack(rack_children,None,rack_id))

    cluster = Cluster(cluster_children,0)
    # set parent
    for rack in cluster.children:
        rack.parent = cluster
        for machine in rack.children:
            machine.parent = rack
            for gpu in machine.children:
                gpu.parent = machine

    # update usage
    for rack in cluster.children:
        for machine in rack.children:
            for gpu in machine.children:
                gpu.set_usage(0 if env.resources[rack.global_id(env)][machine.id][gpu.id]==-1 else 1)

    return cluster

def Node_grow(node:Node,grow_priority:callable,stop_criteria:callable,reverse=False)->Node:
    '''
    param: node: root node
    param: grow_priority: callable, sort children
    param: stop_criteria: callable, determine whether to stop growing stop_criteria(node)->bool
    param: reverse: bool, if True, grow_priority will be reversed
    '''
    if stop_criteria(node)or type(node) is Machine:
        return node
    else:
        children = sorted(node.children,key=grow_priority,reverse=reverse)
        for child in children:
            if not stop_criteria(child):
                return Node_grow(child,grow_priority,stop_criteria,reverse)
        return node
def find_gpus_on_node(node:Node,grow_priority:callable,gpu_request:int,env:Env,reverse=False)->list[Gpu]:
    if not node.is_available:
        return []
    # if type(node) is Gpu:
    #     return [node]
    if type(node) is Machine:
        return node.get_available_gpu()
    nodes = []
    if node.children:
        children = sorted(node.children,key=grow_priority,reverse=reverse)
        for child in children:
            nodes.extend(find_gpus_on_node(child,grow_priority,gpu_request,env,reverse))
    return nodes

