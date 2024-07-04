import numpy as np
from abc import ABCMeta, abstractmethod
from environment.env import Env
from queue import Queue


class Metrics(metaclass=ABCMeta):
    def __init__(self, name: str = None):
        self.name = name

    @abstractmethod
    def __call__(self, env: Env):
        '''
        This method will be called at each call of env.step()
        '''
        pass

    @abstractmethod
    def result(self) -> float:
        '''
        This method will return the result of the metrics
        '''
        pass


class JCT(Metrics):
    def __init__(self):
        super().__init__('JCT')
        self.jct = Queue()

    def __call__(self, env: Env):
        if env.done:
            self.jct.put(np.mean([[job.finish_time for job in env.job_seq]]))

    def result(self):
        return np.mean(self.jct.queue)


class Makespan(Metrics):
    def __init__(self):
        super().__init__('Makespan')
        self.makespan = Queue()

    def __call__(self, env: Env):
        if env.done:
            self.makespan.put(env.curr_time)

    def result(self):
        return np.mean(self.makespan.queue)


class Throughput(Metrics):
    def __init__(self):
        super().__init__('Throughput')
        self.throughput = Queue()

    def __call__(self, env: Env):
        if env.done:
            self.throughput.put(np.mean(env.avg_running_jobs))

    def result(self):
        return np.mean(self.throughput.queue)


class Reward(Metrics):
    communication_cost_ranges = [[0.775, 0.775], [0.8, 1.55], [0.825, 1.825], [
        0.85, 2.1], [1.125, 2.375], [1.15, 2.65], [1.175, 2.925], [1.2, 3.2]]

    def __init__(self):
        super().__init__('Reward')
        self.reward = Queue()

    def __call__(self, env: Env):
        if env.done:
            self.reward.put(np.sum([self.reward_func_communicationCost(
                job.gpu_request, job.communication_cost) for job in env.job_seq]))

    def result(self):
        return np.mean(self.reward.queue)

    def reward_func_communicationCost(self, num_gpus: int, communication_cost: float):
        if num_gpus == 1:
            return 0  # avoid divide by zero
        min, max = self.communication_cost_ranges[num_gpus-1]
        return 1-((communication_cost-min)/(max-min))

class CommunicationCost(Metrics):
    '''
    This metric will calculate the communication cost
    '''
    def __init__(self):
        super().__init__('CommunicationCost')
        self.communication_cost = Queue()

    def __call__(self, env: Env):
        if env.done:
            self.communication_cost.put(np.sum([job.communication_cost for job in env.job_seq]))

    def result(self):
        return np.mean(self.communication_cost.queue)

class CommunicationCostByJobSize(Metrics):
    '''
    This metric will calculate the communication cost by job size
    '''
    communication_cost_ranges = [[0.775,0.775],[0.8,1.55],[0.825,1.825],[0.85,2.1],[1.125,2.375],[1.15,2.65],[1.175,2.925],[1.2,3.2]]
    def __init__(self):
        super().__init__('CommunicationCostByJobSize')
        self.communication_cost_by_job_size = []

    def __call__(self, env: Env):
        if env.done:
            if self.communication_cost_by_job_size == []:
                self.communication_cost_by_job_size=[[] for _ in range(env.max_gpu_request)]
            for job in env.job_seq:
                self.communication_cost_by_job_size[job.gpu_request-1].append(self.reward_func_communicationCost(job.gpu_request, job.communication_cost))

    def result(self):
        normalized_communication_cost_by_job_size = [np.array(job_size).mean() for job_size in self.communication_cost_by_job_size]
        return normalized_communication_cost_by_job_size
    
    def reward_func_communicationCost(self, num_gpus:int, communication_cost:float):
        if num_gpus == 1:
            return 0 #avoid divide by zero
        min,max = self.communication_cost_ranges[num_gpus-1]
        return 1-((communication_cost-min)/(max-min))