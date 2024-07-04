from .metrics import Metrics, JCT, Makespan, Throughput, Reward, CommunicationCostByJobSize, CommunicationCost
import numpy as np
from environment.env import Env
from env_components.job import Job
from abc import ABCMeta, abstractmethod
import sys

import yaml
sys.path.append('.')


class DotDict(dict):
    """
    dot.notation access to dictionary attributes
    for example:
    a = {'a':1, 'b':2}
    a = DotDict(a)
    a.a== a['a']
    """

    def __getattr__(self, attr):
        return self.get(attr)


class JobSelector(metaclass=ABCMeta):
    def __init__(self, name: str = None):
        self.name = name

    def select(self, jobqueue: list, resources: list, env: Env) -> int or None:
        """
        Selects a job from the jobqueue

        Parameters
        -------
        jobqueue: list
            a list of jobs
        resources: list
            a list indicating the status of resources

        Returns
        -------
        int: index of the selected job
        """
        raise NotImplementedError


class GpuSelector(metaclass=ABCMeta):
    def __init__(self, name: str = None):
        self.name = name
        self.empty_action = ([], [], [])

    def select(self, job: Job, jobqueue: list, resources: list, env: Env) -> tuple[list[int], list[int], list[int]]:
        """
        Selects gpu(s) which will satisfy requirements of the selected job

        Parameters
        -------
        job: Job
            a selected job
        jobqueue: list
            a list of jobs
        resources: list
            a list indicating the status of resources

        Returns
        -------
        tuple[list[int], list[int], list[int]]
            unflatten action, a tuple of three lists, each list contains the index of racks, machines, gpus respectively
        """
        raise NotImplementedError

    def unflatten_action(self, action: np.ndarray, env: Env):
        '''
        Convert the action from 1d to 2d
        '''
        if type(action) is not np.ndarray:
            action = np.array(action)
        action = action.reshape(env.resources.shape)
        return np.where(action == 1)

class CoopSelector(metaclass=ABCMeta):
    '''
    This selector will select a job and gpus for the job
    '''
    def __init__(self, name: str = None):
        self.jobselector = JobSelector(name)
        self.gpuselector = GpuSelector(name)
        self.name = name
        self.empty_action = (None, ([], [], []))
        self.action = None#(None, ([], [], []) ) # (job_index, gpus)
        self.jobselector.select = self.jobselection
        self.gpuselector.select = self.gpuselection

    @abstractmethod
    def _select(self, env: Env) -> tuple[int, list[int], list[int], list[int]]:
        """
        Selects job and gpu(s)

        Parameters
        -------
        env: Env
            an environment

        Returns
        -------
        tuple[int, list[int], list[int], list[int]]
            index of the selected job, unflatten action, a tuple of three lists, each list contains the index of racks, machines, gpus respectively
        """
        raise NotImplementedError

    def jobselection(self, jobqueue: list, resources: list, env: Env):
        self.action = self._select(env)
        return self.action[0]
    
    def gpuselection(self, job: Job, jobqueue: list, resources: list, env: Env):
        action = self.action[1]
        self.action = None
        return action

class testEnv(Env):
    def __init__(self, seed=None, config_filepath: str = 'environment/config.yaml'):
        with open(config_filepath) as f:
            pa = DotDict(yaml.load(f, Loader=yaml.SafeLoader))
        super().__init__(pa)
        self.done = False
        self.pa = pa
        if seed is not None:
            self.seed(seed)

    def step(self, job_index: int or None, gpus: tuple[list[int], list[int], list[int]]):
        self.curr_time += 1
        if self.done:
            self.reset()
        if job_index is None or len(gpus[0]) == 0:
            pass  # no success, idle
        else:
            self.currrent_job = self.jobqueue[job_index]
            self.currrent_job.start_time = self.curr_time
            self.currrent_job.gpus = gpus
            self.assign_job_gpus_non_preemptive(job_index, gpus)
            if np.random.random() <= self.new_job_rate and self.j_id < self.target_num_job_arrive:
                self.insert_new_job()
        self.avg_running_jobs = np.append(
            self.avg_running_jobs, len(self.running_queue))

        self.advance_runningjobs_onestep()
        self.update_resources()
        done_jobs = self.get_done_jobs()

        self.remove_jobs(done_jobs)
        self.done = all([job.is_done() for job in self.job_seq])

    def observe(self):
        pass
    
    @property
    def available_gpu_num(self)->int:
        return np.sum(self.resources<0)


class TestCase:
    '''
    A test case for testing the performance of a job selector and a gpu selector for given number of episodes.
    '''

    def __init__(self, jobselector: JobSelector, gpuselector: GpuSelector, seed=0, config_filepath='environment/config.yaml'):
        self.jobselector = jobselector
        self.gpuselector = gpuselector
        self.seed = seed
        self.env = testEnv(seed=seed, config_filepath=config_filepath)
        self.selected_job_index: int = None
        self.selected_gpus = None
        self.metrics: list[Metrics] = [
            JCT(),
            Makespan(), 
            Throughput(), 
            # Reward(),
            # CommunicationCostByJobSize(),
            CommunicationCost()
            ]

    def run(self, num_episode=1):
        for i in range(num_episode):
            self.env.reset()
            self.signal_metrics()
            while not self.env.done:
                jobqueue = self.env.jobqueue
                resources = self.env.resources

                self.selected_job_index = self.jobselector.select(
                    jobqueue, resources, self.env) if len(jobqueue) > 0 else None
                if self.selected_job_index is None or jobqueue[self.selected_job_index].gpu_request > self.env.available_gpu_num:
                    self.selected_gpus = self.gpuselector.empty_action
                else:
                    self.selected_gpus = self.gpuselector.select(
                        jobqueue[self.selected_job_index], jobqueue, resources, self.env)

                self.env.step(self.selected_job_index, self.selected_gpus)
                self.signal_metrics()
        result = {metric.name: metric.result() for metric in self.metrics}
        print(f"""
            Test result over {num_episode} epsisode
            Seed: {self.seed}
            on Job Selector: {self.jobselector.name}, GPU Selector: {self.gpuselector.name}
            {', '.join(f"{key}: {round(value,2)}" for key, value in result.items())}
            """)
        return result

    def reset(self):
        self.env.reset()
        self.selected_job_index = None
        self.selected_gpus = None

    def signal_metrics(self):
        return [metric(self.env) for metric in self.metrics]
