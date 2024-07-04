import os
from ..base import testEnv, TestCase
from typing import Any
from abc import abstractmethod, ABCMeta
import json
import datetime
import numpy as np
# info need to be grabbed from each call of env.step()
# 1. cluster status
# 2. job queue status
# 3. the action

__all__ = ['dump']


class dumper(metaclass=ABCMeta):
    def __init__(self, name: str = None):
        self.record = []
        self.name = name

    @abstractmethod
    def snap(self):
        pass

    def dump(self) -> dict[str, Any]:
        return {self.name: self.record}


class GpuClusterDumper(dumper):
    '''
    dump the cluster status, 1 for free, 0 for occupied
    '''

    def __init__(self):
        super().__init__('GpuClusterStatus')

    def snap(self, env: testEnv, *args, **kwargs):
        self.record.append(env._get_Gpu_available().tolist())


class JobQueueDumper(dumper):
    '''
    dump the job queue status
    '''

    def __init__(self):
        super().__init__('JobQueueStatus')
        # , 'arrival_time','start_time','finish_time','slowdown','stepcounter','pseudo_step','d_ex','m','d_m','gradsize','d_f','d_done','tt_m','rt_m','v_m','color','ts_togo','ts_done','singlejoblimbw','multijoblimbw','scale','gpus','status','prevstatus','waiting_time','slowdown','communication_cost']
        self.attribute_to_record = ['job_len', 'gpu_request']

    def snap(self, env: testEnv, *args, **kwargs):
        record = []
        for index_in_queue, job in enumerate(env.jobqueue):
            record.append({})
            for attr in self.attribute_to_record:
                record[index_in_queue][attr] = int(getattr(job, attr))
        self.record.append(record)


class JobSelectionDumper(dumper):
    '''
    dump the job selection
    '''

    def __init__(self):
        super().__init__('JobSelection')

    def snap(self, env: testEnv, *args, **kwargs):
        self.record.append(int(args[0])if args[0] is not None else args[0])


class GpuSelectionDumper(dumper):
    '''
    dump the gpu selection
    '''

    def __init__(self):
        super().__init__('GpuSelection')

    def snap(self, env: testEnv, *args, **kwargs):
        gpu_select_flatten = np.zeros(shape=env.resources.shape, dtype=int)
        gpu_select_flatten[args[1]] = 1
        self.record.append(gpu_select_flatten.flatten().tolist())


class GpuUtilizationDumper(dumper):
    '''
    dump the gpu utilization
    '''

    def __init__(self):
        super().__init__('GPU_utilization')

    def snap(self, env: testEnv, *args, **kwargs):
        self.record.append(1-env._get_Gpu_available().mean())


class GpuProgressDumper(dumper):
    '''
    dump the gpu progress
    '''

    def __init__(self):
        super().__init__('GPU_progress')

    def snap(self, env: testEnv, *args, **kwargs):
        progress = np.zeros(shape=env.resources.shape, dtype=float)
        for job in env.running_queue:
            progress[job.gpus] = job.get_progress()
        self.record.append(progress.flatten().tolist())


class CommunicationCostDumper(dumper):
    '''
    dump the communication cost
    '''

    def __init__(self):
        super().__init__('CommunicationCost')
        self.is_job_assigned_in_last_step = False

    def snap(self, env: testEnv, *args, **kwargs):
        # communication cost will be calculated in the next step, so the communication cost in this step is from the last job
        communication_cost = env.running_queue[-1].communication_cost if self.is_job_assigned_in_last_step else 0
        self.is_job_assigned_in_last_step = len(args[1][0])>0 # there is a job assigned in this step
        self.record.append(communication_cost)
        # remove the first element, which is introduced by compromising the delay calulation of the comminication cost
        if env.done:
            self.record.pop(0) 
            self.record.append(0)

class Dump_env():
    def __init__(self, test_case: TestCase, logdir: str = None):
        self.iswatching = False
        self.test_case = test_case
        self.env = test_case.env
        self.watch(self.env)
        self.dumper: list[dumper] = [
            GpuClusterDumper(),
            JobQueueDumper(),
            JobSelectionDumper(),
            GpuSelectionDumper(),
            GpuUtilizationDumper(),
            GpuProgressDumper(),
            CommunicationCostDumper(),
        ]
        self.logdir = logdir
        if not os.path.exists(logdir):
            os.makedirs(logdir)

    def dump(self):
        to_write = {}
        selectors_name = '_'.join(
            [self.test_case.jobselector.name, self.test_case.gpuselector.name]).replace('/', '_')
        filename = str(datetime.datetime.now()) + '_' + selectors_name
        filename = filename.replace(':', '_')
        dump_dir = os.path.join(self.logdir, filename+'.json')

        with open(dump_dir, 'w') as f:

            # extra info
            to_write.update({
                'time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'steps': self.env.curr_time,
                'jobselector': self.test_case.jobselector.name,
                'gpuselector': self.test_case.gpuselector.name,
            })

            for dumper in self.dumper:
                to_write.update(dumper.dump())

            json.dump(to_write, f)
        print('dumped to ', dump_dir)

    def watch(self, env: testEnv):
        if self.iswatching:
            raise Exception('already watching')
        self.original_step_func = env.step
        env.step = self.env_step_interceptor(env.step)
        self.iswatching = True
        pass

    def unwatch(self, env: testEnv):
        env.step = self.original_step_func
        self.iswatching = False
        pass

    def env_step_interceptor(self, func):
        def wrapper(*args, **kwargs):
            # signal dumpers
            [dumper.snap(self.env, *args, **kwargs) for dumper in self.dumper]
            _return = func(*args, **kwargs)
            if self.env.done:
                for dumper in self.dumper:
                    dumper.snap(self.env, *args, **kwargs)
                self.dump()
                self.unwatch(self.env)
            return _return
        return wrapper

    def __del__(self):
        self.unwatch(self.env)


def dump(testCase: TestCase, logdir: str = 'testcase'):
    '''
    To view the process of assigning GPUs, use dump() method to dump the environment to a file and view it with GUI.py
    '''
    return Dump_env(testCase, logdir)
