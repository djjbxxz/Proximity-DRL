from enum import Enum

from environment.env import *
from environment.env_util import *
from env_components.job import Job
import numpy as np
import yaml

from environment.env_util import DotDict
__all__ = ['GpuSelectEnv']


with open('environment/config.yaml') as f:
    pa = DotDict(yaml.load(f, Loader=yaml.SafeLoader))
INVALID_ACTION_PENALTY = pa.invalid_action_penalty
TIME_PENALTY = pa.time_penalty
IDLE_REWARD = pa.idle_reward
REWARD_ASSIGN_1GPU = pa.reward_assign_1gpu
REWARD_EPISODE_END = pa.reward_episode_end
UTILIZAATION_SCALE = pa.utilization_scale
del pa


class Action_type(Enum):
    VALID = 0
    INVALID = 1
    ASSIGN_1GPU = 2
    IDLE = 3

communication_cost_ranges = [[0.775,0.775],[0.8,1.55],[0.825,1.825],[0.85,2.1],[1.125,2.375],[1.15,2.65],[1.175,2.925],[1.2,3.2]]
def reward_func_communicationCost(action:np.ndarray, communication_cost:float):
    num_gpus = action.sum()
    if num_gpus == 1:
        return 0 #avoid divide by zero
    min,max = communication_cost_ranges[num_gpus-1]
    return 1-((communication_cost-min)/(max-min))

# gpus		best	worst
# 1	 	0.775	0.775
# 2		0.8	1.55
# 3		0.825	1.825
# 4		0.85	2.1
# 5		1.125	2.375
# 6		1.15	2.65
# 7		1.175	2.925
# 8		1.2	3.2

def reward_func_simple_communicationCost(action:np.ndarray, communication_cost:float):
    return -(communication_cost-0.775)/2.425

class GpuSelectEnv(Env):

    def __init__(self, allow_idle=True, config_filepath=None, test_mode=False):
        config_filepath = config_filepath if config_filepath is not None else 'environment/config.yaml'
        with open(config_filepath) as f:
            pa = DotDict(yaml.load(f, Loader=yaml.SafeLoader))
        super().__init__(pa)
        self.allow_idle = allow_idle
        self.max_episode_len = pa.max_episode_len
        self.input_channel = self.resources.size*2+2
        self.num_actions = self.resources.size+1  # +1 for no gpu assigned
        self.action_cache = []
        self.ob_cache = None
        self.extended_time = 0
        self.test = test_mode
        if test_mode:
            Env.seed(self)
            self.utilization_rate_record = []

    def step(self, action: int):
        self.extended_time += 1
        if self.done:
            self.reset()

        if self.ob_cache is None:
            self.ob_cache = self.observe()
        assert isinstance(action, int)

        # if the action is no gpu assigned, advance all the running jobs for one step
        if action == self.num_actions-1:
            return self._real_step(np.zeros(shape=self.num_actions-1, dtype=int))

        # action invalid condition:
        # 1. the gpu was already selected
        # 2. the gpu is cached by previous selection
        # 3. No job in the queue
        if self.current_job is None or \
                action in self.action_cache or \
                not self._get_Gpu_available()[action]:
            if self.test:
                return np.array(self.ob_cache, copy=True), INVALID_ACTION_PENALTY, False, Action_type.INVALID
            else:
                return np.array(self.ob_cache, copy=True), INVALID_ACTION_PENALTY, False

        self.action_cache.append(action)
        self.ob_cache[self.action_cache] = 1
        remaining_gpus_to_assign = self.current_job.gpu_request - \
            len(self.action_cache)
        if remaining_gpus_to_assign > 0:
            if self.test:
                return np.array(self.ob_cache, copy=True), REWARD_ASSIGN_1GPU, False, Action_type.ASSIGN_1GPU
            else:
                return np.array(self.ob_cache, copy=True), REWARD_ASSIGN_1GPU, False
        else:  # remaining_gpus_to_assign == 1:# last gpu to assign
            f_action = np.zeros(shape=self.num_actions-1, dtype=int)
            f_action[self.action_cache] = 1
            return self._real_step(f_action)

    def _real_step(self, action: np.ndarray):
        ''' 
        Parameters
        ----------
        action: np.ndarray
            Whichever gpu is to be taken is set to 1
        '''
        self.curr_time += 1
        # useful = self.check_useful()
        selected_gpu = self.unflatten_action(action)

        self.action_cache.clear()
        if action.sum() == 0:  # action: no gpu selected
            reward = IDLE_REWARD
            if self.test:
                action_type = Action_type.IDLE
        else:  # action is valid
            job = self.current_job
            selected_job_index = list(self.jobqueue).index(job)
            if self.test:
                action_type = Action_type.VALID
            self.assign_job_gpus_non_preemptive(
                selected_job_index, selected_gpu)
            reward = reward_func_communicationCost(action,job.communication_cost)
            if np.random.random() <= self.new_job_rate and self.j_id < self.target_num_job_arrive:
                self.insert_new_job()

        self.total_step += 1

        self.avg_running_jobs = np.append(
            self.avg_running_jobs, len(self.running_queue))

        self.advance_runningjobs_onestep()
        self.update_resources()
        self.episode_reward = np.append(
            self.episode_reward, reward)

        done_jobs = self.get_done_jobs()

        self.remove_jobs(done_jobs)
        if self.extended_time >= self.max_episode_len or all([job.is_done() for job in self.job_seq]):
            self.done = True
            reward = REWARD_EPISODE_END
        if self.test:
            self.utilization_rate_record.append(self.get_utilization_rate())
        ob = self.observe()
        self.ob_cache = np.array(ob, copy=True)
        if self.test:
            return ob, reward, self.done, action_type
        else:
            return ob, reward, self.done

    def observe(self):
        '''
        Observation include:\n
        `1. Gpu availability`\n
        `2. Job length`\n
        `3. Gpu num requested`\n
        '''
        job = self.current_job
        job_len, gpu_request = (0, 0) if job is None else (
            job.job_len, job.gpu_request)

        count_down = np.zeros(shape=self.resources.shape)
        for job in self.running_queue:
            count_down[job.gpus] = job.ts_togo

        # Normalization
        count_down = np.clip(count_down/50, 0, 1)
        job_len = job_len/self.max_job_len
        gpu_request = gpu_request / self.max_gpu_request

        # Assemble
        ob = np.zeros((count_down.size*2+2))
        ob[-count_down.size:] = count_down.flatten()
        ob[count_down.size] = job_len
        ob[count_down.size+1] = gpu_request
        return ob

    def is_action_valid(self, job: Job, action: np.ndarray):
        '''
        Test whether the selected gpus are available and
        the num of Gpu selected is equal to the num Gpu required in the job
        '''

        if action.sum() == 0:  # action: no gpu selected
            return True
        if job is None:  # no job in the queue and action is not empty
            return False
        return True not in ((self._get_Gpu_available()-action) == -1) and np.where(action == 1)[0].size == job.gpu_request

    def is_done(self):
        return self.done

    @property
    def current_job(self) -> Job:
        return self._first_job_action()

    def _first_job_action(self) -> Job:
        if len(self.jobqueue) > 0:
            return self.jobqueue[0]
        else:
            return None

    def _tetris_action(self) -> Job:
        if len(self.jobqueue) == 0:
            return None
        return max(list(self.jobqueue), key=lambda x: x.gpu_request)

    def unflatten_action(self, action: np.ndarray):
        '''
        Convert the action from 1d to 2d
        '''
        action = action.reshape(self.resources.shape)
        return np.where(action == 1)

    def get_utilization_rate(self):
        '''Uutilization rate range from 0 to 1'''
        return 1-(self._get_Gpu_available().sum()/self.resources.size)

    def reset(self):
        self.action_cache.clear()
        self.ob_cache = None
        self.extended_time = 0
        if self.test:
            self.utilization_rate_record.clear()
        return super().reset()

    def get_valid_mask(self) -> np.ndarray:
        '''
        Return a mask indicates which action is valid, 1 means valid, 0 means invalid
        '''
        available_gpus = self._get_Gpu_available()
        if self.current_job is None or self.current_job.gpu_request > available_gpus.sum():
            mask = np.zeros(shape=self.num_actions, dtype=int)
            mask[-1] = 1
            return mask

        mask = np.ones(shape=self.num_actions-1, dtype=int)
        mask[self.action_cache] = 0
        np.bitwise_and(mask, available_gpus, out=mask)
        # idle is only allowed when there is no gpu is cached
        mask = np.append(mask, 1 if self.allow_idle and len(self.action_cache)==0 else 0)
        return mask
