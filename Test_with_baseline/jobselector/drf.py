from ..base import JobSelector
from environment.env import Env
import numpy as np


class DRF(JobSelector):
    '''
    `DRF`: Dominant Resource Fairness

    This method selects a job which has the smallest dominant share.

    Select the job which has the smallest number of requested GPU.
    '''
    def __init__(self):
        super().__init__('DRF')

    def select(self, jobqueue: list, resources: list, env: Env) -> int or None:
        if len(jobqueue) == 0:
            return None
        jobqueue=list(jobqueue)
        return jobqueue.index(min(list(jobqueue), key=lambda x: x.gpu_request))

