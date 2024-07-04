from ..base import JobSelector
from environment.env import Env
import numpy as np


class SjfJobSelector(JobSelector):
    def __init__(self):
        super().__init__('Short Job First')

    def select(self, jobqueue: list, resources: list, env: Env) -> int or None:
        if len(jobqueue) == 0:
            return None
        jobqueue=list(jobqueue)
        return jobqueue.index(min(list(jobqueue), key=lambda x: x.job_len))

