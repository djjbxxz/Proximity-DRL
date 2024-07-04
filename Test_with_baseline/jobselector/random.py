from ..base import JobSelector
from environment.env import Env
import numpy as np


class RandomJobSelector(JobSelector):
    def __init__(self):
        super().__init__('random')

    def select(self, jobqueue: list, resources: list, env: Env) -> int or None:
        return np.random.randint(0, env.jobqueue.size)if env.jobqueue.size > 0 else None
