from ..base import JobSelector
from environment.env import Env
import numpy as np


class firstcomesfirstserve(JobSelector):
    def __init__(self):
        super().__init__('firstComesFirstServe')

    def select(self, jobqueue: list, resources: list, env: Env) -> int or None:
        return 0 if env.jobqueue.size > 0 else None
