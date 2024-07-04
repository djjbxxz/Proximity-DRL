import numpy as np
import sys
sys.path.append('.')
from environment.env import Env
from environment.gpu_select_env import GpuSelectEnv


def get_env_job_load(env: Env) -> float:
    assert isinstance(env, Env)
    job_len = np.array([job.job_len for job in env.job_seq])
    gpu_requested = np.array([job.gpu_request for job in env.job_seq])
    return np.sum(job_len*gpu_requested)


if __name__=="__main__":
    env = GpuSelectEnv()
    env.reset()
    print(get_env_job_load(env))