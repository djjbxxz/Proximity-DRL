import numpy as np
from scipy.interpolate import interp1d

__all__ = ['Job']


class Job():
    """The class to hold the job object.

    Attributes
    ----------
    job_id : int
        A unique identifier associated with each job.
    progress : int
        A variable that indicates the progress of the job, the job should finish
        when it reaches `len`
    gpu_request : int
        The number of gpu requested for the job, it will not run unless this number
        is satisfied.
    job_len : int
        The length of the job, it represents the maximum time step that a job can run.
    arrival_time: int
        The arrival time of the job at the jobqueue, it represents the waiting time of this job.
    d_ex : float
        Model flops in Flops. distance per example for this job
    m_j : int
        distance per minibatch for this job
    d_m = self.d_ex * self.m #Model flops computed per minibatch
    gradsize : int
        size of the gradient of the job, which is equal to total size of the parameters in MB
    tt_m : float
        train time per minibatch of the job
    d_f : float
        final|total computational distance of the job

    Methods
    -------
    forward_one_step()
        Add one on the progress of this job.
    print_profile()
        Print the attributes of the job
    """

    def __init__(self, id: int, gpu_request: int, len: int, ts_0j, d_exj_tlop, tt_m, m, d_fj_tflop, gradsize):
        """Constructor of the job

        Parameters
        ----------
        progress : int
            the progress of this job
        gpu_request : int
            The number of gpu requested for this job.
        job_len : int
            The number of time step that this job can run
        arrival_time: int
            The arrival time of the job
        status : {'waiting', 'running', 'paused'}
            The status of the job

        """
        self.job_id = id

        self.gpu_request = gpu_request
        self.job_len = len
        self.num_cells = self.gpu_request * self.job_len

        self.arrival_time = ts_0j
        self.start_time = -1
        self.finish_time = -1
        self.slowdown = 0
        self.stepcounter = 0
        self.pseudo_step = 0

        self.d_ex = d_exj_tlop
        self.m = m
        self.d_m = self.d_ex * self.m
        self.gradsize = gradsize

        self.d_f = d_fj_tflop
        self.d_done = 0
        self.tt_m = tt_m
        self.rt_m = 0

        self.v_m = 0
        self.color = None
        self.ts_togo = 0
        self.ts_done = 0
        self.singlejoblimbw = np.Inf
        self.multijoblimbw = np.Inf
        self.scale = 1
        self.gpus = ([], [], [])
        self.status = 'waiting'
        self.prevstatus = 'waiting'
        self.waiting_time = 0
        self.slowdown = 0


        self.communication_cost = 1

    def is_done(self):
        """Check whether the job is done

        Returns
        -------
        bool
            True if the job is done, False otherwise
        """
        return self.d_done >= self.d_f

    def get_progress(self)->float:
        progress = self.d_done/self.d_f
        progress = progress if progress <= 1.000 else 1.000
        progress = "{:.3f}".format(progress)
        return progress


    def print_profile(self):
        """Print the attributes of the job

        Used for debug
        """
        print(f'id: {self.job_id}')
        print(f'gpu_request: {self.gpu_request}')
        print(f'job_len: {self.job_len}')
        print(f'status: {self.status}')
        print(f'progress: {self.progress}')



