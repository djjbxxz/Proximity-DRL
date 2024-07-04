import numpy as np
from scipy.stats import skewnorm
from environment.env_util import *
import csv
import math
import sys
import os
__all__ = ['DataGenerating']

class DataGenerating:
    """A class that generates data that simulate the deep learning training tasks.

    Attributes
    ----------
    max_gpu_request : int
        Maximum number of gpu request for one job
    max_len : int
        Maximum length, which is the numer of time steps to complete, of one job
    seed : int
        The random_gpu seed that used during data generating

    Methods
    -------
    data_gen(size, skew1, skew2)
        Generate a list of job len and gpu request of job with size `size`, that follow skewed normal
        distribution with skewness skew1 and skew2.
    """
    def __init__(self, max_gpu_request, max_len, resources, seed=30, v_c_tfps=10.6, ret_reducer=1.0, time_horizon=20):
        """Constructor of the DataGenerating

        Parameters
        ----------
        max_gpu_request : int
            The maximum gpu request for a job
        max_len : int
            The maximum time step that a job can run
        seed : int, optional, default: 30
            The random_gpu seed used in job generating
        v_c_tfps : float
            Floating Points Operations Per Second on NVIDIA p100
        ret_reducer : float
        time_horizon : int
            number of time steps in the graph

        Attributes
        ----------
        resources : ndarray
            Cluster environment
        max_gpu_request : int
            Maximum gpu request
        max_len : int
            Maximum job length
        seed : int
            Random seed
        v_c_tfps : float
            Floating Points Operations Per Second on NVIDIA p100
        ret_reducer : float
        dnnproperties : dict
            Dictionary stores DNN_properties
        d_exc_tflop : float
            tflop size of vgg-vd-19 model, distance per
        m_c : int
            Minibatch size of calibrating NN

        """
        self.resources = resources
        self.max_gpu_request = max_gpu_request
        self.max_len = max_len
        self.seed = seed
        self.v_c_tfps = v_c_tfps
        self.ret_reducer = ret_reducer

        dirname = os.path.dirname(__file__)
        csvfile = os.path.join(dirname, 'DNN_properties.csv')
        a_csvfile = open(csvfile, 'r')
        input_file = csv.DictReader(a_csvfile)
        self.dnnproperties = []
        for row in input_file:
            self.dnnproperties.append(row)
        a_csvfile.close()

        self.d_exc_tflop = 20e-3  # tflop size of vgg-vd-19 model, distance per
        self.m_c = 256  # minibatch size of calibration NN (neural network)
        self.d_mc_tflop = self.d_exc_tflop * self.m_c
        self.it_fc = 1 * 10000  # total|final iterations calibration NN. Full
        self.d_fc_tflop = self.it_fc * self.d_mc_tflop
        self.t_fc = self.d_fc_tflop / self.v_c_tfps  # final|total time for calibration NN
        self.ts_hc = time_horizon  #

        self.d_tsc_tflop = self.d_fc_tflop / self.ts_hc  # distance per row|timestep
        self.t_tsc = self.t_fc / self.ts_hc  # ~ 483/ts_hc = 483/20 = 24.15 sec



    def data_gen(self, size : int, skew1 : float=0.0, skew2 : float=0.0):
        """Function that generates a list of job len and gpu_request followed a distribution

        Parameters
        ----------
        size : int
            The size of the generating data
        skew1 : float, optional, default: 0.0
            The skewness of the distribution of job len, negative is left skewed,
            positive is right skewed, default 0 is normal distribution
        skew2 : float, optional, default: 0.0
            The skewness of the distribution of gpu request, negative is left skewed,
            positive is right skewed, default 0 is normal distribution

        Returns
        -------
        len_list : ndarray
            A list of job length that follows the desired distribution
        gpu_request_list : ndarray
            A list of gpu request that follows the desired distribution
        """
        len_seq = skewnorm.rvs(a=skew1, loc=self.max_len, size=size)
        len_seq = len_seq - min(len_seq)
        len_seq = len_seq / max(len_seq)
        len_seq = len_seq * self.max_len
        len_seq = np.where(len_seq < 1, 1, len_seq)

        gpu_seq = skewnorm.rvs(a=skew2, loc=self.max_gpu_request, size=size)
        gpu_seq = gpu_seq - min(gpu_seq)
        gpu_seq = gpu_seq / max(gpu_seq)
        gpu_seq = gpu_seq * self.max_gpu_request
        gpu_seq = np.where(gpu_seq < 1, 1, gpu_seq)

        return len_seq.astype(int), gpu_seq.astype(int)

    def data_gen_with_p(self, size: int):
            self.job_cat = [0.3, 0.4, 0.8, 1]
            self.gpu_cat = [0.2, 0.3, 0.6, 1]
            self.probs = [0.5, 0.1, 0.2, 0.2]
            lower_bound = 1
            upper_bound_len = self.max_len + 1  # The upper bound is exclusive in randint
            upper_bound_gpu = self.max_gpu_request + 1

            # Calculate the number of integers for each sub-range
            count_short = int(size * self.probs[0])
            count_medium = int(size * self.probs[1])
            count_long = int(size * self.probs[2])
            count_extreme = size - count_short - count_medium - count_long

            # Function to safely generate random integers
            def safe_randint(low, high, count):
                if low >= high:  # Adjust high if low is equal to or greater than high
                    high = low + 1
                return np.random.randint(max(1, low), high, count)  # Ensure low is at least 1

            len_short = safe_randint(lower_bound, max(2, math.ceil(self.job_cat[0] * upper_bound_len)), count_short)
            len_medium = safe_randint(max(2, int(self.job_cat[0] * upper_bound_len)),
                                    int(self.job_cat[1] * upper_bound_len), count_medium)
            len_long = safe_randint(int(self.job_cat[1] * upper_bound_len), int(self.job_cat[2] * upper_bound_len),
                                    count_long)
            len_extreme = safe_randint(int(self.job_cat[2] * upper_bound_len), upper_bound_len, count_extreme)

            gpu_less = safe_randint(lower_bound, max(2, math.ceil(self.gpu_cat[0] * upper_bound_gpu)), count_short)
            gpu_normal = safe_randint(max(2, int(self.gpu_cat[0] * upper_bound_gpu)),
                                    int(self.gpu_cat[1] * upper_bound_gpu), count_medium)
            gpu_many = safe_randint(int(self.gpu_cat[1] * upper_bound_gpu), int(self.gpu_cat[2] * upper_bound_gpu),
                                    count_long)
            gpu_extreme = safe_randint(int(self.gpu_cat[2] * upper_bound_gpu), upper_bound_gpu, count_extreme)

            len_seq = np.concatenate((len_short, len_medium, len_long, len_extreme))
            gpu_seq = np.concatenate((gpu_less, gpu_normal, gpu_many, gpu_extreme))


            np.random.shuffle(len_seq)
            np.random.shuffle(gpu_seq)

            return len_seq, gpu_seq

    def dnn_data_gen(self, gpu_request, job_len):
        choice = np.random.choice(len(self.dnnproperties), replace=True)
        gradsize = int(self.dnnproperties[choice]['param mem (MB)'])
        m = np.random.choice([32, 64, 128, 256], replace=True)
        d_ex = float(self.dnnproperties[choice]['flops (GFLOPS)']) * 0.001
        d_mj_tflop = d_ex * m
        tt_m = np.divide(d_mj_tflop, self.v_c_tfps)
        gpus_per_rack = self.resources.shape[1] * self.resources.shape[2]
        numracks = num_racks_cal(gpu_request, gpus_per_rack)
        _, v_m_theor = calc_speed_single(gradsize, gpu_request, d_mj_tflop, tt_m, numracks, self.ret_reducer)
        d_fj_tflop = v_m_theor * self.t_tsc * job_len
        return d_ex, m, tt_m, d_fj_tflop, gradsize


    def jobadvancecompdist(self, job):
        # distance travelled by the job this step.
        # Yiming: instead of using  t_tsc, use time since last delta distance
        # calcuation. e.g. currenttime - job.prev_calc_time.

        # delta_distance = job.v_m * (currenttime - job.prev_calc_time)
        distance = job.v_m * self.t_tsc

        # job.prev_calc_time = current time in seconds

        return distance

    def jobnumrowstime(self, job):
        """
        Count job's number of time rows. job's stepcounter and d_done must
        already have been incremented, as we expect they are non zero.

        Important: Does not mutate job object

        :param self:
        :param job:
        :type Job:
        :param numgpusin: numgpus assigned in this step.
        :type numgpusin:
        :returns:
            - ts_togo - extrapolated number of timesteps left based
                number of gpus assigned in the current step.
            - ts_done - extrapolated backwares the number of
                timesteps done based on number of gpus assigned in the
                current step.

        """
        # numgpus = numgpusin
        # if numgpus is None:
        #     numgpus = job.g
        # jobdistleft = job.d_rem
        ts_done = np.int32(np.floor(job.stepcounter))
        gpus_per_rack = self.resources.shape[1] * self.resources.shape[2]
        if job.d_done == 0:
            v = calc_job_minbatch_speed(job=job, singleormulti='single',gpus_per_rack=gpus_per_rack, ret_reducer=self.ret_reducer)
            ts_togo = job.d_f / (v * self.t_tsc)
            ts_togo = np.int32(ts_togo)
        else:
            ts_togo = np.ceil(job.stepcounter * (job.d_f - job.d_done) / job.d_done)
            ts_togo = np.int32(ts_togo)


        job.ts_togo = ts_togo
        job.ts_done = ts_done

        return ts_togo, ts_done




