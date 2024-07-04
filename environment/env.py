import json
from abc import ABC, abstractmethod
# from pickletools import int4

# import networkx
import numpy as np
# import torch
# from scipy.interpolate import interp1d

from env_components.gpu import Gpu
# from env_components.parameters import Parameters
from env_components.topology import NetworkTopology
from env_components.job import *

from env_components.data_generator import *
from environment.env_util import *
# from environment.env_util import choose_gpu_fn, gpu_assign_time_status_fn, compute_GPU_distances
from typing import List, Deque
from collections import deque
from scipy import stats
# import math

########## Hossein ###########
# import pdb
import random
# from matplotlib import pyplot as plt

##############################


__all__ = ['Env']


class Env(ABC):
    """Metaclass for defining customized environments.

    Use this metaclass to create an environment. All customized environments
    are subclasses from this one. In order to create it, step() and observe()
    must be implemented.

    Attributes
    ----------
    args : Argument
        Parameter of the environment
    done : bool
        A varible that indicates whether the environment finished its work or not ()
        (used during the training)
    resources : ndarray
        Stores all resources, each cluster contains a set of gpus
    jobqueue : ndarray
        Stores waiting jobs, a maximum length should constraint the jobqueue
    runningjobqueue : ndarray
        Stores the running jobs
    jobqueue_maxlen : int
        The maximum length of the jobqueue
    datagenerator : DataGenerating
        The generator to generate the job attributes
    backlog : Deque
        Stores the waiting jobs when the jobqueue is full.
    len_seq : ndarray
        The sequence of job length
    gpu_request_seq : ndarray
        The sequence of gpu request of jobs

    Methods
    -------
    reset()
        Reset the environment
    step(action)
        Execute an action on the environment
    observe()
        Returns information of the environment in the designed format.
    reward()
        Returns the reward from the step function
    generate_job_sequence(size, skew1, skew2)
        Generate the job_len sequence and gpu request sequence.
    insert_new_job()
        Insert a new job into the jobqueue

    progress_all_job()
        Progress all running jobs
    get_avl_gpus(*c_idx)
        Find all available gpus in the environment or selected resources
    get_running_jobs()
        Get the index of running jobs
    get_waiting_jobs()
        Get the index of waiting jobs
    get_paused_jobs()
        Get the index of paused jobs
    get_done_jobs()
        Get the index of finished jobs
    get_first_k_gpus(k)
        Get k random_gpu avalible gpus

    """

    def __init__(self, args):
        """Constructor of the environment

        Parameters
        ----------
        pa : Parameters
            The object that stores all customized attributes of the environment
        """
        self.done = False
        self.num_racks_per_cluster = args.num_racks_per_cluster
        self.num_machines_per_rack = args.num_machines_per_rack
        self.num_gpus_per_machine = args.num_gpus_per_machine
        self.max_gpu_request = args.max_gpu_request
        self.max_job_len = args.max_job_len
        self.jobqueue_maxlen = args.jobqueue_maxlen
        self.max_backlog_len = args.max_backlog_len
        self.new_job_rate = args.new_job_rate
        self.target_num_job_arrive = args.target_num_job_arrive
        self.ret_reducer = args.ret_reducer
        self.gpu_request_skew = args.gpu_request_skew
        self.job_len_skew = args.job_len_skew
        self.max_num_timesteps = args.max_num_timesteps
        self.preemptive = args.preemptive
        # Host class, contains gpu
        self.resources = -np.ones((self.num_racks_per_cluster, self.num_machines_per_rack,
                                   self.num_gpus_per_machine), dtype=np.int32)  # Hossein

        self.resources_utilization = np.zeros(shape = (self.num_racks_per_cluster, self.num_machines_per_rack,
                                               self.num_gpus_per_machine), dtype=np.int32)

        self.gpus_array = np.array([], dtype=Gpu)

        self.jobqueue = np.array([])
        self.running_queue = np.array([])
        self.backlog = deque(maxlen=self.max_backlog_len)

        self.j_id = 0
        self.total_step = 0
        self.num_job_finished = 0

        self.datagenerator = DataGenerating(max_gpu_request=self.max_gpu_request, max_len=self.max_job_len,
                                            resources=self.resources,
                                            ret_reducer=self.ret_reducer)
        self.len_seq = np.array([])
        self.gpu_request_seq = np.array([])
        self.gpus_per_rack = self.resources.shape[1] * self.resources.shape[2]
        self.curr_time = 0
        self.episode_reward = np.array([])
        self.slowdowns= np.array([])
        self.avg_running_jobs = np.array([])
        self.avg_throughput = np.array([])
        self.topology_parameters = NetworkTopology(args)

    def reset(self):
        """Reset the environment

        """
        self.done = False

        self.resources = -np.ones((self.num_racks_per_cluster, self.num_machines_per_rack,
                                   self.num_gpus_per_machine), dtype=np.int32)  # Hossein

        self.resources_utilization = np.zeros(shape = (self.num_racks_per_cluster, self.num_machines_per_rack,
                                               self.num_gpus_per_machine), dtype=np.int32)

        self.gpus_array = np.array([], dtype=Gpu)
        self.jobqueue = np.array([])
        self.running_queue = np.array([])
        self.backlog = deque(maxlen=self.max_backlog_len)

        self.j_id = 0
        self.total_step = 0
        self.num_job_finished = 0

        self.datagenerator = DataGenerating(max_gpu_request=self.max_gpu_request, max_len=self.max_job_len,
                                            resources=self.resources,
                                            ret_reducer=self.ret_reducer)
        self.len_seq = np.array([])
        self.gpu_request_seq = np.array([])
        self.gpus_per_rack = self.resources.shape[1] * self.resources.shape[2]
        self.curr_time = 0
        self.episode_reward = np.array([])
        self.slowdowns = np.array([])
        self.avg_running_jobs = np.array([])
        self.avg_throughput = np.array([])
        self.generate_job_sequence(self.target_num_job_arrive)
        self.fill_jobqueue()
        return self.observe()

    @abstractmethod
    def step(self, action):
        """Execute an action on the environment

        This function should be implemented by users for different purposes.

        Raises
        ------
        NotImplementedError
            If the action is none
        """
        if action is None:
            raise NotImplementedError("The step function has not defined yet")

    def generate_job_sequence(self, size: int):

        """Function calls the datagenerating function in data_generator class

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
        A list of job len and gpu request
        """
        self.job_seq = []
        # skew1 = self.job_len_skew
        # skew2 = self.gpu_request_skew
        # self.len_seq, self.gpu_request_seq = self.datagenerator.data_gen(size, skew1, skew2)
        self.len_seq, self.gpu_request_seq = self.datagenerator.data_gen_with_p(size)
        for i in range(self.target_num_job_arrive):
            d_ex, m, tt_m, d_fj_tflop, gradsize = self.datagenerator.dnn_data_gen(self.gpu_request_seq[self.j_id],
                                                                                  self.len_seq[self.j_id])
            j = Job(id=i, gpu_request=self.gpu_request_seq[i],
                    len=self.len_seq[i], ts_0j=self.curr_time, d_exj_tlop=d_ex, tt_m=tt_m,
                    m=m, d_fj_tflop=d_fj_tflop, gradsize=gradsize)
            self.job_seq.append(j)


    def insert_new_job(self):
        """Insert a new job into the jobqueue

        Returns
        -------
        None
        """
        new_job = self.job_seq[self.j_id]
        new_job.arrival_time = self.curr_time
        if len(self.jobqueue) < self.jobqueue_maxlen:
            if len(self.backlog) > 0:
                self.jobqueue = np.append(self.jobqueue, self.backlog.popleft())
            elif (self.num_job_finished + len(self.jobqueue) + len(self.backlog)) < self.target_num_job_arrive:

                self.jobqueue = np.append(self.jobqueue, new_job)
                self.j_id += 1
            else:
                return False
        elif len(self.backlog) < self.max_backlog_len and \
                (self.num_job_finished + len(self.jobqueue) + len(self.backlog)) < self.target_num_job_arrive:

            self.backlog.append(new_job)
            self.j_id += 1
        else:
            return False

        return True



    def advance_runningjobs_onestep(self):
        gpus_per_rack = self.resources.shape[1] * self.resources.shape[2]
        if self.preemptive:
            runnings = self.jobqueue[self.get_running_jobs()]
            waitings = self.jobqueue[self.get_waiting_jobs()]
        else:
            runnings = self.running_queue
            waitings = self.jobqueue
        for job in runnings:
            job.v_m, job.rt_m = calc_job_minbatch_speed(job, gpus_per_rack, self.ret_reducer, singleormulti='multi',
                                                        outdetails=True)
            #print(f'pre {job.v_m} aft {job.v_m/job.communication_cost} --- {job.communication_cost}')
            job.v_m = job.v_m / job.communication_cost
            d_delta = self.datagenerator.jobadvancecompdist(job)
            # print(f'{job.job_id} {d_delta}')
            job.d_done += d_delta
            job.stepcounter += 1
            self.resources_utilization[job.gpus] += 1
            if job.is_done():
                job.status = 'finished'
                job.finish_time = self.curr_time
            job.ts_togo, job.ts_done = self.datagenerator.jobnumrowstime(job)
        for job in waitings:
            job.waiting_time += 1
        
     

    def remove_jobs(self, jobs):
        if self.preemptive:
            for j in jobs[0]:
                self.resources[self.jobqueue[j].gpus] = -1

            self.jobqueue = np.delete(self.jobqueue, jobs)
        else:
            for j in jobs[0]:
                self.resources[self.running_queue[j].gpus] = -1

            self.running_queue = np.delete(self.running_queue, jobs)

    def assign_job_gpus_preemptive(self, j_idx, gpus: List[tuple]) -> bool:
        """Assign gpus to a job

        Parameters
        ---------
        j_indx : int
            The index of the job
        gpus : List[tuple]
            The coordinates of gpus, first item in the tuple contains the cluster index,
            the second item contains the gpu index in the cluster. e.g. ((0,0), (0,1))
            represents the first and the second gpu in the first cluster.

        Returns
        -------
        bool
            Indicate the job has been allocated or not
        """
        if all(x == -1 for x in self.resources[gpus]):
            #self.resources[gpus] = self.jobqueue[j_idx].job_id
            self.jobqueue[j_idx].status = 'running'
            self.jobqueue[j_idx].gpus = gpus
            slowdown = (self.curr_time +  self.jobqueue[j_idx].job_len - self.jobqueue[j_idx].arrival_time)
            # slowdown = (self.curr_time - self.jobqueue[j].arrival_time)/self.jobqueue[j].job_len
            self.jobqueue[j_idx].slowdown = slowdown
            self.slowdowns = np.append(self.slowdowns, slowdown)
            # slowdown = self.curr_time - self.jobqueue[j_idx].arrival_time
            # self.jobqueue[j_idx].slowdown = slowdown
            # print(f'{self.jobqueue[j_idx].arrival_time} {self.curr_time}')

            # self.slowdowns= np.append(self.slowdowns, slowdown)

            gpus_graph = self.topology_parameters.gpus_string_converter(gpus)
            routes, path = self.topology_parameters.find_routes(gpus_graph)
            length = 0
            for pair in path:
                length += (self.topology_parameters.G_clutser[pair[0]][pair[1]]["weight"]*0.5)

            self.jobqueue[j_idx].communication_cost = length if length > 1 else 1

            return True
        return False

    def assign_job_gpus_non_preemptive(self, j_idx, gpus: List[tuple]) -> bool:
        if all(x == -1 for x in self.resources[gpus]):
            self.jobqueue[j_idx].status = 'running'
            self.jobqueue[j_idx].gpus = gpus
            slowdown = (self.curr_time - self.jobqueue[j_idx].arrival_time)
            self.jobqueue[j_idx].slowdown = slowdown
            self.slowdowns = np.append(self.slowdowns, slowdown)
            gpus_graph = self.topology_parameters.gpus_string_converter(gpus)
            routes, path = self.topology_parameters.find_routes(gpus_graph)
            #round to avoid float point calculation error
            length = np.round(np.sum([self.topology_parameters.G_clutser[pair[0]][pair[1]]["weight"] for pair in path]),3)
            self.jobqueue[j_idx].communication_cost = length# if length > 1 else 1
            self.running_queue = np.append(self.running_queue, self.jobqueue[j_idx])
            # pre_len = len(self.jobqueue)
            # job = self.jobqueue[j_idx]
            self.jobqueue = np.delete(self.jobqueue, j_idx)
            # assert pre_len == (len(self.jobqueue) + 1)
            # assert job not in self.jobqueue
            # assert job in self.running_queue

            return True
        return False




    def update_resources(self):
        self.resources=-np.ones((self.num_racks_per_cluster, self.num_machines_per_rack,
                  self.num_gpus_per_machine), dtype=np.int32)
        if self.preemptive:
            q = self.jobqueue
        else:
            q = self.running_queue
        for j_idx in range(len(q)):
            gpus = q[j_idx].gpus
            self.resources[gpus] = q[j_idx].stepcounter
            # assert q[j_idx].job_len >= q[j_idx].stepcounter
            # assert (q[j_idx].job_len - q[j_idx].stepcounter) > -1


    def random_select_k_gpus_for_job(self, j, random=False):
        """Randomly select k gpus for the target job j

        Parameters
        ---------
        j : int
            The index of the job in the jobqueue

        Returns
        -------
        selected gpus : (ndarray, ndarray)
            A set of selected gpus
        """
        q = self.jobqueue
        if j < len(q):
            gpu_request = q[j].gpu_request
            avl_gpus = self.get_avl_gpus()
            if len(avl_gpus[0]) >= 1 and len(avl_gpus[0]) >= gpu_request:
                return self.get_first_k_gpus(gpu_request, random_gpu=random)
            else:
                return ([], [], [])
        else:
            return ([], [], [])

    def get_avl_gpus(self, *c_idx):
        """Find all available gpus in the environment

        Parameters
        ---------
        c_idx : (idx) optional
            A list of index of resources, only find the avalible gpus inside
            the selected clutsers if this parameter is used.

            e.g. get_avl_gpus(0,1) will find all available gpus in the first and the second cluster.
        """
        if not c_idx:
            result = np.where(self.resources == -1)
        else:
            result = np.where(self.resources[np.array(c_idx)] == -1)
        return result



    def get_running_jobs(self):
        """Get the index of running jobs

        Returns
        -------
        running_jobs :
            The index of running jobs
        """

        def getter(j):
            return j.status

        vfunc = np.vectorize(getter, otypes=[str])
        running_jobs = np.where(vfunc(self.jobqueue) == 'running')
        return running_jobs

    def get_waiting_jobs(self):
        """Get the index of waiting jobs

        Returns
        -------
        running_jobs :
            The index of waiting jobs
        """

        def getter(j):
            return j.status

        vfunc = np.vectorize(getter, otypes=[str])
        waiting_jobs = np.where(vfunc(self.jobqueue) == 'waiting')
        return waiting_jobs

    def get_unfit_jobs(self):
        """Get the index of waiting jobs

        Returns
        -------
        running_jobs :
            The index of waiting jobs
        """
        avl_gpus = len(self.get_avl_gpus()[0])
        def getter(j):
            return j.gpu_request > avl_gpus

        vfunc = np.vectorize(getter, otypes=[bool])
        unfit_jobs= np.where(vfunc(self.jobqueue))
        return unfit_jobs

    def get_paused_jobs(self):
        """Get the index of paused jobs

        Returns
        -------
        running_jobs :
            The index of waiting jobs
        """

        def getter(j):
            return j.status

        vfunc = np.vectorize(getter, otypes=[str])
        paused_jobs = np.where(vfunc(self.jobqueue) == 'paused')
        return paused_jobs

    def get_done_jobs(self):
        """Get the index of finished jobs

        Returns
        -------
        done_jobs :
            Thie index of finished jobs
        """
        def getter(j):
            return j.d_done >= j.d_f
        # def getter(j):
        #     return j.stepcounter >= j.job_len
        if self.preemptive:
            runnings = self.jobqueue
        else:
            runnings = self.running_queue
        vfunc = np.vectorize(getter, otypes=[bool])
        done_jobs = np.where(vfunc(runnings))
        return done_jobs


    def get_first_k_gpus(self, k: int, random_gpu: bool=False) -> tuple:
        """Get random_gpu k available gpus

        Parametersgit
        ---------
        k : int
            The number of gpus that would like to pick up.

        Returns
        -------
        gpus : tuple
            A tuple represents the coordinates of selected gpus.git status
        """
        if random_gpu:
            max = len(self.get_avl_gpus()[0])
            if k <= max:
                selected = random.sample(range(0, max), k)
            else:
                selected = random.sample(range(0, max), 0)
            x0 = self.get_avl_gpus()[0][selected]
            x1 = self.get_avl_gpus()[1][selected]
            x2 = self.get_avl_gpus()[2][selected]
        else:
            x0 = self.get_avl_gpus()[0][:k]
            x1 = self.get_avl_gpus()[1][:k]
            x2 = self.get_avl_gpus()[2][:k]


        return (x0, x1, x2)

    def get_first_knn_gpus(self, k: int, random_gpu: bool = False) -> tuple:
        """Get random k available gpus

        Parametersgit
        ---------
        k : int
            The number of gpus that would like to pick up.

        Returns
        -------
        gpus : tuple
            A tuple represents the coordinates of selected gpus.git status
        """

        # print('This is get_random_gpus: ')

        """Get random k available gpus

        Parametersgit
        ---------
        k : int
            The number of gpus that would like to pick up.

        Returns
        -------
        gpus : tuple
            A tuple represents the coordinates of selected gpus.git status
        """

        def get_kth_dims(idx, zMax, xMax, yMax):  # returns the Kth element dimentions in the 3-darray of self.resources
            assert (idx < zMax * xMax * yMax)
            z = int(idx / (xMax * yMax))
            idx -= (z * xMax * yMax);
            y = int(idx / xMax)
            x = int(idx % xMax)
            return z, y, x

        first_avl_gpus = self.get_avl_gpus()

        # x_ = int(self.get_avl_gpus()[0][:1])####find the first idle gpu as before(first dim)
        # y_ = int(self.get_avl_gpus()[1][:1])####find the first idle gpu as before(second dim)
        # z_ = int(self.get_avl_gpus()[2][:1])####find the first idle gpu as before(third dim)

        WIDTH = self.num_racks_per_cluster  # 2
        DEPTH = self.num_machines_per_rack  # 4
        HEIGHT = self.num_gpus_per_machine  # 4

        each_rack_free_gpus = []
        each_rack_free_machine = []
        each_rack_free_machine_k = []
        most_epmpty_rack = 0

        # if len(np.where(self.resources == -1)[0])>k:
        #     print(len(np.where(self.resources == -1)[0]),k)

        for z in range(WIDTH):
            each_rack_free_gpus.append(
                len(np.where(self.resources[z, :, :] == -1)[0]))  # number of free gpus in rack 0 ... i
            erf = np.array(each_rack_free_gpus)

        # print('each_rack_free_gpus',each_rack_free_gpus)
        maxfreerack = each_rack_free_gpus.index(max(each_rack_free_gpus))  # max number of free gpus in each rack
        # print(erf[np.where( erf==np.min(erf[np.nonzero(erf)]))])
        if np.max(erf) == np.min(erf) and np.max(erf) != 0:
            minfreerack = each_rack_free_gpus.index(min(erf))
        else:
            minfreerack = each_rack_free_gpus.index(erf[np.where(erf == np.min(erf[np.nonzero(erf)]))])

        # print('maxfreerack,minfreerack',maxfreerack,minfreerack)
        if k > 4:  ##################### for jobs with more than 4 gpu request
            for f in range(DEPTH):
                each_rack_free_machine.append(
                    len(np.where(self.resources[maxfreerack, f, :] == -1)[0]))  # number of free gpus in rack 0 ... i
            # print(each_rack_free_machine)
            maxfreemachine = each_rack_free_machine.index(
                max(each_rack_free_machine))  # max number of free gpus in each rack
            # print('k>4',each_rack_free_machine)
            # print('maxfreemachine',maxfreemachine)
            a = np.where(self.resources[maxfreerack, maxfreemachine, :] == -1)[0][0]  # third dimension of a free GPU
            # print(maxfreerack,maxfreemachine,a)
            # resources[maxfreerack,maxfreemachine,a]=4444

            x_ = maxfreerack
            y_ = maxfreemachine
            z_ = a
        elif k > 1:  ##################### for jobs with more than 1 and less than 4 gpu request
            for f in range(DEPTH):
                each_rack_free_machine.append(
                    len(np.where(self.resources[minfreerack, f, :] == -1)[0]))  # number of free gpus in rack 0 ... i
                if (len(np.where(self.resources[minfreerack, f, :] == -1)[0]) >= k):
                    each_rack_free_machine_k.append(len(
                        np.where(self.resources[minfreerack, f, :] == -1)[0]))  # number of free k gpus in rack 0 ... i
                    rack = minfreerack
            if not (each_rack_free_machine_k):
                each_rack_free_machine = []
                for f in range(DEPTH):
                    each_rack_free_machine.append(len(
                        np.where(self.resources[maxfreerack, f, :] == -1)[0]))  # number of free gpus in rack 0 ... i
                    if len(np.where(self.resources[maxfreerack, f, :] == -1)[0]) >= k:
                        each_rack_free_machine_k.append(len(np.where(self.resources[maxfreerack, f, :] == -1)[
                                                                0]))  # number of free k gpus in rack 0 ... i
                        rack = maxfreerack
            # print('k>1, each_rack_free_machine',each_rack_free_machine)
            # print('each_rack_free_machine_k',each_rack_free_machine_k)
            # maxfreemachine = each_rack_free_machine.index(max(each_rack_free_machine))# max number of free gpus in each rack
            if each_rack_free_machine_k:
                # print('k>1',each_rack_free_machine)
                # print(each_rack_free_machine_k)
                exactfreemachine = each_rack_free_machine.index(
                    min(each_rack_free_machine_k))  # max number of free gpus in each rack
                # c=(each_rack_free_machine.index(min(each_rack_free_machine_k)))%4
                # print('exactfreemachine', exactfreemachine)
                c = np.where(self.resources[rack, exactfreemachine, :] == -1)[0][0]  # third dimension of a free GPU
                x_ = rack
                y_ = exactfreemachine
                z_ = c
            else:
                maxfreemachine = each_rack_free_machine.index(
                    max(each_rack_free_machine))  # max number of free gpus in each rack
                c = np.where(self.resources[maxfreerack, maxfreemachine, :] == -1)[0][
                    0]  # third dimension of a free GPU
                x_ = maxfreerack
                y_ = maxfreemachine
                z_ = c
                # print('rack, exactfreemachine, c',rack,exactfreemachine,c)
                # # resources[maxfreerack,maxfreemachine,a]=4444


        else:
            for f in range(DEPTH):
                each_rack_free_machine.append(
                    len(np.where(self.resources[minfreerack, f, :] == -1)[0]))  # number of free gpus in rack 0 ... i
            # minfreemachine = each_rack_free_machine.index(max(each_rack_free_machine))# max number of free gpus in each rack
            erfm = np.array(each_rack_free_machine)
            minfreemachine = each_rack_free_machine.index(max(each_rack_free_machine))
            b = np.where(self.resources[minfreerack, minfreemachine, :] == -1)[0][0]  # third dimension of a free GPU
            # print('minfreerack,minfreemachine,b',minfreerack,minfreemachine,b)
            x_ = minfreerack
            y_ = minfreemachine
            z_ = b
            # else:
            #     x_ = int(self.get_avl_gpus()[0][:1])####find the first idle gpu as before(first dim)
            #     y_ = int(self.get_avl_gpus()[1][:1])####find the first idle gpu as before(second dim)
            #     z_ = int(self.get_avl_gpus()[2][:1])####find the first idle gpu as before(third dim)

        ####################################################### First empty GPU will be in resources[maxfreerack,maxfreemachine,a]

        first_gpu_Id = x_ * HEIGHT * DEPTH + y_ * DEPTH + z_  # get the first gpu number dims found randomly in(x,y,z)
        get_nearest = self.topology_parameters.get_gpu_sort_nearest(first_gpu_Id)  # search nearest gpus to it

        x0 = np.where(self.resources == -1)[0][:0]  # create three empty array to put the nearest gpu dims in them
        x1 = np.where(self.resources == -1)[1][:0]
        x2 = np.where(self.resources == -1)[2][
             :0]  # (by appending will be extended to (k,) or np.where(self.resources == -1)[2][:k])

        idleneighbour = 0
        for i in range(len(get_nearest) - 1):  # for each neighbor gpu ID:[3,34,2,5],len=3,i=1,4
            x, y, z = get_kth_dims(get_nearest[i], WIDTH, DEPTH, HEIGHT)  # tell the dims in resources matrice
            if (self.resources[x][y][z] == -1):  # to make sure that the neighbouring GPUs are idle
                idleneighbour = idleneighbour + 1
                if (idleneighbour > k):  # find the rest k-1 gpus from the sorted list
                    # print('get_kth_dims:    ',get_kth_dims)
                    break
                x0 = np.append(x0, x)  # append the  to x0 which contains the k nearest gpus first dims
                x1 = np.append(x1, y)  # append the  to x0 which contains the k nearest gpus second dims
                x2 = np.append(x2, z)  # append the  to x0 which contains the k nearest gpus third dims
        # print('first_gpu_Id:',first_gpu_Id,'get_nearest: ',get_nearest)

        return (x0, x1, x2)  # b


    def get_knn_gpus_shao(self, j):
        if len(self.get_avl_gpus()[0]) >= self.jobqueue[j].gpu_request:
            #print(self.topology_parameters.gpus_string_converter(selected_gpu))
            r = stats.mode(self.get_avl_gpus()[0]).mode[0]
            m = stats.mode(self.get_avl_gpus()[1]).mode[0]
            g = 0
            center = f'r{r}m{m}g{g}'
            sub_G = np.where(self.resources.flat == -1)
            sub_G = np.array(list(self.topology_parameters.G_cluster_only_gpus.nodes()))[sub_G]

            gs = self.topology_parameters.knn(
                self.topology_parameters.G_cluster_only_gpus.subgraph(sub_G), center, self.jobqueue[j].gpu_request - 1)
            selected_gpu = self.topology_parameters.string_gpus_converter(gs + [center])
        else:
            selected_gpu = ([],[],[])
        return selected_gpu

    def check_useful(self):
        num_waiting_jobs = len(self.get_waiting_jobs()[0])
        gpu_request_lst = [j.gpu_request for j in self.jobqueue[self.get_waiting_jobs()]]

        if num_waiting_jobs == 0:
            result = False
        elif all(gpu_request_lst) > len(self.get_avl_gpus()[0]):
            result = False
        else:
            result = True

        return result


    def check_job_can_run(self, j_idx):
        if j_idx < len(self.jobqueue) and len(self.jobqueue) > 0:
            if self.jobqueue[j_idx].gpu_request <= len(self.get_avl_gpus()[0]):
                return True
        return False

    # def return_cluster_data(self):
    #     gpus = self.resources.flatten()
    #     demands = np.zeros((1, 1, self.resources.size+1))
    #     demands[0][0][1:][gpus==-1]=1
    #     demands = demands/float(self.max_gpu_request)
    #     loads = torch.full((1, 1, self.resources.size+1), 1.)
    #     root = self.topology_parameters.root
    #     dynamic = torch.tensor(np.concatenate((loads, demands), axis=1), dtype=torch.float32)[0]
    #     static = torch.from_numpy(self.topology_parameters.static[0])
    #     return static, dynamic, root


    def reward_throughput(self):
        reward = 0
        # for j in self.jobqueue[self.get_running_jobs()]:
        #     reward += j.v_m
        #     if len(j.gpus[0]) != j.gpu_request:
        #         reward += penalty_assigned_gpus(job=j, gpus_per_rack=self.gpus_per_rack, ret_reducer=self..ret_reducer) * self..delay_penalty
        # for j in self.jobqueue[self.get_waiting_jobs()]:
        #     reward -= calc_job_minbatch_speed(job=j, gpus_per_rack=self.gpus_per_rack, ret_reducer=self..ret_reducer, singleormulti='single') * self..hold_penalty
        #
        # #reward += (np.mean(self.resources_utilization) - np.std(self.resources_utilization))
        # for j in self.backlog:
        #     reward -= calc_job_minbatch_speed(job=j, gpus_per_rack=self.gpus_per_rack, ret_reducer=self..ret_reducer, singleormulti='single') * self..dismiss_penalty
        # reward = np.clip(reward, -200, 120)

        reward = 0
        "mao's reward part"
        for j in self.jobqueue[self.get_running_jobs()]:
            reward += (j.v_m)

            "job starting reward"
        for j in self.jobqueue[self.get_waiting_jobs()]:
            reward -= calc_job_minbatch_speed(job=j, gpus_per_rack=self.gpus_per_rack, ret_reducer=self.ret_reducer, singleormulti='single') * self.hold_penalty

        return reward


    def fill_jobqueue(self):
        while len(self.jobqueue) < self.jobqueue_maxlen:
            self.insert_new_job()

    
    def _get_Gpu_available(self) -> np.ndarray:
        '''
        Return a flatten array indicates the availability of gpus.
        1 means available, 0 means unavailable
        '''
        resourses = self.resources.flatten()
        availablity = np.zeros_like(resourses,dtype=int)
        availablity[resourses == -1] = 1
        return availablity
    
    def seed(self, seed=0):
        np.random.seed(seed)