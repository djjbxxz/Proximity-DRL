import random
import math

import numpy as np
from math import ceil
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from env_components.topology import NetworkTopology
__all__ = ['observe_rltaps', 'plot_observe', 'num_racks_cal', 'calc_speed_single', 'get_reduction_asym',
           'calc_job_minbatch_speed', 'penalty_assigned_gpus']


def observe_rltaps(env):
    height = env.pa.max_job_len
    width = env.resources.size + env.jobqueue_maxlen * env.pa.max_gpu_request + int(
        ceil(env.pa.max_backlog_len) / float(env.pa.max_job_len))
    image = np.zeros((height, width))
    image[:, 0: env.resources.size] = get_cluster_canvas(env.resources, env.pa.max_job_len, env.jobqueue)[:, :]
    pt = env.resources.size
    for j in env.jobqueue[env.get_waiting_jobs()]:
        image[: j.job_len, pt: pt + j.gpu_request] = 1
        pt += env.pa.max_gpu_request
    image[: len(env.backlog), width - 1] = 1
    return np.expand_dims(image, axis=2)




def plot_observe(image):
    plt.imshow(image)
    plt.show()


def get_cluster_canvas(cluster, max_job_len, jobqueue):
    image = np.zeros((max_job_len, cluster.size))
    gpus = cluster.reshape(1, cluster.size)
    used = np.where(gpus != -1)

    for i, j in zip(used[0], used[1]):
        # print(f'{get_j_idx_by_id(gpus[i,j], jobqueue)} {gpus[i,j]}')
        j_idx = get_j_idx_by_id(gpus[i, j], jobqueue)[0][0]
        image[0:jobqueue[j_idx].job_len - jobqueue[j_idx].progress, j] = 1
    # plt.imshow(image)
    # plt.show()
    return image


def get_j_idx_by_id(id, jobqueue):
    def getter(j):
        return j.job_id

    vfunc = np.vectorize(getter, otypes=[int])
    idx = np.where(vfunc(jobqueue) == id)
    # for idx in range(len(jobqueue)):
    #     if jobqueue[idx].job_id == id:
    #         return idx
    return idx


def num_racks_cal(gpu_request, gpus_per_rack):
    return 1 if gpu_request <= (gpus_per_rack) else math.ceil(gpu_request / gpus_per_rack)


def num_racks_assigned_cal(gpu_assigned):
    return len(np.unique(gpu_assigned[1]))


def calc_speed_single(gradsize, gpu_request, d_mj_tflop, tt_m, numracks, ret_reducer):
    rt_m = get_reduction_asym(
        gradsize, gpu_request, numracks, ret_reducer)
    minbatch_speed = d_mj_tflop * 1.0 / (tt_m + 1.0 * rt_m)
    minbatch_speed *= gpu_request
    return rt_m, minbatch_speed


def calc_speed_multi(job, gradsize, g_assigned, d_m, tt_m, numracks,
                     scale, ret_reducer=1.):
    # job.scale is computed when self.findlimitingbws() is called.
    assert ~np.isnan(scale)
    reduction_time_m = get_reduction_asym(
        gradsize, g_assigned, numracks, ret_reducer)
    # Todo-cbb. Add in reduction time increaser for number of racks.
    #minbatch_speed = d_m * 1.0 / (tt_m + 1.0 * scale * rt_m+compute_GPU_distances(job))
    minbatch_speed = d_m * 1.0 / (tt_m + 1.0 * scale * reduction_time_m)
    minbatch_speed *= g_assigned
    return reduction_time_m, minbatch_speed


def compute_GPU_distances(job):
    topology_parameters = NetworkTopology()
    switchtime = 0
    jobGpusIDlist = []
    WIDTH = 4  # (machine per rack)
    DEPTH = 4  # (gpu per machine)
    HEIGHT = 4
    distancefromothers = 0
    # print('job.gpus[0]): ',job.gpus)
    if job.gpus != []:
        for cordin in range(len(job.gpus[0])):  # find the selected GPUs' IDs
            x_ = job.gpus[0][cordin - 1]
            y_ = job.gpus[1][cordin - 1]
            z_ = job.gpus[2][cordin - 1]
            gpu_Id = x_ * HEIGHT * DEPTH + y_ * DEPTH + z_  # get the first gpu number dims found randomly in(x,y,z)
            # print('gpu_Id:',gpu_Id,x_,y_,z_)
            jobGpusIDlist.append(gpu_Id)

        # for gp in jobGpusIDlist:
        # if (len(jobGpusIDlist) > 1):
        #     for i in range(len(jobGpusIDlist)):
        #         distancefromothers += topology_parameters.get_gpu_distance_gpu(jobGpusIDlist[i], jobGpusIDlist[0])



        if (len(jobGpusIDlist) > 1):
            for gpu in jobGpusIDlist:
                distancefromothers += topology_parameters.get_gpu_distance_gpu(gpu, jobGpusIDlist[0])

        return distancefromothers

def get_reduction_asym(gradsize, gpu_request, numracks, ret_reducer):
    '''
    Returns the reduction time for a given gradient size ,num of racks, and number of GPUs.
    '''
    def hundredmb(num_gpus):
        """
        Returns reduction time for 100 MB Gradient size.
        :param num_gpus: Any number of GPUs
        :return:
        """
        return 28.29 * np.float_power(num_gpus, 0.5)

    def fivehundredmb(num_gpus):
        """
        Returns reduction time for 100 MB Gradient size.
        :param num_gpus: Any number of GPUs
        :return:
        """
        return 139.77 * np.float_power(num_gpus, 0.5)

    if gradsize < 0:
        raise ValueError("Gradsize cannot be less than 0 MB")

    elif gradsize > 500:  # cbb. do extrapolation.
        x = np.linspace(100, 500, num=2, endpoint=True)
        y = np.array([hundredmb(gpu_request), fivehundredmb(gpu_request)],
                     dtype='float')
        f = interp1d(x, y, kind='linear', fill_value='extrapolate')
        ret = f(gradsize)
    else:
        x = np.array([0, 100, 500], dtype='float')
        y = np.array([0, hundredmb(gpu_request), fivehundredmb(gpu_request)],
                     dtype='float')
        f = interp1d(x, y)
        ret = f(gradsize)

    ret = ret * 0.001  # convert from ms to s
    ret = ret * ret_reducer
    ret = ret * (1 + 0.2 * (numracks - 1))
    return ret


def calc_job_minbatch_speed(job, gpus_per_rack, ret_reducer, singleormulti='multi',
                            outdetails=False):
    if len(job.gpus) == 0:
        # job does not have any assigned gpus yet. So we
        # are doing a theoretical speed calculation based on number
        # of gpus it's requested numgpus = job.res_vec[0]
        numracks = num_racks_cal(job.gpu_request, gpus_per_rack)
    else:
        numracks = num_racks_assigned_cal(job.gpus)
    if singleormulti == 'single':  # pretend speed if job running alone.
        # this formulation is used for when a job is not running but we
        # want to calculated the job's average speed when running alone,
        # i.e. to calcualte the opportunity cost of not running the job
        # and keeping the job in the either the queue or the backlog.
        rt_m, minbatch_speed = calc_speed_single(
            job.gradsize, job.gpu_request, job.d_m, job.tt_m, numracks,
            ret_reducer)
        job.rt_m = rt_m
    elif singleormulti == 'multi':  # actual speed
        # job.scale is computed when self.findlimitingbws() is called.
        assert ~np.isnan(job.scale)
        rt_m, minbatch_speed = calc_speed_multi(job,
            job.gradsize, len(job.gpus[0]), job.d_m, job.tt_m, numracks, job.scale,
            ret_reducer)
        job.rt_m = rt_m
    else:
        raise ValueError("Input parameter 'singleormulti' must be either "
                         "'single or 'multi'")
    assert ~np.isnan(minbatch_speed)
    if not outdetails:
        return minbatch_speed
    else:
        return minbatch_speed, rt_m


def penalty_assigned_gpus(job, gpus_per_rack, ret_reducer):
    gpu_wrong = 0
    if len(job.gpus[0]) < job.gpu_request:
        gpu_wrong = job.gpu_request - len(job.gpus[0])
    elif len(job.gpus[0]) > job.gpu_request:
        gpu_wrong = len(job.gpus[0]) - job.gpu_request
    else:
        return 0

    if len(job.gpus[0]) < job.gpu_request:
        numracks = num_racks_cal(job.gpu_request, gpus_per_rack)
    else:
        numracks = num_racks_assigned_cal(job.gpus)
    _, minbatch_speed = calc_speed_multi(
        job.gradsize, gpu_wrong, job.d_m, job.tt_m, numracks, job.scale,
        ret_reducer)
    # rt_m, minbatch_speed = calc_speed_multi(
    #     job.gradsize, gpu_wrong, job.d_m, job.tt_m, numracks, job.scale,
    #     ret_reducer)
    # assert ~np.isnan(minbatch_speed)
    # if not outdetails:
    #     return minbatch_speed
    # else:
    #     return minbatch_speed, rt_m
    return minbatch_speed


def choose_gpu_fn(gpus, rack_id, host_id, gpu_id):
    indices_obj = rack_id == gpus.rack_id and host_id == gpus.host_id and gpu_id == gpus.gpu_id
    indices_bool = np.array(indices_obj, dtype=bool)
    return indices_bool


def gpu_assign_time_status_fn(selected_gpus, job):
    selected_gpus.remaining_time = job.job_len - job.progress
    selected_gpus.status = 1

class DotDict(dict):
    """
    dot.notation access to dictionary attributes
    for example:
    a = {'a':1, 'b':2}
    a = DotDict(a)
    a.a== a['a']
    """
    def __getattr__(self, attr):
        return self.get(attr)
 