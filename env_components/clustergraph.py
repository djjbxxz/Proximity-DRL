class ClusterGraph:

    def __init__(self, pa):
        self.num_gpus_per_machine = num_gpus_per_machine
        self.num_machines_per_rack = num_machines_per_rack
        self.num_racks_per_cluster = num_racks_per_cluster
        self.nvLink = 40
        self.infiniBand = 4
        self.peer_to_peer = 2

if __name__ == '__main__':
    from env_components.parameters import Parameters
    num_gpus_per_machine = 4
    num_machines_per_rack = 4
    num_racks_per_cluster = 2
    max_gpu_request = 8
    max_job_len = 20
    jobqueue_maxlen = 20
    max_backlog_len = 0
    new_job_rate = 1
    target_num_job_done = 100
    delay_penalty = 1
    hold_penalty = 2
    dismiss_penalty = 1
    gpu_request_skew = -4
    job_len_skew = -4

    pa = Parameters(num_gpus_per_machine=num_gpus_per_machine,
                    num_racks_per_cluster=num_racks_per_cluster,
                    num_machines_per_rack=num_machines_per_rack,
                    max_gpu_request=max_gpu_request,
                    max_job_len=max_job_len,
                    jobqueue_maxlen=jobqueue_maxlen,
                    max_backlog_len=max_backlog_len,
                    new_job_rate=new_job_rate,
                    hold_penalty=hold_penalty,
                    delay_penalty=delay_penalty,
                    dismiss_penalty=dismiss_penalty,
                    target_num_job_done=target_num_job_done,
                    gpu_request_skew=gpu_request_skew,
                    job_len_skew=job_len_skew,
                    max_num_timesteps=400000)

    G = ClusterGraph(pa)
    print(G.nvLink)
