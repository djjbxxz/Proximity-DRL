import os
import tensorflow as tf
from tf_agents.utils import common
from keras.activations import relu
from keras.optimizers import Adam
import time
import numpy as np
from base import PPOKLPenaltyAgent
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.trajectories.time_step import TimeStep
from env_components.job import Job
from model import ValueNetwork, ActorNetwork
from environment.env import Env
from PPO.utils import Checkpoint, Logger
from replay_buffer_m import Replay_buffer
from Test_with_baseline.base import JobSelector
from Test_with_baseline.jobselector import *
from Env import observation_struct


def observation_and_action_constraint_splitter(observation):
    return observation.observation, observation.action_mask


def check_data(experience):
    reward = experience.reward[:, 0].numpy()
    action = experience.action[:, 0].numpy()
    ob1 = experience.observation.observation[:, 0, :].numpy()
    mask1 = experience.observation.action_mask[:, 0, :].numpy()
    ob2 = experience.observation.observation[:, 1, :].numpy()
    mask2 = experience.observation.action_mask[:, 1, :].numpy()

    for i in range(reward.shape[0]):
        assert mask1[i, action[i]] == 1, "action mask is not correct"
        if action[i] != 32:  # action is not idle
            assert ob1[i, action[i]] == 1, "gpu is not available"
            assert ob2[i, action[i]] != 1, "gpu is not assigned successfully"
        if ob1[i, -1] == 0:
            r = [0]*32
            r.append(1)
            assert all(mask1[i] == r)
        else:
            # check every gpu, if it is available, the action mask should be 1, otherwise, it should be 0
            for gpu in range(32):
                if mask1[i, gpu] == 1:
                    assert ob1[i, gpu] == 1
                else:
                    assert ob1[i, gpu] in [0, -1]


class PPO_Agent(PPOKLPenaltyAgent):
    def __init__(self,
                 train_env: PyEnvironment,
                 test_env: PyEnvironment,
                 eval_mode=False,
                 eval_interval: int = 3000,
                 eval_episodes: int = 30,
                 save_interval: int = 10000,
                 log_interval: int = 250,
                 batch_size: int = 64,
                 buffer_length: int = 100000,
                 learning_rate: float = 0.00025,
                 num_epochs:int = 10,
                 logdir: str = os.path.join(
                     'runs/ppo', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                 debug_summaries=False,
                 summarize_grads_and_vars=False,
                 name="",
                 **kwargs,
                 ):
        self._learning_rate = learning_rate
        super().__init__(
            time_step_spec=train_env.time_step_spec(),
            action_spec=train_env.action_spec(),
            actor_net=ActorNetwork(None, train_env.action_spec(),observation_and_action_constraint_splitter),
            value_net=ValueNetwork(None, train_env.action_spec(),observation_and_action_constraint_splitter),
            num_epochs=num_epochs,
            optimizer=Adam(learning_rate=self.get_lr),
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=None,
            name=None,
            **kwargs,
        )
        self.eval_mode = eval_mode
        self._train_env = train_env
        self._test_env = test_env
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.save_interval = save_interval
        self.log_interval = log_interval
        self.batch_size = batch_size
        self.log_dir = logdir if name == ""else logdir+f"_{name}"
        self._ckp = Checkpoint(
            dir=self.log_dir,
            save_interval=save_interval,
            policy=self.policy,
            step_counter=self.train_step_counter
        )
        self.load = self._ckp.load
        if not eval_mode:
            self.logger = Logger(logdir=self.log_dir)
            self.best_eval_return = -np.inf

            self.replay_buffer = Replay_buffer(
                train_env,
                self.collect_data_spec,
                self.collect_policy,
                maxsize=buffer_length,
                batch_size=batch_size)

            self.save = self._ckp.save_latest

    def run(self, num_iterations):
        for i in range(num_iterations):

            # Collect a few steps and save to the replay buffer.
            self.replay_buffer.collect()
            # Sample a batch of data from the buffer and update the agent's network.
            experience = self.replay_buffer.sample()
            # check_data(experience)
            self.train(experience)

            step = self.train_step_counter.numpy()
            self.log_hyper_params()

            # if step % self.log_interval == 0:
            # self.logger.log_data(
            #     step, train_loss, experience)

            if step % self.eval_interval == 0:
                r = self.eval()
                self.logger.eval_log(step, *r)

            self._ckp.save_latest()


    def log_hyper_params(self):
        with tf.name_scope('Hyperparams/'):
            tf.compat.v2.summary.scalar(
                name='learning_rate', 
                data=self._learning_rate, 
                step=self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name='entropy_regularization', 
                data=self._entropy_regularization, 
                step=self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name='value_pred_loss_coef', 
                data=self._value_pred_loss_coef, 
                step=self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name='policy_l2_reg', 
                data=self._policy_l2_reg, 
                step=self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name='value_function_l2_reg', 
                data=self._value_function_l2_reg, 
                step=self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name='shared_vars_l2_reg', 
                data=self._shared_vars_l2_reg, 
                step=self.train_step_counter)

    def useGraphOptimization(self):
        if not hasattr(self, 'old_train'):
            self.old_train = self.train
        self.train = common.function(self.old_train)
        self.graph_enabled = True

    def eval(self):
        total_return = []
        lens_ep = []
        make_span = []
        rewards_ep = []
        gpu_utilization = []
        action_type_record = []
        communication_cost = []
        for _ in range(self.eval_episodes):

            time_step = self._test_env.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = self.policy.action(time_step)
                time_step = self._test_env.step(action_step.action)
                episode_return += time_step.reward
            total_return.append(episode_return)
            lens_ep.append(self._test_env.extended_time)
            make_span.append(self._test_env.curr_time)
            rewards_ep.append(episode_return)
            gpu_utilization.append(tf.reduce_mean(
                self._test_env.utilization_rate_record))
            action_type_record.append(
                self._test_env.action_type_count/self._test_env.extended_time)
            communication_cost.append(np.sum([job.communication_cost for job in self._test_env.job_seq]))
        avg_return = np.mean(total_return)
        if not self.eval_mode and avg_return > self.best_eval_return:
            self.best_eval_return = avg_return
            self._ckp.save_best()

        return np.mean(lens_ep), np.mean(rewards_ep), np.mean(gpu_utilization), np.mean(make_span), np.mean(action_type_record, axis=0), np.mean(communication_cost)

    def infer(self,env:Env, job: Job, observation:np.ndarray ,allow_idle: bool) -> list:
        '''infer GPU selection for a job, return a list of GPU id

        Parameters
        ----------
        env : Env
            The environment
        job : Job
            The job to be infered
        allow_idle : bool, optional
            Whether allow idle action when there is enough GPU, by default False

        Returns
        -------
        action : list
            A set of GPU satisfying the requested number of GPUs
        '''
        actions = []
        mask = self.get_valid_mask(
            env, job=job, allow_idle=allow_idle, action_cache=[])
        _input = observation_struct(observation=observation, action_mask=mask)
        for i in range(job.gpu_request):
            time_step = TimeStep(0, 0, 0, observation=_input)
            # time_step = tf.nest.map_structure(
            #     lambda x: tf.convert_to_tensor(x, tf.float64), time_step)
            action = self.policy.action(time_step).action.numpy()
            actions.append(action)
            if action == 32:
                return []  # idle action
            _input.observation[action] = 1
            _input.action_mask[action] = 0
        return actions

    def observe(self, env: Env, job: Job) -> np.ndarray:
        '''
        Observation include:\n
        `[0:32]: action cache`\n
        `[33]: job_len`\n
        `[34]: gpu_request`\n
        `[35:66]: Gpu progress count_down (normalized)`\n

        '''
        job_len, gpu_request = (0, 0) if job is None else (
            job.job_len, job.gpu_request)

        count_down = np.zeros(shape=env.resources.shape)
        for job in env.running_queue:
            count_down[job.gpus] = job.ts_togo

        # Normalization
        count_down = np.clip(count_down/50, 0, 1)
        job_len = job_len/env.max_job_len
        gpu_request = gpu_request / env.max_gpu_request

        # Assemble
        ob = np.zeros((count_down.size*2+2))
        ob[-count_down.size:] = count_down.flatten()
        ob[count_down.size] = job_len
        ob[count_down.size+1] = gpu_request
        return ob

    def get_valid_mask(self, env: Env, job: Job, allow_idle: bool = False, action_cache: list[int] = []) -> np.ndarray:
        '''
        Return a mask indicates which action is valid, 1 means valid, 0 means invalid
        '''
        available_gpus = env._get_Gpu_available()
        if job is None or job.gpu_request > available_gpus.sum():
            mask = np.zeros(shape=33, dtype=int)
            mask[-1] = 1
            return mask

        mask = np.ones(shape=32, dtype=int)
        mask[action_cache] = 0
        np.bitwise_and(mask, available_gpus, out=mask)
        mask = np.append(mask, 1 if allow_idle else 0)
        return mask

    def get_lr(self):
        return self._learning_rate