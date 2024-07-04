import tensorflow as tf
from tf_agents.policies import py_tf_eager_policy
from Env import py_GpuSelectEnv
from tf_agents.drivers import dynamic_step_driver, py_driver
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies import TFPolicy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer


class Replay_buffer():

    def __init__(self,
                 train_env: py_GpuSelectEnv,
                 collect_data_spec,
                 collect_policy: TFPolicy,
                 maxsize: int,
                 batch_size: int) -> None:
        self.train_env = train_env
        self.collect_data_spec = collect_data_spec
        self.collect_policy = collect_policy
        self.maxsize = maxsize
        self.batch_size = batch_size
        self.env_batch_size = train_env.batch_size
        self.last_timestamp = None
        self.get_replay_buffer()

    def collect(self):
        self.replay_buffer.clear()
        if self.last_timestamp is None:
            self.last_timestamp = self.train_env.reset()
        self.last_timestamp, _ = self._driver.run(self.last_timestamp)

    def get_replay_buffer(self):
        replay_buffer = TFUniformReplayBuffer(
            self.collect_data_spec,
            batch_size=self.env_batch_size,
            device='gpu:0',
            max_length=self.maxsize)

        collect_driver = DynamicStepDriver(self.train_env,
                                           self.collect_policy,
                                           observers=[replay_buffer.add_batch],
                                           num_steps=self.batch_size)
        self._driver = collect_driver
        self.replay_buffer = replay_buffer

    def sample(self):
        '''
        Take all the data in the replay buffer out
        '''
        return self.replay_buffer.gather_all()
