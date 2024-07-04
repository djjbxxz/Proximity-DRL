import yaml
import threading
import logging
import os
import time
import numpy as np
import tensorflow as tf
import shutil
from environment.gpu_select_env import Action_type


def isDebug():
    import sys
    return True if sys.gettrace() else False


class Logger():
    def __init__(self, logdir):
        self.logdir = logdir
        self.tf_writer = None
        self.start_time = time.time()
        self.n_eps = 0
        self.is_tb_enabled = not isDebug()  # not isDebug()

        if self.is_tb_enabled:
            if not os.path.exists(self.logdir):
                os.makedirs(self.logdir)
            # create_noop_writer
            self.writer = tf.summary.create_file_writer(self.logdir)
            self.writer.set_as_default()

            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s %(message)s',
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler(self.logdir + '/logger.log'),
                ],
                datefmt='%Y/%m/%d-%I:%M:%S'
            )
            # self.save_config(self.logdir)

    def log_str(self, content):
        if self.is_tb_enabled:
            logging.info(content)
        else:
            print(f"{self.get_current_time_str()}  {content}")

    def get_current_time_str(self) -> str:
        return time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time()))

    def add_scalar(self, tag, scalar_value, global_step):
        if self.is_tb_enabled:
            tf.summary.scalar(name=tag, data=scalar_value, step=global_step)

    def eval_log(self, step, episode_length, episode_reward, gpu_utilization, make_span, action_type_record,communication_cost):
        with tf.name_scope('Eval'):
            self.add_scalar(tag='avg_episode_length',
                            scalar_value=episode_length, global_step=step)
            self.add_scalar(tag='avg_episode_reward',
                            scalar_value=episode_reward, global_step=step)
            self.add_scalar(tag='avg_gpu_utilization',
                            scalar_value=gpu_utilization, global_step=step)
            self.add_scalar(tag='avg_make_span',
                            scalar_value=make_span, global_step=step)
            self.add_scalar(tag='avg_communication_cost',
                            scalar_value=communication_cost, global_step=step)
        with tf.name_scope('Action_type_in_eval'):
            for i in range(len(action_type_record)):
                self.add_scalar(
                    tag=f"{Action_type(i).name}", scalar_value=action_type_record[i], global_step=step)
        self.log_str(
            f"step {step} | avg_episode_length={episode_length:.3f} | avg_episode_reward={episode_reward:.3f} | avg_gpu_utilization={gpu_utilization:.3f} | avg_make_span={make_span:.3f}")

    def save_config(self, file_to_copy: list[str]):
        '''
        Save config files to logdir
        file_to_copy: list[str]
            list of file path to copy to logdir
            For example: ['PPO/config.yaml', 'environment/config.yaml']
        '''
        path = self.logdir
        self.target_file_list = []
        for file in file_to_copy:
            target_file = os.path.join(path, file.replace('/', '_'))
            shutil.copy(file, target_file)
            self.target_file_list.append(target_file)


def reduce_gpu_memory_usage():
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print('Memory efficiency enabled')
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass


class Checkpoint:
    def __init__(self, dir: str, save_interval,
                 policy,
                 step_counter
                 ) -> None:
        self._path = dir
        self.train_step_counter = step_counter
        self._save_interval = save_interval
        self._checkpoint = tf.train.Checkpoint(
            policy=policy,
            step_counter=step_counter,
        )
        self._latest_manager = tf.train.CheckpointManager(
            self._checkpoint, directory=os.path.join(dir, 'latest'), max_to_keep=None)
        self._best_manager = tf.train.CheckpointManager(
            self._checkpoint, directory=os.path.join(dir, 'best'), max_to_keep=None)

    def load(self, path, catergory: str = 'latest', index=-1):
        '''
        Load the checkpoint from the path.
        catergory: str, 'latest' or 'best'
        index: int, the index of the checkpoint to load, default to using the latest checkpoint
        '''
        splited_path:list[str] = path.split('/')
        if 'ckpt-' not in splited_path[-1]:

            if 'latest' in splited_path[-1]:
                catergory = 'latest'
                splited_path.pop(-1)
            elif 'best' in splited_path[-1]:
                catergory = 'best'
                splited_path.pop(-1)


            _manager = tf.train.CheckpointManager(
                self._checkpoint, directory=os.path.join(*splited_path,catergory), max_to_keep=None)

            load_path = _manager.checkpoints[index]
        else:
            load_path = path
        status = self._checkpoint.restore(load_path)
        status.assert_consumed()
        print(f'Using checkpointer step: {self.train_step_counter.numpy()}, from {load_path}')

    def save_best(self):
        self._best_manager.save()
        pass

    def save_latest(self):
        if self.train_step_counter % self._save_interval == 0:
            self._latest_manager.save()
        pass


class FileWatcher:
    '''
    Watch a file for changes. The watcher will run in a separate thread.
    '''

    def __init__(self, filename, _func: callable, watch_interval=1):
        self.filename = filename
        self.interval = watch_interval
        self.last_modified = os.path.getmtime(filename)
        self._func: callable = _func
        self.should_stop = False

    def _watch_loop(self):
        while not self.should_stop:
            current_modified = os.path.getmtime(self.filename)
            if current_modified != self.last_modified:
                self.last_modified = current_modified
            self._func()
            time.sleep(self.interval)

    def start(self):
        self.should_stop = False
        self._thread = threading.Thread(target=self._watch_loop)
        self._thread.start()

    def stop(self):
        self.should_stop = True
        self._thread.join()


class HyperParamsWatcher:
    def __init__(self, config_file_to_watch: str, params_mapping_dict: dict, agent, watch_interval=1):
        '''
        Monitor a file for params changes. The watcher will run in a separate thread.
        params_mapping_dict: dict, mapping name from config file to agent's variables names. eg: {"learning_rate": '_learning_rate'}
        '''
        self.watcher = FileWatcher(
            config_file_to_watch, self.onchanged, watch_interval)
        self.params_mapping_dict = params_mapping_dict
        self.agent = agent
        print(f"HyperParamsWatcher: watching at {config_file_to_watch}")

    def onchanged(self):
        '''
        Called when the file is changed.
        '''
        with open(self.watcher.filename) as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        updates = self.updateparams(config)
        return updates

    def updateparams(self, yaml_config: dict):
        '''
        Update the params of the agent.
        '''
        change = []
        any_param_changed = False
        for new_param_key, new_param_value in yaml_config.items():
            if new_param_key not in self.params_mapping_dict:
                continue
            old_param_value = getattr(
                self.agent, self.params_mapping_dict[new_param_key])
            if old_param_value != new_param_value:
                setattr(
                    self.agent, self.params_mapping_dict[new_param_key], new_param_value)
                any_param_changed = True
                print(
                    f"Update {new_param_key} from {old_param_value} to {new_param_value}")
            change.append(
                (self.params_mapping_dict[new_param_key], (old_param_value, new_param_value)))
        if any_param_changed:
            self.agent.useGraphOptimization()
        return change

    def start(self):
        self.watcher.start()

    def stop(self):
        self.watcher.stop()
