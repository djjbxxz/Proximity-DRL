import numpy as np


class Gpu():
    """The class to hold the gpu object.

        Attributes
        ----------
        compute_power : float
            The computing power of the current gpu

        Methods
        -------

        """

    def __init__(self, gpu_id: int, rack_id: int, host_id: int, remaining_time: int, status: int = 0):
        # compute_power: float=1.0):
        """Constructor of the gpu

        Parameters
        ----------
        compute_power : float, optional, default: 1.0
            The computing power of the current gpu, it is not used when
            all gpus have the same computing power in the environment
        """
        self.gpu_id = gpu_id
        self.rack_id = rack_id
        self.host_id = host_id
        self.remaining_time = remaining_time
        self.status = status
        # self.compute_power = compute_power
