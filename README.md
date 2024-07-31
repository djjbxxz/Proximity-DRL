# Summary

This project is to develop a scheduling policy for large GPU cluster, aiming at reducing communication cost for large scale distributed learning (For example, like training LLM) please refer the publication for more detail. [GPU Job Scheduler with Deep Reinforcement Learning](https://caiac.pubpub.org/pub/u6j20p40)

# Installation

This package is tested with Python 3.9.16. It may work with higher versions but has not been officially tested.

```bash
pip install -r requirements.txt
```

# Usage

Refer to the example in [test.py](test.py).

# Repository

This repo includes a non-preemptive environment and a Proximity-DRL model.

# Citation

```bibtex
@inproceedings{Deng2024,
  author = {Junjie Deng and Aijun An and Hajer Ayadi and Yiming Shao and Hossein Pourmodheji and Hao Zhou and Michael Feiman},
  title = {GPU Job Scheduler with Deep Reinforcement Learning},
  booktitle = {Proceedings of the 37th Canadian Conference on Artificial Intelligence},
  year = {2024},
  publisher = {Canadian Artificial Intelligence Association},
  url = {https://caiac.pubpub.org/pub/u6j20p40},
}
```

Deng, J., An, A., Ayadi, H., Shao, Y., Pourmodheji, H., Zhou, H., & Feiman, M. (2024). [GPU Job Scheduler with Deep Reinforcement Learning](https://caiac.pubpub.org/pub/u6j20p40). Proceedings of the Canadian Conference on Artificial Intelligence. Retrieved from 
