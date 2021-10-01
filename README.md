# Curiosity-driven Exploration by Self-supervised Prediction

## 1. Setup
####  Requirements

------------

- python3.6
- gym
- [OpenCV Python](https://pypi.python.org/pypi/opencv-python)
- [PyTorch](http://pytorch.org/)
- [tensorboardX](https://github.com/lanpa/tensorboardX)
- [atari roms](http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html) 

## 2. How to Train
Modify the parameters in `config.conf`.
```
python train.py
```
In a separate terminal:
```
tensorboard --logdir runs
```

## 3. How to Eval
```
python eval.py
```

References
----------

[1] [Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf)    
[2] [Efficient Parallel Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1705.04862)  
[3] [Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/abs/1705.05363)   
[4] [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)  
[5] [Large-Scale Study of Curiosity-Driven Learning](https://arxiv.org/abs/1808.04355)  
  
