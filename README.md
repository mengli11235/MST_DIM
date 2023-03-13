# Unsupervised Representation Learning on Partially Observable Atari Games

This repo is based on the code from the benchmark and techniques introduced in the paper [Unsupervised State Representation Learning in Atari](https://arxiv.org/abs/1906.08226). Please visit https://github.com/mila-iqia/atari-representation-learning for detailed instructions on the benchmark.

To run the script:

```bash
python run_probe.py
```

An example of setting the environment, pretrain with masked images and masking ratio 0.8, seed 2:

```bash
run_probe.py --env-name VideoPinballNoFrameskip-v4 --pretrain-masks --mask-ratio 0.8 --seed 2
```