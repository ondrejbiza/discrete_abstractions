### Discrete abstractions

Source code for Learning Discrete State Abstractions With Deep Variational Inference. Biza, Platt, van de Meent, Wong. AABI'21.
https://arxiv.org/abs/2003.04300

## Setup

Optional: set up a virtual environment.

```
virtualenv --system-site-packages -p python3 ~/venv
source ~/venv/bin/activate
```

Install Python 3.6 and all dependencies:

```
pip install -r requirements.txt
cd envs/MinAtar
pip install . --no-deps
```

Install  CUDA 10.1, cuDNN 7.6.2 and tensorflow-gpu 1.14.0. You will also need pytorch 1.3.0 to train a DQN on the 
MinAtar environments (we use the authors' code; hence, the mismatch between deep learning frameworks).

## Usage

### Training a DQN

The *scripts/solve* folder contains scripts for training a deep Q-network on all our environments.
For our hyper-parameter settings, check *shell_scripts/dataset*.

Example:
```
# train a dqn to stack 2 pucks
python -m scripts.solve.continuous_puck_stack_n.dqn 4 4 2 \
    --height-noise-std 0.05 --num-filters 32 64 128 256 \
    --filter-sizes 4 4 4 4 --strides 2 2 2 2 --hiddens 512 \
    --target-size 64 --batch-size 32 --learning-rate 0.00005 \
    --buffer-size 100000 --max-time-steps 100000 \
    --exploration-fraction 0.8 --max-episodes 1000000 \
    --gpu-memory-fraction 0.6 --fix-dones --gpus 0
```

### Learning a state abstraction

You first need to collect a dataset of transitions. Again, you can check *shell_scripts/dataset* for collection scripts.
The *scripts/bisim* folder contain all our abstraction scripts. We include code to reproduce all our
tables below.

Example:
```
# learn an abstraction with an HMM prior on the 2 pucks stacking task
beta=0.000001
lre=0.001
lrm=0.01

python -m scripts.bisim.puck_stack.learn_predict_q_hmm_prior dataset/2_4x4_dqn_eps_0.1_fd.pickle 4 2 \
    --num-blocks 32 --num-components 1000 --beta0 1 --beta1 $beta --beta2 $beta --new-dones \
    --encoder-learning-rate $lre --model-learning-rate $lrm --encoder-optimizer adam --num-steps 50000 \
    --fix-prior-training --post-train-hmm --post-train-hmm-steps 50000 --gpus 0 --save \
    --base-dir results/icml/q_hmm_predictor_pucks
```

## Reproducing experiments

### Collect datasets

First, we need to collect datasets for training our abstract models.

Column World:

```
./shell_scripts/dataset/collect_lehnert.sh
```

Puck and Shapes Worlds, Buildings:

```
./shell_scripts/dataset/collect_pucks.sh
./shell_scripts/dataset/collect_shapes.sh
./shell_scripts/dataset/collect_multi_task_shapes.sh
./shell_scripts/dataset/collect_multi_task_buildings.sh
```

MinAtar (use pytorch):

```
./shell_scripts/dataset/collect_minatar.sh
```

Note that each dataset can take up to 10GB.

### Comparing against approximate bisimulation baseline (Figure 4)

```
# ours
./shell_scripts/lehnert/q_hmm_lehnert_data_search.sh
# baseline (first train models, then use them to find approx. bisim.)
./shell_scripts/lehnert/q_model_lehnert_data_search.sh
./shell_scripts/lehnert/q_approx_bisim_lehnert_data_search.sh
```

### Comparing GMM and HMM (Table 1)

Pucks:

```
./shell_scripts/single_task_stack/q_gmm_predictor_pucks.sh
./shell_scripts/single_task_stack/q_hmm_predictor_pucks.sh
```

Various shapes:

```
./shell_scripts/single_task_stack/q_gmm_predictor_shapes.sh
./shell_scripts/single_task_stack/q_hmm_predictor_shapes.sh
```

### Shapes World transfer experiment (Table 2)

```
./shell_scripts/multi_task_shapes/q_hmm_multi_task_pucks_1_shapes.sh
./shell_scripts/multi_task_shapes/q_hmm_multi_task_pucks_2_shapes.sh
```

### Buildings transfer experiment (Table 4)

```
./shell_scripts/multi_task_buildings/q_hmm_predictor_buildings_1.sh
./shell_scripts/multi_task_buildings/q_hmm_predictor_buildings_2.sh
```

### MinAtar (Table 3, Figure 7)

```
# games
./shell_scripts/single_task_minatar/q_hmm_predictor_asterix.sh
./shell_scripts/single_task_minatar/q_hmm_predictor_breakout.sh
./shell_scripts/single_task_minatar/q_hmm_predictor_freeway.sh
./shell_scripts/single_task_minatar/q_hmm_predictor_space_invaders.sh

# num. components search on breakout
./shell_scripts/single_task_minatar/q_hmm_predictor_breakout_components.sh
```

### Analyzing results

The results get saved in *results/icml*. To aggregate them over all runs, you can use:
```
experiment='experiment name'
task_index='task index'

python -m scripts.analyze.analyze_gridsearch_in_dirs_v2 \ 
results/icml/${experiment} rewards_task_${task_index}_inactive.dat -a -m
```
