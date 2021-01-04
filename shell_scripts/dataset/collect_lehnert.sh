#!/usr/bin/env bash

if [ ! -d "dataset" ]; then
    mkdir -p "dataset"
fi

python -m scripts.solve.lehnert_gridworld.dqn 30 3 --max-episodes 100000 --gpus "" \
--save-exp-path dataset/lehnert_gridworld_100k_dqn_eps_0.5.pickle --save-exp-num 100000 \
--save-q-values --save-exp-after-training --save-exp-eps 0.5