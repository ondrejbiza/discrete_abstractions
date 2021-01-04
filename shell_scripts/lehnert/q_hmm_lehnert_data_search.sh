#!/usr/bin/env bash


for data_limit in {200..20000..100}; do

    for i in {0..9} ; do

        python -m scripts.bisim.lehnert_gridworld.learn_lehnert_gridworld_hmm \
        dataset/lehnert_gridworld_100k_dqn_eps_0.5.pickle 30 3 --data-limit ${data_limit} \
        --num-blocks 32 --num-components 6 --learning-rate 0.01 --model-learning-rate 0.01 --optimizer adam --beta0 1 \
        --beta1 0.0001 --beta2 0.0001 --mixtures-mu-init-sd 0.01 --num-steps 20000 --gpus "" --save \
        --base-dir results/icml/q_hmm_lehnert_data_search

    done

done
