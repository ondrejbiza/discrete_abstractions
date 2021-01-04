#!/usr/bin/env bash


for data_limit in {200..20000..100}; do

    for i in {0..9} ; do

        python -m scripts.model.learn_lehnert_gridworld \
        dataset/lehnert_gridworld_100k_dqn_eps_0.5.pickle 30 3 --data-limit ${data_limit} \
        --learning-rate 0.0001 --optimizer adam --num-steps 20000 --gpus "" --save --save-model \
        --base-dir results/icml/q_model_lehnert_data_search

    done

done
