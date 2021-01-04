#!/usr/bin/env bash

eps=5

for data_limit in {200..20000..100}; do

    for i in {1..10} ; do

        if [ ! -f results/icml/q_approx_bisim_lehnert_data_search/lehnert_gridworld_approx_v1_round_to_32_0.50_eps_${data_limit}_dl/run${i}/cluster_purity.dat ]; then

            python -m scripts.bisim.lehnert_gridworld.approx_partition \
            results/icml/q_model_lehnert_data_search/lehnert_gridworld_model_v1_0.0001_elr_adam_opt_20000_steps_${data_limit}_dl/run${i}/model. \
            --epsilon 0.${eps} --save --base-dir results/icml/q_approx_bisim_lehnert_data_search --data-limit ${data_limit}

        fi

    done

done
