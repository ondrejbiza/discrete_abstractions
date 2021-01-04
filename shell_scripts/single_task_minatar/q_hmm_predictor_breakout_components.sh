#!/usr/bin/env bash


dir_path="results/icml/q_hmm_predictor_breakout_components"

beta=0.00005
learning_rate=0.001
num_components_list=(50 100 150 200 250 300 350 400 450 500)
num_runs=10

if [[ ! -d "$dir_path" ]]; then
    mkdir -p "$dir_path"
fi

for (( i=1; i<=$num_runs; i++ )); do
    for num_components in "${num_components_list[@]}"; do
        python -m scripts.bisim.minatar.learn_predict_q_hmm_prior dataset/breakout.pickle breakout \
        --num-components ${num_components} --beta0 1 --beta1 ${beta} --beta2 ${beta} \
        --encoder-learning-rate ${learning_rate} \
        --encoder-optimizer adam --num-steps 50000 --gpus 0 --save --base-dir ${dir_path} --validation-freq 5000 \
        --fix-prior-training --model-learning-rate 0.01 --post-train-hmm --post-train-hmm-steps 50000
    done
done
