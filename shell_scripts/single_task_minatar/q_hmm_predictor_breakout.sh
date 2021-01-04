#!/usr/bin/env bash


dir_path="results/icml/q_hmm_predictor_breakout"

num_runs=10

if [[ ! -d "$dir_path" ]]; then
    mkdir -p "$dir_path"
fi

for (( i=1; i<=$num_runs; i++ )); do

    python -m scripts.bisim.minatar.learn_predict_q_hmm_prior dataset/breakout.pickle breakout \
    --num-components 500 --beta0 1 --beta1 0.00005 --beta2 0.00005 \
    --encoder-learning-rate 0.001 --encoder-optimizer adam --num-steps 50000 \
    --gpus 0 --save --save-model --validation-freq 5000 --fix-prior-training \
    --model-learning-rate 0.01 --post-train-hmm --post-train-hmm-steps 50000 \
    --q-scaling-factor 10.0 --eval-episodes 100 --softmax-policy --softmax-policy-temp 0.0005 \
    --base-dir ${dir_path}
done
