#!/usr/bin/env bash


beta=0.000001

for i in {0..9} ; do

    python -m scripts.bisim.puck_stack.learn_predict_q_hmm_prior_multi_task_buildings \
    dataset/4x4_b_dqn_2B_eps_0.1.pickle 4 2 \
    --active-indices 0 --eval-indices 1 2 3 4 5 --dones-index 0 \
     --num-blocks 32 --num-components 1000 --encoder-learning-rate 0.001 --model-learning-rate 0.01 \
    --beta0 1 --beta1 $beta --beta2 $beta --goal-rewards-threshold 0.99 --softmax-policy --softmax-policy-temp 0.01 \
    --encoder-optimizer adam --num-steps 50000 --fix-prior-training --post-train-hmm --post-train-hmm-steps 50000 \
    --gpus 0 --save --base-dir results/icml/4x4_b_dqn_2B_eps_0.1

    python -m scripts.bisim.puck_stack.learn_predict_q_hmm_prior_multi_task_buildings \
    dataset/4x4_b_dqn_3B_eps_0.1.pickle 4 2 \
    --active-indices 1 --eval-indices 0 2 3 4 5 --dones-index 1 \
    --num-blocks 32 --num-components 1000 --encoder-learning-rate 0.001 --model-learning-rate 0.01 \
    --beta0 1 --beta1 $beta --beta2 $beta --goal-rewards-threshold 0.99 --softmax-policy --softmax-policy-temp 0.01 \
    --encoder-optimizer adam --num-steps 50000 --fix-prior-training --post-train-hmm --post-train-hmm-steps 50000 \
    --gpus 0 --save --base-dir results/icml/4x4_b_dqn_3B_eps_0.1

    python -m scripts.bisim.puck_stack.learn_predict_q_hmm_prior_multi_task_buildings \
    dataset/4x4_b_dqn_3_top_eps_0.1.pickle 4 2 \
    --active-indices 2 --eval-indices 0 1 3 4 5 --dones-index 2 \
    --num-blocks 32 --num-components 1000 --encoder-learning-rate 0.001 --model-learning-rate 0.01 \
    --beta0 1 --beta1 $beta --beta2 $beta --goal-rewards-threshold 0.99 --softmax-policy --softmax-policy-temp 0.01 \
    --encoder-optimizer adam --num-steps 50000 --fix-prior-training --post-train-hmm --post-train-hmm-steps 50000 \
    --gpus 0 --save --base-dir results/icml/4x4_b_dqn_3_top_eps_0.1

    python -m scripts.bisim.puck_stack.learn_predict_q_hmm_prior_multi_task_buildings \
    dataset/4x4_b_dqn_3_left_eps_0.1.pickle 4 2 \
    --active-indices 4 --eval-indices 0 1 2 3 5 --dones-index 4 \
    --num-blocks 32 --num-components 1000 --encoder-learning-rate 0.001 --model-learning-rate 0.01 \
    --beta0 1 --beta1 $beta --beta2 $beta --goal-rewards-threshold 0.99 --softmax-policy --softmax-policy-temp 0.01 \
    --encoder-optimizer adam --num-steps 50000 --fix-prior-training --post-train-hmm --post-train-hmm-steps 50000 \
    --gpus 0 --save --base-dir results/icml/4x4_b_dqn_3_left_eps_0.1

done
