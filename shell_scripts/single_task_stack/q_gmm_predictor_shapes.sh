#!/usr/bin/env bash


for i in {0..9} ; do

    python -m scripts.bisim.puck_stack.learn_predict_q_gmm_prior dataset/2_2x2_dqn_eps_0.1_rs.pickle 2 2 --num-blocks 32 \
    --num-components 1000 --beta0 1 --beta1 0.00001 --beta2 0.00001 --encoder-learning-rate 0.005 \
    --encoder-optimizer adam --num-steps 50000 --gpus 0 --save --base-dir results/icml/q_gmm_predictor_shapes ;

    python -m scripts.bisim.puck_stack.learn_predict_q_gmm_prior dataset/2_3x3_dqn_eps_0.1_rs.pickle 3 2 --num-blocks 32 \
    --num-components 1000 --beta0 1 --beta1 0.00001 --beta2 0.00001 --encoder-learning-rate 0.005 \
    --encoder-optimizer adam --num-steps 50000 --gpus 0 --save --base-dir results/icml/q_gmm_predictor_shapes ;

    python -m scripts.bisim.puck_stack.learn_predict_q_gmm_prior dataset/2_4x4_dqn_eps_0.1_rs.pickle 4 2 --num-blocks 32 \
    --num-components 1000 --beta0 1 --beta1 0.00001 --beta2 0.00001 --encoder-learning-rate 0.005 \
    --encoder-optimizer adam --num-steps 50000 --gpus 0 --save --base-dir results/icml/q_gmm_predictor_shapes ;


    python -m scripts.bisim.puck_stack.learn_predict_q_gmm_prior dataset/3_2x2_dqn_eps_0.1_rs.pickle 2 3 --num-blocks 32 \
    --num-components 1000 --beta0 1 --beta1 0.00001 --beta2 0.00001 --encoder-learning-rate 0.005 \
    --encoder-optimizer adam --num-steps 50000 --gpus 0 --save --base-dir results/icml/q_gmm_predictor_shapes ;

    python -m scripts.bisim.puck_stack.learn_predict_q_gmm_prior dataset/3_3x3_dqn_eps_0.1_rs.pickle 3 3 --num-blocks 32 \
    --num-components 1000 --beta0 1 --beta1 0.00001 --beta2 0.00001 --encoder-learning-rate 0.005 \
    --encoder-optimizer adam --num-steps 50000 --gpus 0 --save --base-dir results/icml/q_gmm_predictor_shapes ;

    python -m scripts.bisim.puck_stack.learn_predict_q_gmm_prior dataset/3_4x4_dqn_eps_0.1_rs.pickle 4 3 --num-blocks 32 \
    --num-components 1000 --beta0 1 --beta1 0.00001 --beta2 0.00001 --encoder-learning-rate 0.005 \
    --encoder-optimizer adam --num-steps 50000 --gpus 0 --save --base-dir results/icml/q_gmm_predictor_shapes ;

done
