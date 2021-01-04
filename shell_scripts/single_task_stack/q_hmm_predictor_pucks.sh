#!/usr/bin/env bash


beta=0.000001
base_dir=results/icml/q_hmm_predictor_pucks
lre=0.001
lrm=0.01

for i in {0..9} ; do

    python -m scripts.bisim.puck_stack.learn_predict_q_hmm_prior dataset/2_2x2_dqn_eps_0.1_fd.pickle 2 2 \
    --num-blocks 32 --num-components 1000 --beta0 1 --beta1 $beta --beta2 $beta --new-dones \
    --encoder-learning-rate $lre --model-learning-rate $lrm --encoder-optimizer adam --num-steps 50000 \
    --fix-prior-training --post-train-hmm --post-train-hmm-steps 50000 --gpus 0 --save --base-dir $base_dir

    python -m scripts.bisim.puck_stack.learn_predict_q_hmm_prior dataset/2_3x3_dqn_eps_0.1_fd.pickle 3 2 \
    --num-blocks 32 --num-components 1000 --beta0 1 --beta1 $beta --beta2 $beta --new-dones \
    --encoder-learning-rate $lre --model-learning-rate $lrm --encoder-optimizer adam --num-steps 50000 \
    --fix-prior-training --post-train-hmm --post-train-hmm-steps 50000 --gpus 0 --save --base-dir $base_dir

    python -m scripts.bisim.puck_stack.learn_predict_q_hmm_prior dataset/2_4x4_dqn_eps_0.1_fd.pickle 4 2 \
    --num-blocks 32 --num-components 1000 --beta0 1 --beta1 $beta --beta2 $beta --new-dones \
    --encoder-learning-rate $lre --model-learning-rate $lrm --encoder-optimizer adam --num-steps 50000 \
    --fix-prior-training --post-train-hmm --post-train-hmm-steps 50000 --gpus 0 --save --base-dir $base_dir


    python -m scripts.bisim.puck_stack.learn_predict_q_hmm_prior dataset/3_2x2_dqn_eps_0.1_fd.pickle 2 3 \
    --num-blocks 32 --num-components 1000 --beta0 1 --beta1 $beta --beta2 $beta --new-dones \
    --encoder-learning-rate $lre --model-learning-rate $lrm --encoder-optimizer adam --num-steps 50000 \
    --fix-prior-training --post-train-hmm --post-train-hmm-steps 50000 --gpus 0 --save --base-dir $base_dir

    python -m scripts.bisim.puck_stack.learn_predict_q_hmm_prior dataset/3_3x3_dqn_eps_0.1_fd.pickle 3 3 \
    --num-blocks 32 --num-components 1000 --beta0 1 --beta1 $beta --beta2 $beta --new-dones \
    --encoder-learning-rate $lre --model-learning-rate $lrm --encoder-optimizer adam --num-steps 50000 \
    --fix-prior-training --post-train-hmm --post-train-hmm-steps 50000 --gpus 0 --save --base-dir $base_dir

    python -m scripts.bisim.puck_stack.learn_predict_q_hmm_prior dataset/3_4x4_dqn_eps_0.1_fd.pickle 4 3 \
    --num-blocks 32 --num-components 1000 --beta0 1 --beta1 $beta --beta2 $beta --new-dones \
    --encoder-learning-rate $lre --model-learning-rate $lrm --encoder-optimizer adam --num-steps 50000 \
    --fix-prior-training --post-train-hmm --post-train-hmm-steps 50000 --gpus 0 --save --base-dir $base_dir

done
