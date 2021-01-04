#!/usr/bin/env bash

if [ ! -d "dataset" ]; then
    mkdir -p "dataset"
fi

eps=0.1

python -m scripts.solve.continuous_puck_stack_n.dqn 2 2 2 --height-noise-std 0.05 --num-filters 32 64 128 256 \
    --filter-sizes 4 4 4 4 --strides 2 2 2 2 --hiddens 512 --target-size 64 --batch-size 32 --learning-rate 0.0001 \
    --buffer-size 10000 --max-time-steps 10000 --exploration-fraction 0.5 --max-episodes 1000000 \
    --save-exp-path dataset/2_2x2_dqn_eps_${eps}_rs.pickle --save-q-values --gpu-memory-fraction 0.6 --gpus 0 \
    --save-weights dataset/2_2x2_dqn_eps_${eps}_rs --save-exp-num 50000 --save-exp-after-training --save-exp-eps $eps \
    --rewards-file dataset/2_2x2_dqn_eps_${eps}_rs.dat --fix-dones --random-shape

python -m scripts.solve.continuous_puck_stack_n.dqn 3 3 2 --height-noise-std 0.05 --num-filters 32 64 128 256 \
    --filter-sizes 4 4 4 4 --strides 2 2 2 2 --hiddens 512 --target-size 64 --batch-size 32 --learning-rate 0.0001 \
    --buffer-size 10000 --max-time-steps 10000 --exploration-fraction 0.5 --max-episodes 1000000 \
    --save-exp-path dataset/2_3x3_dqn_eps_${eps}_rs.pickle --save-q-values --gpu-memory-fraction 0.6 --gpus 0 \
    --save-weights dataset/2_3x3_dqn_eps_${eps}_rs --save-exp-num 50000 --save-exp-after-training --save-exp-eps $eps \
    --rewards-file dataset/2_3x3_dqn_eps_${eps}_rs.dat --fix-dones --random-shape

python -m scripts.solve.continuous_puck_stack_n.dqn 4 4 2 --height-noise-std 0.05 --num-filters 32 64 128 256 \
    --filter-sizes 4 4 4 4 --strides 2 2 2 2 --hiddens 512 --target-size 64 --batch-size 32 --learning-rate 0.00005 \
    --buffer-size 100000 --max-time-steps 100000 --exploration-fraction 0.8 --max-episodes 1000000 \
    --save-exp-path dataset/2_4x4_dqn_eps_${eps}_rs.pickle --save-q-values --gpu-memory-fraction 0.6 --gpus 0 \
    --save-weights dataset/2_4x4_dqn_eps_${eps}_rs --save-exp-num 50000 --save-exp-after-training --save-exp-eps $eps \
    --rewards-file dataset/2_4x4_dqn_eps_${eps}_rs.dat --fix-dones --random-shape

python -m scripts.solve.continuous_puck_stack_n.dqn 2 2 3 --height-noise-std 0.05 --num-filters 32 64 128 256 \
    --filter-sizes 4 4 4 4 --strides 2 2 2 2 --hiddens 512 --target-size 64 --batch-size 32 --learning-rate 0.0001 \
    --buffer-size 10000 --max-time-steps 10000 --exploration-fraction 0.5 --max-episodes 1000000 \
    --save-exp-path dataset/3_2x2_dqn_eps_${eps}_rs.pickle --save-q-values --gpu-memory-fraction 0.6 --gpus 0 \
    --save-weights dataset/3_2x2_dqn_eps_${eps}_rs --save-exp-num 50000 --save-exp-after-training --save-exp-eps $eps \
    --rewards-file dataset/3_2x2_dqn_eps_${eps}_rs.dat --fix-dones --random-shape

python -m scripts.solve.continuous_puck_stack_n.dqn 3 3 3 --height-noise-std 0.05 --num-filters 32 64 128 256 \
    --filter-sizes 4 4 4 4 --strides 2 2 2 2 --hiddens 512 --target-size 64 --batch-size 32 --learning-rate 0.00005 \
    --buffer-size 50000 --max-time-steps 50000 --exploration-fraction 0.5 --max-episodes 1000000 \
    --save-exp-path dataset/3_3x3_dqn_eps_${eps}_rs.pickle --save-q-values --gpu-memory-fraction 0.6 --gpus 0 \
    --save-weights dataset/3_3x3_dqn_eps_${eps}_rs --save-exp-num 50000 --save-exp-after-training --save-exp-eps $eps \
    --rewards-file dataset/3_3x3_dqn_eps_${eps}_rs.dat --fix-dones --random-shape

python -m scripts.solve.continuous_puck_stack_n.dqn 4 4 3 --height-noise-std 0.05 --num-filters 32 64 128 256 \
    --filter-sizes 4 4 4 4 --strides 2 2 2 2 --hiddens 512 --target-size 64 --batch-size 32 --learning-rate 0.00005 \
    --buffer-size 400000 --max-time-steps 400000 --exploration-fraction 0.5 --max-episodes 1000000 \
    --save-exp-path dataset/3_4x4_dqn_eps_${eps}_rs.pickle --save-q-values --gpu-memory-fraction 0.6 --gpus 0 \
    --save-weights dataset/3_4x4_dqn_eps_${eps}_rs --save-exp-num 50000 --save-exp-after-training --save-exp-eps $eps \
    --rewards-file dataset/3_4x4_dqn_eps_${eps}_rs.dat --fix-dones --random-shape
