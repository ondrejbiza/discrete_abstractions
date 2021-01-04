#!/usr/bin/env bash

if [ ! -d "dataset" ]; then
    mkdir -p "dataset"
fi

# 2 building
python -m scripts.solve.continuous_multi_task_buildings.dqn 4 4 2 1 --active-indices 0 --height-noise-std 0.05 \
    --num-filters 32 64 128 256 --filter-sizes 4 4 4 4 --strides 2 2 2 2 --hiddens 512 --target-size 64 \
    --batch-size 32 --learning-rate 0.00005 --buffer-size 100000 --max-time-steps 100000 --exploration-fraction 0.5 \
    --max-episodes 200000 --gpus 0 --save-exp-path dataset/4x4_b_dqn_2B_eps_0.1.pickle --save-q-values \
    --save-exp-num 100000 --save-exp-after-training --save-exp-eps 0.1 \
    --rewards-file dataset/4x4_b_dqn_2B_eps_0.1 \
    --save-weights dataset/4x4_b_dqn_2B_eps_0.1

# 3 building
python -m scripts.solve.continuous_multi_task_buildings.dqn 4 4 2 1 --active-indices 1 --height-noise-std 0.05 \
    --num-filters 32 64 128 256 --filter-sizes 4 4 4 4 --strides 2 2 2 2 --hiddens 512 --target-size 64 \
    --batch-size 32 --learning-rate 0.00005 --buffer-size 100000 --max-time-steps 100000 --exploration-fraction 0.5 \
    --max-episodes 200000 --gpus 0 --save-exp-path dataset/4x4_b_dqn_3B_eps_0.1.pickle --save-q-values \
    --save-exp-num 100000 --save-exp-after-training --save-exp-eps 0.1 \
    --rewards-file dataset/4x4_b_dqn_3B_eps_0.1 \
    --save-weights dataset/4x4_b_dqn_3B_eps_0.1

# 3 top
python -m scripts.solve.continuous_multi_task_buildings.dqn 4 4 2 1 --active-indices 2 --height-noise-std 0.05 \
    --num-filters 32 64 128 256 --filter-sizes 4 4 4 4 --strides 2 2 2 2 --hiddens 512 --target-size 64 \
    --batch-size 32 --learning-rate 0.00005 --buffer-size 100000 --max-time-steps 100000 --exploration-fraction 0.5 \
    --max-episodes 200000 --gpus 0 --save-exp-path dataset/4x4_b_dqn_3_top_eps_0.1.pickle --save-q-values \
    --save-exp-num 100000 --save-exp-after-training --save-exp-eps 0.1 \
    --rewards-file dataset/4x4_b_dqn_3_top_eps_0.1 \
    --save-weights dataset/4x4_b_dqn_3_top_eps_0.1

# 3 left
python -m scripts.solve.continuous_multi_task_buildings.dqn 4 4 2 1 --active-indices 4 --height-noise-std 0.05 \
    --num-filters 32 64 128 256 --filter-sizes 4 4 4 4 --strides 2 2 2 2 --hiddens 512 --target-size 64 \
    --batch-size 32 --learning-rate 0.00005 --buffer-size 100000 --max-time-steps 100000 --exploration-fraction 0.5 \
    --max-episodes 200000 --gpus 0 --save-exp-path dataset/4x4_b_dqn_3_left_eps_0.1.pickle --save-q-values \
    --save-exp-num 100000 --save-exp-after-training --save-exp-eps 0.1 \
    --rewards-file dataset/4x4_b_dqn_3_left_eps_0.1 \
    --save-weights dataset/4x4_b_dqn_3_left_eps_0.1

# 3 building, 3 top
python -m scripts.solve.continuous_multi_task_buildings.dqn 4 4 2 1 --active-indices 1 2 --height-noise-std 0.05 \
    --num-filters 32 64 128 256 --filter-sizes 4 4 4 4 --strides 2 2 2 2 --hiddens 512 --target-size 64 \
    --batch-size 32 --learning-rate 0.00005 --buffer-size 100000 --max-time-steps 100000 --exploration-fraction 0.5 \
    --max-episodes 200000 --gpus 0 --save-exp-path dataset/4x4_b_dqn_3B_3_top_eps_0.1.pickle --save-q-values \
    --save-exp-num 100000 --save-exp-after-training --save-exp-eps 0.1 \
    --rewards-file dataset/4x4_b_dqn_3B_3_top_eps_0.1 \
    --save-weights dataset/4x4_b_dqn_3B_3_top_eps_0.1

# 3 top, 3 left
python -m scripts.solve.continuous_multi_task_buildings.dqn 4 4 2 1 --active-indices 2 4 --height-noise-std 0.05 \
    --num-filters 32 64 128 256 --filter-sizes 4 4 4 4 --strides 2 2 2 2 --hiddens 512 --target-size 64 \
    --batch-size 32 --learning-rate 0.00005 --buffer-size 100000 --max-time-steps 100000 --exploration-fraction 0.5 \
    --max-episodes 200000 --gpus 0 --save-exp-path dataset/4x4_b_dqn_3B_3_left_eps_0.1.pickle --save-q-values \
    --save-exp-num 100000 --save-exp-after-training --save-exp-eps 0.1 \
    --rewards-file dataset/4x4_b_dqn_3B_3_left_eps_0.1 \
    --save-weights dataset/4x4_b_dqn_3B_3_left_eps_0.1

# 3 building, 3 top, 3 left
python -m scripts.solve.continuous_multi_task_buildings.dqn 4 4 2 1 --active-indices 1 2 4 --height-noise-std 0.05 \
    --num-filters 32 64 128 256 --filter-sizes 4 4 4 4 --strides 2 2 2 2 --hiddens 512 --target-size 64 \
    --batch-size 32 --learning-rate 0.00005 --buffer-size 100000 --max-time-steps 100000 --exploration-fraction 0.5 \
    --max-episodes 200000 --gpus 0 --save-exp-path dataset/4x4_b_dqn_3B_3_top_3_left_eps_0.1.pickle --save-q-values \
    --save-exp-num 100000 --save-exp-after-training --save-exp-eps 0.1 \
    --rewards-file dataset/4x4_b_dqn_3B_3_top_3_left_eps_0.1 \
    --save-weights dataset/4x4_b_dqn_3B_3_top_3_left_eps_0.1
