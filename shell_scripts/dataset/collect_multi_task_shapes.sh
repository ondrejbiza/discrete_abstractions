#!/usr/bin/env bash

if [ ! -d "dataset" ]; then
    mkdir -p "dataset"
fi

# 2 pucks 1 redundant
python -m scripts.solve.continuous_multi_task.dqn 4 4 3 --active-indices 0 --height-noise-std 0.05 \
    --num-filters 32 64 128 256 --filter-sizes 4 4 4 4 --strides 2 2 2 2 --hiddens 512 --target-size 64 \
    --batch-size 32 --learning-rate 0.00005 --buffer-size 100000 --max-time-steps 100000 --exploration-fraction 0.5 \
    --max-episodes 200000 --gpus 0 --save-exp-path dataset/4x4_dqn_2_pucks_1_r_eps_0.1_rs.pickle --save-q-values \
    --save-exp-num 100000 --save-exp-after-training --save-exp-eps 0.1 \
    --rewards-file dataset/4x4_dqn_2_pucks_1_r_eps_0.1_rs \
    --save-weights dataset/4x4_dqn_2_pucks_1_r_eps_0.1_rs --random-shape

# 3 pucks
python -m scripts.solve.continuous_multi_task.dqn 4 4 3 --active-indices 1 --height-noise-std 0.05 \
    --num-filters 32 64 128 256 --filter-sizes 4 4 4 4 --strides 2 2 2 2 --hiddens 512 --target-size 64 \
    --batch-size 32 --learning-rate 0.00005 --buffer-size 100000 --max-time-steps 100000 --exploration-fraction 0.5 \
    --max-episodes 200000 --gpus 0 --save-exp-path dataset/4x4_dqn_3_pucks_eps_0.1_rs.pickle --save-q-values \
    --save-exp-num 100000 --save-exp-after-training --save-exp-eps 0.1 \
    --rewards-file dataset/4x4_dqn_3_pucks_eps_0.1_rs \
    --save-weights dataset/4x4_dqn_3_pucks_eps_0.1_rs --random-shape

# 3 stairs xx
python -m scripts.solve.continuous_multi_task.dqn 4 4 3 --active-indices 5 --height-noise-std 0.05 \
    --num-filters 32 64 128 256 --filter-sizes 4 4 4 4 --strides 2 2 2 2 --hiddens 512 --target-size 64 \
    --batch-size 32 --learning-rate 0.00005 --buffer-size 100000 --max-time-steps 100000 --exploration-fraction 0.5 \
    --max-episodes 200000 --gpus 0 --save-exp-path dataset/4x4_dqn_3_stairs_eps_0.1_rs.pickle --save-q-values \
    --save-exp-num 100000 --save-exp-after-training --save-exp-eps 0.1 \
    --rewards-file dataset/4x4_dqn_3_stairs_eps_0.1_rs \
    --save-weights dataset/4x4_dqn_3_stairs_eps_0.1_rs --random-shape

# 3 pucks, 3 row
python -m scripts.solve.continuous_multi_task.dqn 4 4 3 --active-indices 1 3 --height-noise-std 0.05 \
    --num-filters 32 64 128 256 --filter-sizes 4 4 4 4 --strides 2 2 2 2 --hiddens 512 --target-size 64 \
    --batch-size 32 --learning-rate 0.00005 --buffer-size 100000 --max-time-steps 100000 --exploration-fraction 0.5 \
    --max-episodes 200000 --gpus 0 --save-exp-path dataset/4x4_dqn_3_pucks_and_3_row_eps_0.1_rs.pickle --save-q-values \
    --save-exp-num 100000 --save-exp-after-training --save-exp-eps 0.1 \
    --rewards-file dataset/4x4_dqn_3_pucks_and_3_row_eps_0.1_rs \
    --save-weights dataset/4x4_dqn_3_pucks_and_3_row_eps_0.1_rs --random-shape

# 3 pucks, 3 stairs
python -m scripts.solve.continuous_multi_task.dqn 4 4 3 --active-indices 1 5 --height-noise-std 0.05 \
    --num-filters 32 64 128 256 --filter-sizes 4 4 4 4 --strides 2 2 2 2 --hiddens 512 --target-size 64 \
    --batch-size 32 --learning-rate 0.00005 --buffer-size 100000 --max-time-steps 100000 --exploration-fraction 0.5 \
    --max-episodes 200000 --gpus 0 --save-exp-path dataset/4x4_dqn_3_pucks_and_3_stairs_eps_0.1_rs.pickle --save-q-values \
    --save-exp-num 100000 --save-exp-after-training --save-exp-eps 0.1 \
    --rewards-file dataset/4x4_dqn_3_pucks_and_3_stairs_eps_0.1_rs \
    --save-weights dataset/4x4_dqn_3_pucks_and_3_stairs_eps_0.1_rs --random-shape

# 2 pucks, 2 row
python -m scripts.solve.continuous_multi_task.dqn 4 4 3 --active-indices 0 2 --height-noise-std 0.05 \
    --num-filters 32 64 128 256 --filter-sizes 4 4 4 4 --strides 2 2 2 2 --hiddens 512 --target-size 64 \
    --batch-size 32 --learning-rate 0.00005 --buffer-size 100000 --max-time-steps 100000 --exploration-fraction 0.5 \
    --max-episodes 200000 --gpus 0 --save-exp-path dataset/4x4_dqn_2_pucks_2_row_1_r_eps_0.1_rs.pickle --save-q-values \
    --save-exp-num 100000 --save-exp-after-training --save-exp-eps 0.1 \
    --rewards-file dataset/4x4_dqn_2_pucks_2_row_1_r_eps_0.1_rs \
    --save-weights dataset/4x4_dqn_2_pucks_2_row_1_r_eps_0.1_rs --random-shape

# 2 and 2 pucks
python -m scripts.solve.continuous_multi_task.dqn 4 4 4 --active-indices 4 --height-noise-std 0.05 \
    --num-filters 32 64 128 256 --filter-sizes 4 4 4 4 --strides 2 2 2 2 --hiddens 512 --target-size 64 \
    --batch-size 32 --learning-rate 0.00005 --buffer-size 100000 --max-time-steps 100000 --exploration-fraction 0.5 \
    --max-episodes 200000 --gpus 0 --save-exp-path dataset/4x4_dqn_2_and_2_pucks_eps_0.1_rs.pickle --save-q-values \
    --save-exp-num 100000 --save-exp-after-training --save-exp-eps 0.1 \
    --rewards-file dataset/4x4_dqn_2_and_2_pucks_eps_0.1_rs \
    --save-weights dataset/4x4_dqn_2_and_2_pucks_eps_0.1_rs --random-shape

# 3 pucks, 2 and 2 pucks
python -m scripts.solve.continuous_multi_task.dqn 4 4 4 --active-indices 1 4 --height-noise-std 0.05 \
    --num-filters 32 64 128 256 --filter-sizes 4 4 4 4 --strides 2 2 2 2 --hiddens 512 --target-size 64 \
    --batch-size 32 --learning-rate 0.00005 --buffer-size 100000 --max-time-steps 100000 --exploration-fraction 0.5 \
    --max-episodes 200000 --gpus 0 --save-exp-path dataset/4x4_dqn_2_and_2_pucks_3_pucks_eps_0.1_rs.pickle --save-q-values \
    --save-exp-num 100000 --save-exp-after-training --save-exp-eps 0.1 \
    --rewards-file dataset/4x4_dqn_2_and_2_pucks_3_pucks_eps_0.1_rs \
    --save-weights dataset/4x4_dqn_2_and_2_pucks_3_pucks_eps_0.1_rs --random-shape

# 3 stairs, 2 and 2 pucks xx
python -m scripts.solve.continuous_multi_task.dqn 4 4 4 --active-indices 4 5 --height-noise-std 0.05 \
    --num-filters 32 64 128 256 --filter-sizes 4 4 4 4 --strides 2 2 2 2 --hiddens 512 --target-size 64 \
    --batch-size 32 --learning-rate 0.00005 --buffer-size 100000 --max-time-steps 100000 --exploration-fraction 0.5 \
    --max-episodes 200000 --gpus 0 --save-exp-path dataset/4x4_dqn_2_and_2_pucks_3_stairs_eps_0.1_rs.pickle --save-q-values \
    --save-exp-num 100000 --save-exp-after-training --save-exp-eps 0.1 \
    --rewards-file dataset/4x4_dqn_2_and_2_pucks_3_stairs_eps_0.1_rs \
    --save-weights dataset/4x4_dqn_2_and_2_pucks_3_stairs_eps_0.1_rs --random-shape

# 3 row, 2 and 2 pucks xx
python -m scripts.solve.continuous_multi_task.dqn 4 4 4 --active-indices 3 4 --height-noise-std 0.05 \
    --num-filters 32 64 128 256 --filter-sizes 4 4 4 4 --strides 2 2 2 2 --hiddens 512 --target-size 64 \
    --batch-size 32 --learning-rate 0.00005 --buffer-size 100000 --max-time-steps 100000 --exploration-fraction 0.5 \
    --max-episodes 200000 --gpus 0 --save-exp-path dataset/4x4_dqn_2_and_2_pucks_3_row_eps_0.1_rs.pickle --save-q-values \
    --save-exp-num 100000 --save-exp-after-training --save-exp-eps 0.1 \
    --rewards-file dataset/4x4_dqn_2_and_2_pucks_3_row_eps_0.1_rs \
    --save-weights dataset/4x4_dqn_2_and_2_pucks_3_row_eps_0.1_rs --random-shape
