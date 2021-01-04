#!/usr/bin/env bash

if [ ! -d "dataset" ]; then
    mkdir -p "dataset"
fi

python -m scripts.solve.minatar.dqn_pytorch -g breakout -v --save-dataset --num-frames 3000000
python -m scripts.solve.minatar.dqn_pytorch -g space_invaders -v --save-dataset --num-frames 3000000
python -m scripts.solve.minatar.dqn_pytorch -g freeway -v --save-dataset --num-frames 3000000
python -m scripts.solve.minatar.dqn_pytorch -g asterix -v --save-dataset --num-frames 3000000
