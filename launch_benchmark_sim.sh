#!/bin/sh

echo
echo
echo '       ██████  ██████  ██    ██                       '
echo '      ██       ██   ██ ██    ██                       '
echo '      ██   ███ ██████  ██    ██                       '
echo '      ██    ██ ██      ██    ██                       '
echo '       ██████  ██       ██████                        '
echo '                                                      '
echo '                                                      '
echo '       ██████   ██████                                '
echo '      ██       ██    ██                               '
echo '      ██   ███ ██    ██                               '
echo '      ██    ██ ██    ██                               '
echo '       ██████   ██████                                '
echo '                                                      '
echo '                                                      '
echo '      ██████  ██████  ██████  ██████  ██████  ██████  '
echo '      ██   ██ ██   ██ ██   ██ ██   ██ ██   ██ ██   ██ '
echo '      ██████  ██████  ██████  ██████  ██████  ██████  '
echo '      ██   ██ ██   ██ ██   ██ ██   ██ ██   ██ ██   ██ '
echo '      ██████  ██   ██ ██   ██ ██   ██ ██   ██ ██   ██ '
echo
echo

CUDA_VISIBLE_DEVICES=0 uv run python benchmark_sim.py & \
CUDA_VISIBLE_DEVICES=1 uv run python benchmark_sim.py & \
CUDA_VISIBLE_DEVICES=2 uv run python benchmark_sim.py & \
CUDA_VISIBLE_DEVICES=3 uv run python benchmark_sim.py & \
CUDA_VISIBLE_DEVICES=4 uv run python benchmark_sim.py & \
CUDA_VISIBLE_DEVICES=5 uv run python benchmark_sim.py & \
CUDA_VISIBLE_DEVICES=6 uv run python benchmark_sim.py & \
CUDA_VISIBLE_DEVICES=7 uv run python benchmark_sim.py &
