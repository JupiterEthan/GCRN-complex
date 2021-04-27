#!/bin/bash


python -B ./utils/gen_mix.py --mode=tt

python -B ./utils/gen_mix.py --mode=cv

python -B ./utils/gen_mix.py --mode=tr
