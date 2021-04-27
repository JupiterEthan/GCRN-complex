import os
import random
from datetime import datetime


random.seed(datetime.now())

wsj0_path = '../../data/corpora/WSJ0_83spks'

tr_n_ex = 320000

tr_snr_range = (-5.0, 0.0)
cv_snr = -5.0
tt_snrs = [-5.0, 0.0, 5.0]


with open('trainFileList.txt', 'r') as f:
    tr_list = [line.strip() for line in f.readlines()]

with open('validFileList.txt', 'r') as f:
    cv_list = [line.strip() for line in f.readlines()]

with open('testFileList.txt', 'r') as f:
    tt_list = [line.strip() for line in f.readlines()]


with open('mix_tr.txt', 'w') as f:
    for i in range(tr_n_ex):
        snr = random.uniform(tr_snr_range[0], tr_snr_range[1])
        f.write(random.sample(tr_list, 1)[0] + ' {:.5f}\n'.format(snr))

with open('mix_cv.txt', 'w') as f:
    for filename in cv_list:
        f.write(filename + ' {:.5f}\n'.format(cv_snr))

for snr in tt_snrs:
    with open('mix_tt_snr{}.txt'.format(snr), 'w') as f:
        for filename in tt_list:
            f.write(filename + ' {:.5f}\n'.format(snr))

           
