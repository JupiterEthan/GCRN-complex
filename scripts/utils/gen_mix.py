import sys
import os
import random
from datetime import datetime

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import h5py
from tqdm import tqdm, trange
import gflags
import pymp


################# 1. Configuration
#### parse commands
flags = gflags.FLAGS
gflags.DEFINE_string('mode', '', 'tr, cv, or tt')
flags(sys.argv)

#### file paths
filelist_path = '../filelists'
speech_path = '../data/corpora/WSJ0_83spks'
tr_noise_path = '../data/corpora/noise_10000/'
cv_noise_path = '../data/corpora/noise'
tt_noise_path = '../data/corpora/noise'

tr_mix_path = '../data/datasets/tr'
cv_mix_path = '../data/datasets/cv'
tt_mix_path = '../data/datasets/tt'

#### settings
n_workers = 20 # number of processes used for creating training set
folder_capacity = 5000 # the capacity of each subfolder for training set
sample_rate = 16000 # sampling rate of signals (Hz)
rms = 1.0
tt_snrs = [-5.0, 0.0, 5.0]

tr_noise = 'long_wave.bin'
cv_noise = 'factory1.wav'
tt_noises = ['ADTbabble.wav', 'ADTcafeteria1.wav']

################# 2. Generating mixtures
assert flags.mode in {'tr', 'cv', 'tt'}

# read the recipe
if flags.mode in {'tr', 'cv'}:
    with open(os.path.join(filelist_path, 'recipe', 'mix_{}.txt'.format(flags.mode)), 'r') as f:
        lines = f.readlines()
        recipe_lines = []
        for line in lines:
            line = line.strip().split()
            line = {'sph': line[0],
                    'snr': float(line[1])}
            recipe_lines.append(line)
else:
    recipe_lines_t = dict()
    for tt_snr in tt_snrs:
        with open(os.path.join(filelist_path, 'recipe', 'mix_{}_snr{}.txt'.format(flags.mode, tt_snr)), 'r') as f:
            lines = f.readlines()
            recipe_lines = []
            for line in lines:
                line = line.strip().split()
                line = {'sph': line[0],
                        'snr': float(line[1])}
                recipe_lines.append(line)
        recipe_lines_t[tt_snr] = recipe_lines

random.seed(datetime.now())

print('[{}] Generating mixtures...'.format(flags.mode))
#create mixtures
if flags.mode == 'tr':
    # read noise
    n = np.memmap(os.path.join(tr_noise_path, tr_noise), dtype=np.float32, mode='r')


    pbar = tqdm(total=len(recipe_lines)) # set up a progress bar
    pbar_count = pymp.shared.list() # count generated mixtures, and share it among multiple processes
    noi_segs = pymp.shared.dict()
    pbar_count.append(0) # init with zero

    with pymp.Parallel(n_workers) as p:
        for count in p.range(len(recipe_lines)):
            # prepare speech signals
            sph_info = recipe_lines[count]
            sph, fs = sf.read(os.path.join(speech_path, sph_info['sph']), dtype=np.float32)
            if fs != sample_rate:
                sph = resample_poly(sph, sample_rate, fs)
            sph_len = sph.size

            snr = sph_info['snr']

            # choose a point where we start to cut
            s_point = random.randint(0, n.size-sph_len)
            while np.sum(n[s_point:s_point+sph_len]**2) == 0.0:
                s_point = random.randint(0, n.size-sph_len)
            # cut noise
            noi = n[s_point:s_point+sph_len]
            # mixture = speech + noise
            scale_noi = np.sqrt(np.sum(sph**2) / (np.sum(noi**2) * (10**(snr/10))))
            mix = sph + noi * scale_noi

            # normalize the root mean square of the mixture to a constant
            c = rms * np.sqrt(sph_len / np.sum(mix**2))
            mix *= c
            sph *= c

            # save the data
            subdir = '{}-{}'.format(count // folder_capacity * folder_capacity,
                (count // folder_capacity + 1) * folder_capacity - 1)
            filepath = os.path.join(tr_mix_path, subdir)
            if not os.path.isdir(filepath):
                os.makedirs(filepath)
            filename = '{}_{}.ex'.format(flags.mode, count) 
            writer = h5py.File(os.path.join(filepath, filename), 'w')
            writer.create_dataset('mix', data=mix.astype(np.float32), shape=mix.shape, chunks=True)
            writer.create_dataset('sph', data=sph.astype(np.float32), shape=sph.shape, chunks=True)
            writer.close()

            with p.lock:
                pbar.update(pbar_count[0])
                pbar_count[0] += 1
                noi_segs[count] = (s_point, s_point+sph_len)

    pbar.close()

    # create the file list for training set
    with open(os.path.join(filelist_path, 'tr_list.txt'), 'w') as f:
        for count in range(len(recipe_lines)):
            subdir = '{}-{}'.format(count // folder_capacity * folder_capacity,
                (count // folder_capacity + 1) * folder_capacity - 1)
            filename = '{}_{}.ex'.format(flags.mode, count)
            f.write(os.path.join(tr_mix_path, subdir, filename) + '\n')

    # save noise segments info
    with open(os.path.join(filelist_path, 'recipe', 'tr_noi_segs.info'), 'w') as f:
        for count in range(len(recipe_lines)):
            f.write('{}, {}\n'.format(noi_segs[count][0], noi_segs[count][1]))

elif flags.mode == 'cv':
    # read noise
    n, _ = sf.read(os.path.join(cv_noise_path, cv_noise))

    noi_segs = []

    filepath = cv_mix_path
    if not os.path.isdir(filepath):
        os.makedirs(filepath)
    filename = '{}.ex'.format(flags.mode)
    writer = h5py.File(os.path.join(filepath, filename), 'w')
    for count in trange(len(recipe_lines)):
        # prepare speech signals
        sph_info = recipe_lines[count]
        sph, fs = sf.read(os.path.join(speech_path, sph_info['sph']), dtype=np.float32)
        if fs != sample_rate:
            sph = resample_poly(sph, sample_rate, fs)
        sph_len = sph.size

        snr = sph_info['snr']

        # choose a point where we start to cut
        s_point = random.randint(0, n.size-sph_len)
        while np.sum(n[s_point:s_point+sph_len]**2) == 0.0:
            s_point = random.randint(0, n.size-sph_len)
        # cut noise
        noi = n[s_point:s_point+sph_len]
        noi_segs.append((s_point, s_point+sph_len))
        # mixture = speech + noise
        scale_noi = np.sqrt(np.sum(sph**2) / (np.sum(noi**2) * (10**(snr/10))))
        mix = sph + noi * scale_noi

        # normalize the root mean square of the mixture to a constant
        c = rms * np.sqrt(sph_len / np.sum(mix**2))
        mix *= c
        sph *= c
        
        writer_grp = writer.create_group(str(count))
        writer_grp.create_dataset('mix', data=mix.astype(np.float32), shape=mix.shape, chunks=True)
        writer_grp.create_dataset('sph', data=sph.astype(np.float32), shape=sph.shape, chunks=True)

    writer.close()

    # save noise segments info
    with open(os.path.join(filelist_path, 'recipe', 'cv_noi_segs.info'), 'w') as f:
        for seg_info in noi_segs:
            f.write('{}, {}\n'.format(seg_info[0], seg_info[1]))

elif flags.mode == 'tt':
    filepath = tt_mix_path
    if not os.path.isdir(filepath):
        os.makedirs(filepath)
    for tt_snr in tt_snrs:
        for tt_noise in tt_noises:
            # read noise
            n, _ = sf.read(os.path.join(tt_noise_path, tt_noise))

            noi_segs = []

            filename = '{}_{}_snr{}.ex'.format(flags.mode, tt_noise.replace('.wav', ''), tt_snr)
            writer = h5py.File(os.path.join(filepath, filename), 'w')
            for count in trange(len(recipe_lines_t[tt_snr])):
                # prepare speech signals
                sph_info = recipe_lines_t[tt_snr][count]
                sph, fs = sf.read(os.path.join(speech_path, sph_info['sph']), dtype=np.float32)
                if fs != sample_rate:
                    sph = resample_poly(sph, sample_rate, fs)
                sph_len = sph.size

                snr = sph_info['snr']

                # choose a point where we start to cut
                s_point = random.randint(0, n.size-sph_len)
                while np.sum(n[s_point:s_point+sph_len]**2) == 0.0:
                    s_point = random.randint(0, n.size-sph_len)
                # cut noise
                noi = n[s_point:s_point+sph_len]
                noi_segs.append((s_point, s_point+sph_len))
                # mixture = speech + noise
                scale_noi = np.sqrt(np.sum(sph**2) / (np.sum(noi**2) * (10**(snr/10))))
                mix = sph + noi * scale_noi
        
                # normalize the root mean square of the mixture to a constant
                c = rms * np.sqrt(sph_len / np.sum(mix**2))
                mix *= c
                sph *= c
       
                writer_grp = writer.create_group(str(count))
                writer_grp.create_dataset('mix', data=mix.astype(np.float32), shape=mix.shape, chunks=True)
                writer_grp.create_dataset('sph', data=sph.astype(np.float32), shape=sph.shape, chunks=True)

            writer.close()

            # save noise segments info
            with open(os.path.join(filelist_path, 'recipe', 'tt_noi_segs_{}_snr{}.info'.format(tt_noise.replace('.wav', ''), tt_snr)), 'w') as f:
                for seg_info in noi_segs:
                    f.write('{}, {}\n'.format(seg_info[0], seg_info[1]))

print('[{}] Finish generating mixtures.\n'.format(flags.mode))
