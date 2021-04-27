import os

import soundfile as sf
import numpy as np
from pystoi import stoi
from pypesq import pesq

from configs import exp_conf
from utils.utils import getLogger


def snr(ref, est):
    ratio = 10 * np.log10(np.sum(ref**2) / np.sum((ref - est)**2))
    return ratio
    

class Metric(object): 
    def __init__(self, args):
        self.sample_rate = exp_conf['sample_rate']
        
        self.est_path = args.est_path
        self.tt_list = args.tt_list
        self.metric = args.metric
        assert self.metric in {'stoi', 'estoi', 'pesq', 'snr'}
        
        self.ckpt_dir = args.ckpt_dir
        if not os.path.isdir(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

    def evaluate(self):
        getattr(self, self.metric)()

    def apply_metric(self, metric_func):
        logger = getLogger(os.path.join(self.ckpt_dir, self.metric+'_scores.log'), log_file=True)

        all_scores_dir = os.path.join(self.ckpt_dir, 'all_scores')
        if not os.path.isdir(all_scores_dir):
            os.makedirs(all_scores_dir)
        if not os.path.isdir(os.path.join(all_scores_dir, 'scores_arrays')):
            os.makedirs(os.path.join(all_scores_dir, 'scores_arrays'))
        
        if self.tt_list == '':
            conditions = os.listdir(self.est_path)
        else:
            with open(self.tt_list, 'r') as f:
                conditions = [t.strip().split('/')[-1].replace('.ex', '') for t in f.readlines()]

        for condition in conditions:
            mix_scores_array = []
            est_scores_array = []

            score_name = condition + '_' + self.metric
            f = open(os.path.join(all_scores_dir, score_name + '.txt'), 'w')
            count = 0
            for filename in os.listdir(os.path.join(self.est_path, condition)):
                if not filename.endswith('_mix.wav'):
                    continue
                count += 1
                mix, _ = sf.read(os.path.join(self.est_path, condition, filename), dtype=np.float32)
                sph, _ = sf.read(os.path.join(self.est_path, condition, filename.replace('_mix', '_sph')), dtype=np.float32)
                sph_est, _ = sf.read(os.path.join(self.est_path, condition, filename.replace('_mix', '_sph_est')), dtype=np.float32)
                mix_score = metric_func(sph, mix)
                est_score = metric_func(sph, sph_est)
                f.write('utt {}: mix {:.4f}, est {:.4f}\n'.format(filename, mix_score, est_score))
                f.flush()
                mix_scores_array.append(mix_score)
                est_scores_array.append(est_score)

            mix_scores_array = np.array(mix_scores_array, dtype=np.float32)
            est_scores_array = np.array(est_scores_array, dtype=np.float32)
            f.write('========================================\n')
            f.write('{} results: ({} utts)\n'.format(self.metric, count))
            f.write('mix : {:.4f} +- {:.4f}\n'.format(np.mean(mix_scores_array), np.std(mix_scores_array)))
            f.write('est : {:.4f} +- {:.4f}\n'.format(np.mean(est_scores_array), np.std(est_scores_array)))
            f.close()
            np.save(os.path.join(all_scores_dir, 'scores_arrays', score_name + '_mix.npy'), mix_scores_array)
            np.save(os.path.join(all_scores_dir, 'scores_arrays', score_name + '_est.npy'), est_scores_array)

            message = 'Evaluating {}: {} utts: '.format(condition, count) + \
                '{} [ mix: {:.4f}, est: {:.4f} | delta: {:.4f} ]'.format(self.metric, 
                np.mean(mix_scores_array), np.mean(est_scores_array), 
                np.mean(est_scores_array)-np.mean(mix_scores_array))
            logger.info(message)

    def stoi(self):
        fn = lambda ref, est: stoi(ref, est, self.sample_rate, extended=False)
        self.apply_metric(fn)

    def estoi(self):
        fn = lambda ref, est: stoi(ref, est, self.sample_rate, extended=True)
        self.apply_metric(fn)
    
    def pesq(self):
        fn = lambda ref, est: pesq(ref, est, self.sample_rate)
        self.apply_metric(fn)
    
    def snr(self):
        fn = lambda ref, est: snr(ref, est)
        self.apply_metric(fn)
