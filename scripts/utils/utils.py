import os
import json
import logging

import numpy as np
import torch


def getLogger(name,
              format_str='%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s',
              date_format='%Y-%m-%d %H:%M:%S',
              log_file=False):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # file or console
    handler = logging.StreamHandler() if not log_file else logging.FileHandler(name)
    formatter = logging.Formatter(fmt=format_str, datefmt=date_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def numParams(net):
    count = sum([int(np.prod(param.shape)) for param in net.parameters()])
    return count


def countFrames(n_samples, win_size, hop_size):
    n_overlap = win_size // hop_size
    fn = lambda x: x // hop_size + n_overlap - 1
    n_frames = torch.stack(list(map(fn, n_samples)), dim=0)
    return n_frames


def lossMask(shape, n_frames, device):
    loss_mask = torch.zeros(shape, dtype=torch.float32, device=device)
    for i, seq_len in enumerate(n_frames):
        loss_mask[i,:,0:seq_len,:] = 1.0
    return loss_mask


def lossLog(log_filename, ckpt, logging_period):
    if ckpt.ckpt_info['cur_epoch'] == 0 and ckpt.ckpt_info['cur_iter'] + 1 == logging_period:
        with open(log_filename, 'w') as f:
            f.write('epoch, iter, tr_loss, cv_loss\n')
            f.write('{}, {}, {:.4f}, {:.4f}\n'.format(ckpt.ckpt_info['cur_epoch']+1,
                ckpt.ckpt_info['cur_iter']+1, ckpt.ckpt_info['tr_loss'], ckpt.ckpt_info['cv_loss']))
    else:
        with open(log_filename, 'a') as f:
            f.write('{}, {}, {:.4f}, {:.4f}\n'.format(ckpt.ckpt_info['cur_epoch']+1,
                ckpt.ckpt_info['cur_iter']+1, ckpt.ckpt_info['tr_loss'], ckpt.ckpt_info['cv_loss']))


def wavNormalize(*sigs):
    # sigs is a list of signals to be normalized
    scale = max([np.max(np.abs(sig)) for sig in sigs]) + np.finfo(np.float32).eps
    sigs_norm = [sig / scale for sig in sigs]
    return sigs_norm


def dump_json(filename, obj):
    with open(filename, 'w') as f:
        json.dump(obj, f, indent=4, sort_keys=True)
    return


def load_json(filename):
    if not os.path.isfile(filename):
        raise FileNotFoundError('Could not find json file: {}'.format(filename))
    with open(filename, 'r') as f:
        obj = json.load(f)
    return obj
