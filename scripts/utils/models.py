import os
import shutil
import timeit

import numpy as np
import soundfile as sf
import torch
from torch.nn import DataParallel
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler

from configs import exp_conf
from utils.utils import getLogger, numParams, countFrames, lossMask, lossLog, wavNormalize
from utils.pipeline_modules import NetFeeder, Resynthesizer
from utils.data_utils import AudioLoader
from utils.networks import Net
from utils.criteria import LossFunction


class CheckPoint(object):
    def __init__(self, ckpt_info=None, net_state_dict=None, optim_state_dict=None):
        self.ckpt_info = ckpt_info
        self.net_state_dict = net_state_dict
        self.optim_state_dict = optim_state_dict
    
    def save(self, filename, is_best, best_model=None):
        torch.save(self, filename)
        if is_best:
            shutil.copyfile(filename, best_model)

    def load(self, filename, device):
        if not os.path.isfile(filename):
            raise FileNotFoundError('No checkpoint found at {}'.format(filename))
        ckpt = torch.load(filename, map_location=device)
        self.ckpt_info = ckpt.ckpt_info 
        self.net_state_dict = ckpt.net_state_dict
        self.optim_state_dict = ckpt.optim_state_dict


class Model(object):
    def __init__(self):
        self.in_norm = exp_conf['in_norm']
        self.sample_rate = exp_conf['sample_rate']
        self.win_len = exp_conf['win_len']
        self.hop_len = exp_conf['hop_len']

        self.win_size = int(self.win_len * self.sample_rate)
        self.hop_size = int(self.hop_len * self.sample_rate)
        
    def train(self, args):
        with open(args.tr_list, 'r') as f:
            self.tr_list = [line.strip() for line in f.readlines()]
        self.tr_size = len(self.tr_list)
        self.cv_file = args.cv_file
        self.ckpt_dir = args.ckpt_dir
        self.logging_period = args.logging_period
        self.resume_model = args.resume_model
        self.time_log = args.time_log
        self.lr = args.lr
        self.lr_decay_factor = args.lr_decay_factor
        self.lr_decay_period = args.lr_decay_period
        self.clip_norm = args.clip_norm
        self.max_n_epochs = args.max_n_epochs
        self.batch_size = args.batch_size
        self.buffer_size = args.buffer_size
        self.loss_log = args.loss_log
        self.unit = args.unit
        self.segment_size = args.segment_size
        self.segment_shift = args.segment_shift

        self.gpu_ids = tuple(map(int, args.gpu_ids.split(',')))
        if len(self.gpu_ids) == 1 and self.gpu_ids[0] == -1:
            # cpu only
            self.device = torch.device('cpu')
        else:
            # gpu
            self.device = torch.device('cuda:{}'.format(self.gpu_ids[0]))

        if not os.path.isdir(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        logger = getLogger(os.path.join(self.ckpt_dir, 'train.log'), log_file=True)
        
        # create data loaders for training and cross validation
        tr_loader = AudioLoader(self.tr_list, self.sample_rate, self.unit,
                                self.segment_size, self.segment_shift,
                                self.batch_size, self.buffer_size,
                                self.in_norm, mode='train')
        cv_loader = AudioLoader(self.cv_file, self.sample_rate, unit='utt',
                                segment_size=None, segment_shift=None,
                                batch_size=1, buffer_size=10,
                                in_norm=self.in_norm, mode='eval')

        # create a network
        net = Net()
        logger.info('Model summary:\n{}'.format(net))

        net = net.to(self.device)
        if len(self.gpu_ids) > 1:
            net = DataParallel(net, device_ids=self.gpu_ids)

        # calculate model size
        param_count = numParams(net)
        logger.info('Trainable parameter count: {:,d} -> {:.2f} MB\n'.format(param_count, param_count*32/8/(2**20)))

        # net feeder
        feeder = NetFeeder(self.device, self.win_size, self.hop_size)

        # training criterion and optimizer
        criterion = LossFunction()
        optimizer = Adam(net.parameters(), lr=self.lr, amsgrad=True)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=self.lr_decay_period, gamma=self.lr_decay_factor)
        
        # resume model if needed
        if self.resume_model:
            logger.info('Resuming model from {}'.format(self.resume_model))
            ckpt = CheckPoint()
            ckpt.load(self.resume_model, self.device)
            state_dict = {}
            for key in ckpt.net_state_dict:
                if len(self.gpu_ids) > 1:
                    state_dict['module.'+key] = ckpt.net_state_dict[key]
                else:
                    state_dict[key] = ckpt.net_state_dict[key]
            net.load_state_dict(state_dict)
            optimizer.load_state_dict(ckpt.optim_state_dict)
            ckpt_info = ckpt.ckpt_info
            logger.info('model info: epoch {}, iter {}, cv_loss - {:.4f}\n'.format(ckpt.ckpt_info['cur_epoch']+1,
                ckpt.ckpt_info['cur_iter']+1, ckpt.ckpt_info['cv_loss']))
        else:
            logger.info('Training from scratch...\n')
            ckpt_info = {'cur_epoch': 0,
                         'cur_iter': 0,
                         'tr_loss': None,
                         'cv_loss': None,
                         'best_loss': float('inf')}
        
        start_iter = 0
        # train model
        while ckpt_info['cur_epoch'] < self.max_n_epochs:
            accu_tr_loss = 0.
            accu_n_frames = 0
            net.train()
            for n_iter, egs in enumerate(tr_loader):
                n_iter += start_iter
                mix = egs['mix']
                sph = egs['sph']
                n_samples = egs['n_samples']

                mix = mix.to(self.device)
                sph = sph.to(self.device)
                n_samples = n_samples.to(self.device)

                n_frames = countFrames(n_samples, self.win_size, self.hop_size)

                start_time = timeit.default_timer()
                
                # prepare features and labels
                feat, lbl = feeder(mix, sph)
                loss_mask = lossMask(shape=lbl.shape, n_frames=n_frames, device=self.device)
                # forward + backward + optimize
                optimizer.zero_grad()
                with torch.enable_grad():
                    est = net(feat)
                loss = criterion(est, lbl, loss_mask, n_frames)
                loss.backward()
                if self.clip_norm >= 0.0:
                    clip_grad_norm_(net.parameters(), self.clip_norm)
                optimizer.step()
                # calculate loss
                running_loss = loss.data.item()
                accu_tr_loss += running_loss * sum(n_frames)
                accu_n_frames += sum(n_frames)

                end_time = timeit.default_timer()
                batch_time = end_time - start_time

                if self.time_log:
                    with open(self.time_log, 'a+') as f:
                        print('Epoch [{}/{}], Iter [{}], tr_loss = {:.4f} / {:.4f}, batch_time (s) = {:.4f}'.format(ckpt_info['cur_epoch']+1,
                            self.max_n_epochs, n_iter, running_loss, accu_tr_loss / accu_n_frames, batch_time), file=f)
                        f.flush()
                else:
                    print('Epoch [{}/{}], Iter [{}], tr_loss = {:.4f} / {:.4f}, batch_time (s) = {:.4f}'.format(ckpt_info['cur_epoch']+1,
                        self.max_n_epochs, n_iter, running_loss, accu_tr_loss / accu_n_frames, batch_time), flush=True)
 
        
                if (n_iter + 1) % self.logging_period == 0:
                    avg_tr_loss = accu_tr_loss / accu_n_frames
                    avg_cv_loss = self.validate(net, cv_loader, criterion, feeder)
                    net.train()
                
                    ckpt_info['cur_iter'] = n_iter
                    is_best = True if avg_cv_loss < ckpt_info['best_loss'] else False
                    ckpt_info['best_loss'] = avg_cv_loss if is_best else ckpt_info['best_loss']
                    latest_model = 'latest.pt'
                    best_model = 'best.pt'
                    ckpt_info['tr_loss'] = avg_tr_loss
                    ckpt_info['cv_loss'] = avg_cv_loss
                    if len(self.gpu_ids) > 1:
                        ckpt = CheckPoint(ckpt_info, net.module.state_dict(), optimizer.state_dict())
                    else:
                        ckpt = CheckPoint(ckpt_info, net.state_dict(), optimizer.state_dict())
                    logger.info('Saving checkpoint into {}'.format(os.path.join(self.ckpt_dir, latest_model)))
                    if is_best:
                        logger.info('Saving checkpoint into {}'.format(os.path.join(self.ckpt_dir, best_model)))
                    logger.info('Epoch [{}/{}], ( tr_loss: {:.4f} | cv_loss: {:.4f} )\n'.format(ckpt_info['cur_epoch']+1,
                        self.max_n_epochs, avg_tr_loss, avg_cv_loss))
                    
                    model_path = os.path.join(self.ckpt_dir, 'models')
                    if not os.path.isdir(model_path):
                        os.makedirs(model_path)

                    ckpt.save(os.path.join(model_path, latest_model),
                              is_best,
                              os.path.join(model_path, best_model))
                    
                    lossLog(os.path.join(self.ckpt_dir, self.loss_log), ckpt, self.logging_period)
            
                    accu_tr_loss = 0.
                    accu_n_frames = 0

                    if n_iter + 1 == self.tr_size // self.batch_size:
                        start_iter = 0
                        ckpt_info['cur_iter'] = 0
                        break
                    
            ckpt_info['cur_epoch'] += 1
            scheduler.step() # learning rate decay
        
        return

    def validate(self, net, cv_loader, criterion, feeder):
        accu_cv_loss = 0.
        accu_n_frames = 0

        if len(self.gpu_ids) > 1:
            net = net.module
        
        net.eval()
        for k, egs in enumerate(cv_loader):
            mix = egs['mix']
            sph = egs['sph']
            n_samples = egs['n_samples']

            mix = mix.to(self.device)
            sph = sph.to(self.device)
            n_samples = n_samples.to(self.device)
            
            n_frames = countFrames(n_samples, self.win_size, self.hop_size)

            feat, lbl = feeder(mix, sph)
            
            with torch.no_grad():
                loss_mask = lossMask(shape=lbl.shape, n_frames=n_frames, device=self.device)
                est = net(feat)
                loss = criterion(est, lbl, loss_mask, n_frames)

            accu_cv_loss += loss.data.item() * sum(n_frames)
            accu_n_frames += sum(n_frames)
                
        avg_cv_loss = accu_cv_loss / accu_n_frames
        return avg_cv_loss
                
    def test(self, args):
        with open(args.tt_list, 'r') as f:
            self.tt_list = [line.strip() for line in f.readlines()]
        self.model_file = args.model_file
        self.ckpt_dir = args.ckpt_dir
        self.est_path = args.est_path
        self.write_ideal = args.write_ideal
        self.gpu_ids = tuple(map(int, args.gpu_ids.split(',')))
        if len(self.gpu_ids) == 1 and self.gpu_ids[0] == -1:
            # cpu only
            self.device = torch.device('cpu')
        else:
            # gpu
            self.device = torch.device('cuda:{}'.format(self.gpu_ids[0]))

        if not os.path.isdir(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        logger = getLogger(os.path.join(self.ckpt_dir, 'test.log'), log_file=True)

        # create a network
        net = Net()
        logger.info('Model summary:\n{}'.format(net))

        net = net.to(self.device)

        # calculate model size
        param_count = numParams(net)
        logger.info('Trainable parameter count: {:,d} -> {:.2f} MB\n'.format(param_count, param_count*32/8/(2**20)))
        
        # training criterion and optimizer
        criterion = LossFunction()
        
        # net feeder
        feeder = NetFeeder(self.device, self.win_size, self.hop_size)
        
        # resynthesizer
        resynthesizer = Resynthesizer(self.device, self.win_size, self.hop_size)
        
        # load model
        logger.info('Loading model from {}'.format(self.model_file))
        ckpt = CheckPoint()
        ckpt.load(self.model_file, self.device)
        net.load_state_dict(ckpt.net_state_dict)
        logger.info('model info: epoch {}, iter {}, cv_loss - {:.4f}\n'.format(ckpt.ckpt_info['cur_epoch']+1,
            ckpt.ckpt_info['cur_iter']+1, ckpt.ckpt_info['cv_loss']))
        
        net.eval()
        for i in range(len(self.tt_list)):
            # create a data loader for testing
            tt_loader = AudioLoader(self.tt_list[i], self.sample_rate, unit='utt',
                                    segment_size=None, segment_shift=None,
                                    batch_size=1, buffer_size=10,
                                    in_norm=self.in_norm, mode='eval')
            logger.info('[{}/{}] Estimating on {}'.format(i+1, len(self.tt_list), self.tt_list[i]))

            est_subdir = os.path.join(self.est_path, self.tt_list[i].split('/')[-1].replace('.ex', ''))
            if not os.path.isdir(est_subdir):
                os.makedirs(est_subdir)
        
            accu_tt_loss = 0.
            accu_n_frames = 0        
            for k, egs in enumerate(tt_loader):
                mix = egs['mix']
                sph = egs['sph']
                n_samples = egs['n_samples']

                n_frames = countFrames(n_samples, self.win_size, self.hop_size)
                
                mix = mix.to(self.device)
                sph = sph.to(self.device)

                feat, lbl = feeder(mix, sph)

                with torch.no_grad():
                    loss_mask = lossMask(shape=lbl.shape, n_frames=n_frames, device=self.device)
                    est = net(feat)
                    loss = criterion(est, lbl, loss_mask, n_frames)

                accu_tt_loss += loss.data.item() * sum(n_frames)
                accu_n_frames += sum(n_frames)
                
                sph_idl = resynthesizer(lbl, mix)
                sph_est = resynthesizer(est, mix)
                
                # save estimates
                mix = mix[0].cpu().numpy()
                sph = sph[0].cpu().numpy()
                sph_est = sph_est[0].cpu().numpy()
                sph_idl = sph_idl[0].cpu().numpy()
                mix, sph, sph_est, sph_idl = wavNormalize(mix, sph, sph_est, sph_idl)
                sf.write(os.path.join(est_subdir, '{}_mix.wav'.format(k)), mix, self.sample_rate)
                sf.write(os.path.join(est_subdir, '{}_sph.wav'.format(k)), sph, self.sample_rate)
                sf.write(os.path.join(est_subdir, '{}_sph_est.wav'.format(k)), sph_est, self.sample_rate)
                if self.write_ideal:
                    sf.write(os.path.join(est_subdir, '{}_sph_idl.wav'.format(k)), sph_idl, self.sample_rate)

            avg_tt_loss = accu_tt_loss / accu_n_frames
            logger.info('loss: {:.4f}\n'.format(avg_tt_loss))

        return
