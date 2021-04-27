import torch
import torch.nn.functional as F

from utils.stft import STFT


class NetFeeder(object):
    def __init__(self, device, win_size=320, hop_size=160):
        self.eps = torch.finfo(torch.float32).eps
        self.stft = STFT(win_size, hop_size).to(device)

    def __call__(self, mix, sph):
        real_mix, imag_mix = self.stft.stft(mix)
        feat = torch.stack([real_mix, imag_mix], dim=1)
        
        real_sph, imag_sph = self.stft.stft(sph)
        lbl = torch.stack([real_sph, imag_sph], dim=1)

        return feat, lbl


class Resynthesizer(object):
    def __init__(self, device, win_size=320, hop_size=160):
        self.stft = STFT(win_size, hop_size).to(device)

    def __call__(self, est, mix):
        sph_est = self.stft.istft(est)
        sph_est = F.pad(sph_est, [0, mix.shape[1]-sph_est.shape[1]])

        return sph_est
