import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy


class STFT(nn.Module):
    def __init__(self, win_size=320, hop_size=160, requires_grad=False):
        super(STFT, self).__init__()

        self.win_size = win_size
        self.hop_size = hop_size
        self.n_overlap = self.win_size // self.hop_size
        self.requires_grad = requires_grad

        win = torch.from_numpy(scipy.hamming(self.win_size).astype(np.float32))
        win = F.relu(win)
        win = nn.Parameter(data=win, requires_grad=self.requires_grad)
        self.register_parameter('win', win)

        fourier_basis = np.fft.fft(np.eye(self.win_size))
        fourier_basis_r = np.real(fourier_basis).astype(np.float32)
        fourier_basis_i = np.imag(fourier_basis).astype(np.float32)

        self.register_buffer('fourier_basis_r', torch.from_numpy(fourier_basis_r))
        self.register_buffer('fourier_basis_i', torch.from_numpy(fourier_basis_i))

        idx = torch.tensor(range(self.win_size//2-1, 0, -1), dtype=torch.long)
        self.register_buffer('idx', idx)

        self.eps = torch.finfo(torch.float32).eps

    def kernel_fw(self):
        fourier_basis_r = torch.matmul(self.fourier_basis_r, torch.diag(self.win))
        fourier_basis_i = torch.matmul(self.fourier_basis_i, torch.diag(self.win))

        fourier_basis = torch.stack([fourier_basis_r, fourier_basis_i], dim=-1)
        forward_basis = fourier_basis.unsqueeze(dim=1)

        return forward_basis

    def kernel_bw(self):
        inv_fourier_basis_r = self.fourier_basis_r / self.win_size
        inv_fourier_basis_i = -self.fourier_basis_i / self.win_size

        inv_fourier_basis = torch.stack([inv_fourier_basis_r, inv_fourier_basis_i], dim=-1)
        backward_basis = inv_fourier_basis.unsqueeze(dim=1)
        return backward_basis

    def window(self, n_frames):
        assert n_frames >= 2
        seg = sum([self.win[i*self.hop_size:(i+1)*self.hop_size] for i in range(self.n_overlap)])
        seg = seg.unsqueeze(dim=-1).expand((self.hop_size, n_frames-self.n_overlap+1))
        window = seg.contiguous().view(-1).contiguous()

        return window

    def stft(self, sig):
        batch_size = sig.shape[0]
        n_samples = sig.shape[1]

        cutoff = self.win_size // 2 + 1

        sig = sig.view(batch_size, 1, n_samples)
        kernel = self.kernel_fw()
        kernel_r = kernel[...,0]
        kernel_i = kernel[...,1]
        spec_r = F.conv1d(sig,
                          kernel_r[:cutoff],
                          stride=self.hop_size,
                          padding=self.win_size-self.hop_size)
        spec_i = F.conv1d(sig,
                          kernel_i[:cutoff],
                          stride=self.hop_size,
                          padding=self.win_size-self.hop_size)
        spec_r = spec_r.transpose(-1, -2).contiguous()
        spec_i = spec_i.transpose(-1, -2).contiguous()

        mag = torch.sqrt(spec_r**2 + spec_i**2)
        pha = torch.atan2(spec_i.data, spec_r.data)

        return spec_r, spec_i

    def istft(self, x):
        spec_r = x[:,0,:,:]
        spec_i = x[:,1,:,:]

        n_frames = spec_r.shape[1]

        spec_r = torch.cat([spec_r, spec_r.index_select(dim=-1, index=self.idx)], dim=-1)
        spec_i = torch.cat([spec_i, -spec_i.index_select(dim=-1, index=self.idx)], dim=-1)
        spec_r = spec_r.transpose(-1, -2).contiguous()
        spec_i = spec_i.transpose(-1, -2).contiguous()

        kernel = self.kernel_bw()
        kernel_r = kernel[...,0].transpose(0, -1)
        kernel_i = kernel[...,1].transpose(0, -1)

        sig = F.conv_transpose1d(spec_r,
                                 kernel_r,
                                 stride=self.hop_size,
                                 padding=self.win_size-self.hop_size) \
            - F.conv_transpose1d(spec_i,
                                 kernel_i,
                                 stride=self.hop_size,
                                 padding=self.win_size-self.hop_size)
        sig = sig.squeeze(dim=1)

        window = self.window(n_frames)
        sig = sig / (window + self.eps)

        return sig
