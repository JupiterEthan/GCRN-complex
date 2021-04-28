# Learning Complex Spectral Mapping with Gated Convolutional Recurrent Networks for Monaural Speech Enhancement

This repository provides an implementation of the gated convolutional recurrent network (GCRN) for monaural speech enhancement, developed in ["Learning complex spectral mapping with gated convolutional recurrent networks for monaural speech enhancement"](https://web.cse.ohio-state.edu/~wang.77/papers/Tan-Wang.taslp20.pdf), IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 28, pp. 380-390, 2020.

## Installation
The program is developed using Python 3.7.
Clone this repo, and install the dependencies:
```
git clone https://github.com/JupiterEthan/GCRN-complex.git
cd GCRN-complex
pip install -r requirements.txt
```

## Data preparation
To use this program, data and file lists need to be prepared. If configured correctly, the directory tree should look like this:

```
.
├── data
│   └── datasets
│       ├── cv
│       │   └── cv.ex
│       ├── tr
│       │   ├── tr_0.ex
│       │   ├── tr_1.ex
│       │   ├── tr_2.ex
│       │   ├── tr_3.ex
│       │   └── tr_4.ex
│       └── tt
│           ├── tt_snr0.ex
│           ├── tt_snr-5.ex
│           └── tt_snr5.ex
├── filelists
│   ├── tr_list.txt
│   └── tt_list.txt
├── README.md
├── requirements.txt
└── scripts
    ├── configs.py
    ├── measure.py
    ├── run_evaluate.sh
    ├── run_train.sh
    ├── test.py
    ├── train.py
    └── utils
        ├── criteria.py
        ├── data_utils.py
        ├── metrics.py
        ├── models.py
        ├── networks.py
        ├── pipeline_modules.py
        ├── stft.py
        └── utils.py
```

Follow these instructions:
1. Write your own scripts to prepare data for training, validation and testing. 
- For the training set, each example needs to be saved into an HDF5 file, which contains two HDF5 datasets, named ```mix``` and ```sph``` respectively. ```mix``` stores a noisy mixture utterance, ```sph``` the corresponding clean speech utterance.
    - Example:
      ```
      import h5py
    
      # some settings
      ...

      for idx in range(n_tr_ex): # n_tr_ex is the number of training examples 
          # generate a noisy mixture
          ...
          filename = 'tr_{}.ex'.format(idx)
          writer = h5py.File(os.path.join(filepath, filename), 'w')
          writer.create_dataset('mix', data=mix.astype(np.float32), shape=mix.shape, chunks=True)
          writer.create_dataset('sph', data=sph.astype(np.float32), shape=sph.shape, chunks=True)
          writer.close()
      ```
- For validation set, all examples need to be saved into a single HDF5 file, each of which is stored in a 


## How to cite
```
@article{tan2020learning,
  title={Learning complex spectral mapping with gated convolutional recurrent networks for monaural speech enhancement},
  author={Tan, Ke and Wang, DeLiang},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={28},
  pages={380--390},
  year={2020},
  publisher={IEEE}
}
```
