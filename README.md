# Learning Complex Spectral Mapping with Gated Convolutional Recurrent Networks for Monaural Speech Enhancement

This repository provides an implementation of the gated convolutional recurrent network (GCRN) for monaural speech enhancement, developed in ["Learning complex spectral mapping with gated convolutional recurrent networks for monaural speech enhancement"](https://web.cse.ohio-state.edu/~wang.77/papers/Tan-Wang.taslp20.pdf), IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 28, pp. 380-390, 2020. The paper proposed to perform complex spectral mapping with the GCRN, which is an extension of the CRN designed in ["A convolutional recurrent neural network for real-time speech enhancement"](https://web.cse.ohio-state.edu/~wang.77/papers/Tan-Wang1.interspeech18.pdf), Proceedings of Interspeech, pp. 3229-3233, 2018. An implementation of the original CRN is provided [here](https://github.com/JupiterEthan/CRN-causal).

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
├── examples
│   └── filelists
│       ├── tr_list.txt
│       └── tt_list.txt
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
You will find that some files above are missing in your directory tree. Those are for you to prepare. Don't worry. Follow these instructions:
1. Write your own scripts to prepare data for training, validation and testing. 
    - For the training set, each example needs to be saved into an HDF5 file, which contains two HDF5 datasets, named ```mix``` and ```sph``` respectively. ```mix``` stores a noisy mixture utterance, ```sph``` the corresponding clean speech utterance.
        - Example code:
          ```
          import os

          import h5py
          import numpy as np
   

          # some settings
          ...
          rms = 1.0

          for idx in range(n_tr_ex): # n_tr_ex is the number of training examples 
              # generate a noisy mixture
              ...
              mix = sph + noi
              # normalize
              c = rms * np.sqrt(mix.size / np.sum(mix**2))
              mix *= c
              sph *= c

              filename = 'tr_{}.ex'.format(idx)
              writer = h5py.File(os.path.join(filepath, filename), 'w')
              writer.create_dataset('mix', data=mix.astype(np.float32), shape=mix.shape, chunks=True)
              writer.create_dataset('sph', data=sph.astype(np.float32), shape=sph.shape, chunks=True)
              writer.close()
          ```
    - For the validation set, all examples need to be saved into a single HDF5 file, each of which is stored in a HDF5 group. Each group contains two HDF5 datasets, one named ```mix``` and the other named ```sph```.
        - Example code:
          ```
          import os

          import h5py
          import numpy as np


          # some settings
          ...
          rms = 1.0
          
          filename = 'cv.ex'
          writer = h5py.File(os.path.join(filepath, filename), 'w')
          for idx in range(n_cv_ex):
              # generate a noisy mixture
              ...
              mix = sph + noi
              # normalize
              c = rms * np.sqrt(mix.size / np.sum(mix**2))
              mix *= c
              sph *= c

              writer_grp = writer.create_group(str(count))
              writer_grp.create_dataset('mix', data=mix.astype(np.float32), shape=mix.shape, chunks=True)
              writer_grp.create_dataset('sph', data=sph.astype(np.float32), shape=sph.shape, chunks=True)
          writer.close()
          ```
    
    - For the test set(s), all examples (in each condition) need to be saved into a single HDF5 file, each of which is stored in a HDF5 group. Each group contains two HDF5 datasets, one named ```mix``` and the other named ```sph```.
        - Example code:
          ```
          import os

          import h5py
          import numpy as np


          # some settings
          ...
          rms = 1.0
          
          filename = 'tt_snr-5.ex'
          writer = h5py.File(os.path.join(filepath, filename), 'w')
          for idx in range(n_cv_ex):
              # generate a noisy mixture
              ...
              mix = sph + noi
              # normalize
              c = rms * np.sqrt(mix.size / np.sum(mix**2))
              mix *= c
              sph *= c

              writer_grp = writer.create_group(str(count))
              writer_grp.create_dataset('mix', data=mix.astype(np.float32), shape=mix.shape, chunks=True)
              writer_grp.create_dataset('sph', data=sph.astype(np.float32), shape=sph.shape, chunks=True)
          writer.close()
          ```
    - In the example code above, the root mean square power of the mixture is normalized to 1. The same scaling factor is applied to clean speech.
2. Generate the file lists for training and test sets, and save them into a folder named ```filelists```. See [examples/filelists](examples/filelists) for the examples.


## How to run
1. Change the directory: ```cd scripts```. Remember that this is your working directory. All paths and commands below are relative to it.
2. Check ```utils/networks.py``` for the GCRN configurations. By default, ```G=2``` (see the original paper) is used for LSTM grouping.
3. Train the model: ```./run_train.sh```. By default, a directory named ```exp``` will be automatically generated. Two model files will be generated under ```exp/models/```: ```latest.pt```(the model from the latest checkpoint) and ```best.pt```(the model that performs best on the validation set by far). ```latest.pt``` can be used to resume training if interrupted, and ```best.pt``` is typically used for testing. You can check the loss values in ```exp/loss.txt```.
4. Evaluate the model: ```./run_evaluate.sh```. WAV files will be generated under ```../data/estimates```. STOI, PESQ and SNR results will be written into three files under ```exp```: ```stoi_scores.log```, ```pesq_scores.log``` and ```snr_scores.log```.


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
