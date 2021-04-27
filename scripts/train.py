import argparse
import pprint

import torch

from utils.models import Model
from utils.utils import getLogger


logger = getLogger(__name__)


def main():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    # parse the configurations
    parser = argparse.ArgumentParser(description='Additioal configurations for training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu_ids',
                        type=str,
                        default='-1',
                        help='IDs of GPUs to use (please use `,` to split multiple IDs); -1 means CPU only')
    parser.add_argument('--tr_list',
                        type=str,
                        required=True,
                        help='Path to the list of training files')
    parser.add_argument('--cv_file',
                        type=str,
                        required=True,
                        help='Path to the cross validation file')
    parser.add_argument('--ckpt_dir',
                        type=str,
                        required=True,
                        help='Name of the directory to dump checkpoint')
    parser.add_argument('--unit',
                        type=str,
                        required=True,
                        help='Unit of sample, can be either `seg` or `utt`')
    parser.add_argument('--logging_period',
                        type=int,
                        default=1000,
                        help='Logging period (also the period of cross validation) represented by the number of iterations')
    parser.add_argument('--time_log',
                        type=str,
                        default='',
                        help='Log file for timing batch processing')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='Minibatch size')
    parser.add_argument('--buffer_size',
                        type=int,
                        default=32,
                        help='Buffer size')
    parser.add_argument('--segment_size',
                        type=float,
                        default=4.0,
                        help='Length of segments used for training (seconds)')
    parser.add_argument('--segment_shift',
                        type=float,
                        default=1.0,
                        help='Shift of segments used for training (seconds)')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='Initial learning rate for training')
    parser.add_argument('--lr_decay_factor',
                        type=float,
                        default=0.98,
                        help='Decaying factor of learning rate')
    parser.add_argument('--lr_decay_period',
                        type=int,
                        default=2,
                        help='Decaying period of learning rate (epochs)')
    parser.add_argument('--clip_norm',
                        type=float,
                        default=-1.0,
                        help='Gradient clipping (L2-norm)')
    parser.add_argument('--max_n_epochs',
                        type=int,
                        default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--loss_log',
                        type=str,
                        default='loss.txt',
                        help='Filename of the loss log')
    parser.add_argument('--resume_model',
                        type=str,
                        default='',
                        help='Existing model to resume training from')

    args = parser.parse_args()
    logger.info('Arguments in command:\n{}'.format(pprint.pformat(vars(args))))

    model = Model()
    model.train(args) 


if __name__ == '__main__':
    main()
