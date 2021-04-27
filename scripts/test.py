import argparse
import pprint

import torch

from utils.models import Model
from utils.utils import getLogger


logger = getLogger(__name__)


def main():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    # parse the configuarations
    parser = argparse.ArgumentParser(description='Additioal configurations for testing',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu_ids',
                        type=str,
                        default='-1',
                        help='IDs of GPUs to use (please use `,` to split multiple IDs); -1 means CPU only')
    parser.add_argument('--tt_list',
                        type=str,
                        required=True,
                        help='Path to the list of testing files')
    parser.add_argument('--ckpt_dir',
                        type=str,
                        required=True,
                        help='Name of the directory to write log')
    parser.add_argument('--model_file',
                        type=str,
                        required=True,
                        help='Path to the model file')
    parser.add_argument('--est_path',
                        type=str,
                        default='../data/estimates',
                        help='Path to dump estimates')
    parser.add_argument('--write_ideal',
                        default=False,
                        action='store_true',
                        help='Whether to write ideal signals (the speech signals resynthesized from the ideal training targets; ex. for time-domain enhancement, it is the same as clean speech)')

    args = parser.parse_args()
    logger.info('Arguments in command:\n{}'.format(pprint.pformat(vars(args))))

    model = Model()
    model.test(args)
    

if __name__ == '__main__':
    main()
