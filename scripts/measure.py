import argparse
import pprint

import torch

from utils.metrics import Metric
from utils.utils import getLogger


logger = getLogger(__name__)


def main():
    # parse the configurations
    parser = argparse.ArgumentParser(description='Additioal configurations for measurement',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--metric',
                        type=str,
                        required=True,
                        help='Name of the evaluation metric')    
    parser.add_argument('--tt_list',
                        type=str,
                        default='',
                        help='Path to the list of testing files')
    parser.add_argument('--ckpt_dir',
                        type=str,
                        required=True,
                        help='Name of the directory to write log')
    parser.add_argument('--est_path',
                        type=str,
                        default='../data/estimates',
                        help='Path to saved estimates')

    args = parser.parse_args()
    logger.info('Arguments in command:\n{}'.format(pprint.pformat(vars(args))))
    
    metric = Metric(args)
    metric.evaluate()


if __name__ == '__main__':
    main()
