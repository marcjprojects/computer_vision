import json

import numpy as np

def check_results(ious):
    solution = np.load('data/exercise1_check.npy')
    assert (ious == solution).sum() == 40, 'The iou calculation is wrong!'
    print('Congrats, the iou calculation is correct')

