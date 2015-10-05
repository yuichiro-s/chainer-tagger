import numpy as np

from ctagger.train import *


def test_split_into_batches():
    lst = [
        [1],
        [2, 2],
        [2, 2],
        [2, 2],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [4, 4, 4],
        [4, 4, 4],
    ]

    batches = split_into_batches(lst, 2, len)
    assert len(batches) == 6
