import numpy as np

from ctagger.nn import *


def test_forward_test():
    nn = NnTagger(window=3)
    x_data = np.arange(50).reshape((5, 10)).astype(np.int32)
    y_data = nn.forward_test(x_data)
    assert y_data.data.shape == (5, 10, 45)

