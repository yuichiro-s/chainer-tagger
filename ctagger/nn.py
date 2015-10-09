import cPickle as pickle

import chainer
import chainer.functions as F
import numpy as np


class NnTagger(object):
    """Neural Network POS tagger."""

    def __init__(self, vocab_size = 10000, emb_dim=100, window=5, hidden_dim=300, tag_num=45, init_emb=None):
        """
        :param emb_dim: dimension of word embeddings
        :param window: window size
        :param hidden_dim: dimension of hidden layer
        :param tag_num: number of tags
        """

        assert window % 2 == 1, 'Window size must be odd'

        self.emb_dim = emb_dim
        self.window = window
        self.hidden_dim = hidden_dim
        self.tag_num = tag_num

        self.model = chainer.FunctionSet(
            emb=F.EmbedID(vocab_size, emb_dim),
            conv=F.Convolution2D(1, hidden_dim,
                                 ksize=(window, emb_dim),
                                 stride=(1, emb_dim),
                                 pad=(window/2, 0)),
            linear=F.Linear(hidden_dim, tag_num),
        )

        # initialize embeddings
        if init_emb is not None:
            self.model.emb.W = init_emb

    def _forward(self, x_data, volatile):
        """
        Forward computation.

        :param batch: numpy array (batch dimension is the second dimension)
        :param t_data: target data
        :return: variable of size (batch * sentence length, number of tags)
        """
        batch_size = x_data.shape[0]
        x = chainer.Variable(x_data, volatile=volatile)
        emb = self.model.emb(x)
        emb_reshape = F.reshape(emb, (1, 1, batch_size, self.emb_dim))

        h = self.model.conv(emb_reshape)
        h_transpose = F.swapaxes(h, 1, 2)  # TODO: maybe inefficient
        h_reshape = F.reshape(h_transpose, (batch_size, self.hidden_dim))

        y = self.model.linear(F.relu(h_reshape))
        return y

    def forward_train(self, x_data, t_data):
        y = self._forward(x_data, volatile=False)
        t = chainer.Variable(t_data)
        # TODO: Currently, tags of paddings are also predicted. They must be excluded from the loss.
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    def forward_test(self, x_data):
        y = self._forward(x_data, volatile=True)
        return y

    def save(self, path):
        with open(path, 'wb') as f:
            p = self.model.parameters[0]
            device = None
            if hasattr(p, 'device'):
                device = p.device
                self.model.to_cpu()

            pickle.dump(self, f)

            if device is not None:
                self.model.to_gpu(device=device)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

