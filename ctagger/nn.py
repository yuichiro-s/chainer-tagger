import cPickle as pickle

import chainer
import chainer.functions as F


class NnTagger(object):
    """Neural Network POS tagger."""

    def __init__(self, vocab_size = 10000, emb_dim=100, window=5, hidden_dim=300, tag_num=45):
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
                                 ksize=(emb_dim * window, 1),
                                 stride=(emb_dim, 1),
                                 pad=(window / 2 * emb_dim, 0)),
            linear=F.Linear(hidden_dim, tag_num),
        )

    def _forward(self, x_data, volatile):
        """
        Forward computation.

        :param batch: numpy array (batch dimension is the second dimension)
        :param t_data: target data
        :return: variable of size (batch * sentence length, number of tags)
        """
        batch_size, length = x_data.shape
        x = chainer.Variable(x_data.flatten(), volatile=volatile)
        emb = self.model.emb(x)
        emb_reshape = F.reshape(emb, (batch_size, 1, self.emb_dim * length, 1))

        h = self.model.conv(emb_reshape)
        h_transpose = F.transpose(h, (0, 2, 1, 3))  # TODO: maybe inefficient
        h_reshape = F.reshape(h_transpose, (batch_size * length, self.hidden_dim))

        y = F.relu(self.model.linear(h_reshape))
        return y

    def forward_train(self, x_data, t_data):
        y = self._forward(x_data, volatile=False)
        t = chainer.Variable(t_data.flatten())
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    def forward_test(self, x_data):
        y = self._forward(x_data, volatile=True)
        batch_size, length = x_data.shape
        pred = F.softmax(y)
        t = F.reshape(pred, (batch_size, length, self.tag_num))
        return t

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
        with open(path) as f:
            return pickle.load(f)

