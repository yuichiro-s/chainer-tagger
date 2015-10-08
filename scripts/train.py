#!/usr/bin/env python

from ctagger import train


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train NN tagger.')

    parser.add_argument('data', help='path to training data')
    parser.add_argument('model', help='destination of model')

    # NN architecture
    parser.add_argument('--vocab', type=int, default=10000, help='vocabulary size')
    parser.add_argument('--emb', type=int, default=100, help='dimension of embeddings')
    parser.add_argument('--window', type=int, default=5, help='window size for convolution')
    parser.add_argument('--hidden', type=int, default=300, help='dimension of hidden layer')
    parser.add_argument('--tag', type=int, default=45, help='number of tags')

    # training options
    parser.add_argument('--batch', type=int, default=128, help='batch size')
    parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--init-emb', default=None, help='initial embedding file (word2vec output)')
    parser.add_argument('--init-emb-words', default=None, help='corresponding words of initial embedding file')
    parser.add_argument('--optim', nargs='+', default=['SGD', '0.0075'], help='optimization method supported by chainer (optional arguments can be omitted)')
    parser.add_argument('--decay-lr', action='store_true', default=False, help='divide learning rate of SGD by epoch number')
    parser.add_argument('--no-shuffle', action='store_true', default=False, help='don\'t shuffle training data')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU ID (-1 to use CPU)')

    train.train(parser.parse_args())