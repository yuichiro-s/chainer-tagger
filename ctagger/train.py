#!/usr/bin/env python

from . import nn
from . import util

import sys
import os

import chainer.optimizers as O



def train(args):
    tagger = nn.NnTagger()

    print >> sys.stderr, 'Loading data...'
    corpus, vocab_word, vocab_tag = util.load_conll(args.data)

    # create batches
    batches = util.create_batches(corpus, vocab_word, vocab_tag, args.batch, args.vocab)

    # set up optimizer
    optimizer = O.Adam()
    optimizer.setup(tagger.model)

    # set up GPU
    if args.gpu >= 0:
        tagger.model.to_gpu(args.gpu)

    # create directory
    os.makedirs(args.model)
    vocab_word.save(os.path.join(args.model, 'vocab_word'))
    vocab_tag.save(os.path.join(args.model, 'vocab_tag'))

    # training loop
    for n in range(args.epoch):
        for i, (x_data, t_data) in enumerate(batches):
            batch_size, length = x_data.shape

            optimizer.zero_grads()
            loss, acc = tagger.forward_train(x_data, t_data)
            loss.backward()
            optimizer.update()

            print >> sys.stderr, 'Epoch {}\tBatch {}\tloss:\t{}\tacc:\t{}\tsize:\t{}\tlen:\t{}'.format(
                n + 1, i + 1, loss.data, acc.data, batch_size, length)

        # save current model
        dest_path = os.path.join(args.model, 'epoch' + str(n + 1))
        tagger.save(dest_path)


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
    parser.add_argument('--gpu', type=int, default=-1, help='GPU ID (-1 to use CPU)')

    train(parser.parse_args())