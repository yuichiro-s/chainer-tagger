#!/usr/bin/env python

from . import nn
from . import util

import sys
import os
import random

import numpy as np
import chainer.optimizers as O


def split_into_batches(corpus, batch_size, length_func=lambda t: len(t[0])):
    batches = []
    batch = []
    last_len = 0
    for sen in corpus:
        current_len = length_func(sen)
        if (last_len != current_len and len(batch) > 0) or len(batch) == batch_size:
            # next batch
            batches.append(batch)
            batch = []
        last_len = current_len
        batch.append(sen)
    if batch:
        batches.append(batch)
    return batches


def create_batches(corpus, vocab_word, vocab_tag, batch_size, vocab_size):
    # convert to IDs
    id_corpus = []
    for sen in corpus:

        w_ids = []
        t_ids = []
        for w, t in sen:
            w_id = vocab_word.get_id(w)
            t_id = vocab_tag.get_id(t)
            if w_id >= vocab_size:
                w_id = vocab_word.get_id(util.UNK)
            assert w_id is not None
            assert t_id is not None
            w_ids.append(w_id)
            t_ids.append(t_id)
        id_corpus.append((w_ids, t_ids))

    # sort by lengths
    id_corpus.sort(key=lambda w_t: len(w_t[0]))

    # split into batches
    batches = split_into_batches(id_corpus, batch_size)

    # shuffle batches
    random.shuffle(batches)

    # convert to numpy arrays
    batches = map(lambda batch: map(lambda arr: np.asarray(arr, dtype=np.int32), zip(*batch)), batches)

    return batches


def train(args):
    tagger = nn.NnTagger()

    print >> sys.stderr, 'Loading data...'
    corpus, vocab_word, vocab_tag = util.load_conll(args.data)

    # create batches
    batches = create_batches(corpus, vocab_word, vocab_tag, args.batch, args.vocab)

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