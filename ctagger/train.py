#!/usr/bin/env python

from . import nn
from . import util

import sys
import os

import chainer.optimizers as O


def train(args):
    # load pre-trained embeddings
    vocab_word = None
    init_emb = None
    if args.init_emb:
        print >> sys.stderr, 'Loading embeddings...'
        assert args.init_emb_words
        init_emb, vocab_word = util.load_init_emb(args.init_emb, args.init_emb_words)
        emb_dim = init_emb.shape[1]
    else:
        emb_dim = args.emb

    print >> sys.stderr, 'Loading data...'
    corpus, vocab_word_tmp, vocab_tag = util.load_conll(args.data, args.vocab)
    if vocab_word is None:
        vocab_word = vocab_word_tmp

    # create batches
    print >> sys.stderr, 'Creating batches...'
    batches = util.create_batches(corpus, vocab_word, vocab_tag, args.batch, shuffle=not args.no_shuffle)

    # set up tagger
    tagger = nn.NnTagger(vocab_size=vocab_word.size(), emb_dim=emb_dim, window=args.window, hidden_dim=args.hidden,
                         tag_num=args.tag, init_emb=init_emb)

    # set up optimizer
    optimizer = O.Adam()
    optimizer.setup(tagger.model)

    # set up GPU
    if args.gpu >= 0:
        tagger.model.to_gpu(args.gpu)
        # TODO: move data to GPU

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
    parser.add_argument('--init-emb', default=None, help='initial embedding file (word2vec output)')
    parser.add_argument('--init-emb-words', default=None, help='corresponding words of initial embedding file')
    parser.add_argument('--no-shuffle', action='store_true', default=False, help='don\'t shuffle training data')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU ID (-1 to use CPU)')

    train(parser.parse_args())
