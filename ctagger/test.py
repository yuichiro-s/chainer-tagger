from . import nn
from . import util

import sys
import os

import numpy as np


def test(args):
    print >> sys.stderr, 'Loading model...'
    tagger = nn.NnTagger.load(args.model)

    print >> sys.stderr, 'Loading vocabulary...'
    model_dir_path = os.path.dirname(args.model)
    vocab_word_path = os.path.join(model_dir_path, 'vocab_word')
    vocab_tag_path = os.path.join(model_dir_path, 'vocab_tag')
    vocab_word = util.Vocab.load(vocab_word_path)
    vocab_tag = util.Vocab.load(vocab_tag_path)

    print >> sys.stderr, 'Loading data...'
    corpus = util.load_conll(args.data, 0)[0]
    corpus_size = len(corpus)

    print >> sys.stderr, 'Creating batches...'
    batches = util.create_batches(corpus, vocab_word, vocab_tag, batch_size=128)

    # main loop
    total = 0
    correct = 0
    processed_num = 0
    for i, (x_data, t_data) in enumerate(batches):
        processed_num += x_data.shape[0]
        pred = tagger.forward_test(x_data)
        predicted_tags = np.argmax(pred.data, axis=2)
        for t, pred_tag in zip(t_data.flatten(), predicted_tags.flatten()):
            if t == pred_tag:
                correct += 1
            total += 1
        print >> sys.stderr, 'Processed {}/{} [{:.2%}]'.format(
            processed_num, corpus_size, float(processed_num) / corpus_size)

    # report result
    print '{:.2%}%'.format(float(processed_num, corpus_size))
    print correct
    print total


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train NN tagger.')

    parser.add_argument('data', help='path to test data')
    parser.add_argument('model', help='destination of model')

    test(parser.parse_args())