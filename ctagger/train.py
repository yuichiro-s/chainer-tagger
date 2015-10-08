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
    batches = util.create_batches(corpus, vocab_word, vocab_tag, args.batch, gpu=args.gpu, shuffle=not args.no_shuffle)

    # set up tagger
    tagger = nn.NnTagger(vocab_size=vocab_word.size(), emb_dim=emb_dim, window=args.window, hidden_dim=args.hidden,
                         tag_num=args.tag, init_emb=init_emb)

    # set up optimizer
    optim_name = args.optim[0]
    assert not args.decay_lr or optim_name == 'SGD', 'learning-rate decay is only supported for SGD'
    optim_args = map(float, args.optim[1:])
    optimizer = getattr(O, optim_name)(*optim_args)
    optimizer.setup(tagger.model)

    initial_lr = None
    if args.decay_lr:
        initial_lr = optimizer.lr

    # set up GPU
    if args.gpu >= 0:
        tagger.model.to_gpu(args.gpu)

    # create directory
    os.makedirs(args.model)
    vocab_word.save(os.path.join(args.model, 'vocab_word'))
    vocab_tag.save(os.path.join(args.model, 'vocab_tag'))

    # training loop
    for n in range(args.epoch):
        # decay learning rate
        if args.decay_lr:
            optimizer.lr = initial_lr / (n + 1)
            print >> sys.stderr, 'Learning rate set to: {}'.format(optimizer.lr)

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
