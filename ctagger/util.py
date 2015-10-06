from collections import defaultdict
import random

import numpy as np


EOS = u'<EOS>'
UNK = u'<UNK>'


class Vocab(object):
    """Mapping between words and IDs."""

    def __init__(self):
        self.i2w = []
        self.w2i = {}

    def add_word(self, word):
        assert isinstance(word, unicode)
        if word not in self.w2i:
            new_id = self.size()
            self.i2w.append(word)
            self.w2i[word] = new_id

    def get_id(self, word):
        assert isinstance(word, unicode)
        return self.w2i.get(word)

    def get_word(self, w_id):
        return self.i2w[w_id]

    def size(self):
        return len(self.i2w)

    def save(self, path):
        with open(path, 'w') as f:
            for i, w in enumerate(self.i2w):
                print >> f, str(i) + '\t' + w.encode('utf-8')

    @classmethod
    def load(cls, path):
        vocab = Vocab()
        with open(path) as f:
            for line in f:
                w = line.split('\t')[1].decode('utf-8')
                vocab.add_word(w)
        return vocab


def load_conll(path, file_encoding='utf-8'):
    """Load CoNLL-format file.
    :return triple of corpus (pairs of words and tags), word vocabulary, tag vocabulary"""

    corpus = []
    word_freqs = defaultdict(int)

    vocab_word = Vocab()
    vocab_tag = Vocab()
    vocab_word.add_word(EOS)
    vocab_word.add_word(UNK)

    with open(path) as f:
        wts = []
        for line in f:
            es = line.rstrip().split('\t')
            if len(es) == 10:
                word = es[1].decode(file_encoding)
                tag = es[4].decode(file_encoding)
                vocab_tag.add_word(tag)
                wt = (word, tag)
                wts.append(wt)
                word_freqs[word] += 1
            else:
                # reached end of sentence
                corpus.append(wts)
                wts = []
        if wts:
            corpus.append(wts)

    for w, _ in sorted(word_freqs.items(), key=lambda (k, v): -v):
        vocab_word.add_word(w)

    return corpus, vocab_word, vocab_tag


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


def create_batches(corpus, vocab_word, vocab_tag, batch_size, vocab_size, shuffle=False):
    # convert to IDs
    id_corpus = []
    for sen in corpus:
        w_ids = []
        t_ids = []
        for w, t in sen:
            w_id = vocab_word.get_id(w)
            t_id = vocab_tag.get_id(t)
            if w_id >= vocab_size:
                w_id = vocab_word.get_id(UNK)
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
    if shuffle:
        random.shuffle(batches)

    # convert to numpy arrays
    batches = map(lambda batch: map(lambda arr: np.asarray(arr, dtype=np.int32), zip(*batch)), batches)

    return batches
