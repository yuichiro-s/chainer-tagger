from collections import defaultdict


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

    def get_word(self, id):
        return self.i2w[id]

    def size(self):
        return len(self.i2w)


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

