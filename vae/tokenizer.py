from typing import List
import collections
import re

import torch
import torch as T


class SMILESTokenizer(object):

    def __init__(self):

        self._vocab = {}
        self._n_bits_per_word = 8

    def fit(self, X: List[str], n_merges: int = 4):
        '''
        Adapted from https://arxiv.org/pdf/1508.07909.pdf
        '''

        _words = [' '.join(smi) for smi in X]

        for i in range(n_merges):

            pairs = collections.defaultdict(int)
            for word in _words:
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pairs[symbols[i], symbols[i + 1]] += 1

            best = max(pairs, key=pairs.get)

            _new_words = []
            bigram = re.escape(' '.join(best))
            p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
            for word in _words:
                _new_words.append(p.sub(''.join(best), word))

            _words = _new_words

        self._vocab = {}
        for word in _words:
            chars = word.split(' ')
            for c in chars:
                if c not in self._vocab.keys():
                    self._vocab[c] = 1
                else:
                    self._vocab[c] += 1

    def fit_transform(self, X: List[str], n_merges: int = 4) -> 'torch.tensor':

        self.fit(X, n_merges)
        return self.transform(X)

    def transform(self, X: List[str]) -> 'torch.tensor':

        trf_vals = []
        for smi in X:
            trf_vals.append(self._tokenize(smi))
        return trf_vals  # TODO: to torch.tensor

    def inverse_transform(self, X: 'torch.tensor') -> List[str]:

        return

    def _tokenize(self, smiles: str):

        return  # TODO: this next
