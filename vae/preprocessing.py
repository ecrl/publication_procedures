from typing import List, Iterable
import collections
import re

import numpy
import numpy as np


def floats_to_onehot(X: 'numpy.array') -> 'numpy.array':

    max_idxs = np.argmax(X, axis=2)
    result = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
    for i in range(len(max_idxs)):
        for j in range(len(max_idxs[0])):
            result[i][j][max_idxs[i][j]] = 1
    return result


def remove_slashes(smiles: str) -> str:

    return smiles.replace('\\', '').replace('/', '')


class SMILESTokenizer(object):

    def __init__(self, X: List[str], n_merges: int = 16):
        """
        Fits tokenizer to SMILES strings, taking into account subwords
        (frequently observed groups of characters, often functional groups)
        using byte pair encoding (BPE)

        Adapted from https://arxiv.org/pdf/1508.07909.pdf

        Args:
            X (List[str]): SMILES strings to fit tokenizer
            n_merges (int): number of merges (iterations of BPE)
        """

        _words = [' '.join(smi) for smi in X]

        self._best = []
        self._p = []

        for i in range(n_merges):

            pairs = collections.defaultdict(int)
            for word in _words:
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pairs[symbols[i], symbols[i + 1]] += 1
            best = max(pairs, key=pairs.get)

            bigram = re.escape(' '.join(best))
            p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')

            _new_words = []
            for word in _words:
                _new_words.append(p.sub(''.join(best), word))
            _words = _new_words

            self._best.append(best)
            self._p.append(p)

        self._vocab = {}
        for word in _words:
            chars = word.split(' ')
            for c in chars:
                if c not in self._vocab.keys():
                    self._vocab[c] = 1
                else:
                    self._vocab[c] += 1

    @property
    def vocab(self) -> List[str]:
        """
        Returns:
            List[str]: learned vocabulary of tokens
        """

        return list(self._vocab.keys())

    def tokenize(self, X: List[str]) -> List[List[str]]:
        """
        Tokenize SMILES strings using pre-trained BPE vocabulary

        Args:
            X (List[str]): SMILES strings to tokenize

        Returns:
            List[List[str]]: tokenized SMILES strings, each list element
                represents a SMILES string; each element is a list of "words"
                (subgroups defined by BPE) that make up each SMILES string
        """

        _words = [' '.join(smi) for smi in X]

        for i in range(len(self._p)):

            _new_words = []
            for word in _words:
                _new_words.append(
                    self._p[i].sub(''.join(self._best[i]), word)
                )
            _words = _new_words

        _words = [w.split(' ') for w in _words]
        return _words


class SMILESEncoder(SMILESTokenizer):

    def __init__(self, X: List[str], **kwargs):
        """
        First tokenizes using SMILESTokenizer, then uses tokens to create
        vector space for encoding SMILES strings (one-hot vectors)

        Args:
            X (List[str]): SMILES strings to train tokenizer/encoder
            max_tokens (int, optional): expected max number of tokens per
                SMILES string; okay to over-estimate, should be >= n_samples
                (default: 16)
            **kwargs: optional arguments passed to SMILESTokenizer
        """

        super().__init__(X, **kwargs)
        self._max_tokens = max([len(x) for x in X])
        vc = ['']
        vc.extend(self.vocab)
        self._trf_forward = dict((c, i) for i, c in enumerate(vc))
        self._trf_reverse = dict((i, c) for i, c in enumerate(vc))

    def encode(self, X: List[str]) -> 'numpy.array':
        """
        Encodes SMILES strings to one-hot vectors

        Args:
            X (List[str]): SMILES strings to encode

        Returns:
            numpy.array: encoded SMILES, shape (n_samples, max_tokens,
                len(token vocab))
        """

        X = self.tokenize(X)
        return self._encode(X)

    def _encode(self, tokenized_X: List[List[str]]) -> 'numpy.array':
        """
        Encodes tokenized SMILES strings to one-hot vectors

        Args:
            List[List[str]]: tokenized SMILES strings

        Returns:
            numpy.array: encoded SMILES, shape (n_samples, max_tokens,
                len(token vocab))
        """

        results = []
        for i, smi in enumerate(tokenized_X):
            _sample = [[0 for _ in range(len(self.vocab) + 1)]
                       for _ in range(self._max_tokens)]
            for s in range(len(_sample)):
                _sample[s][0] = 1
            for j, token in enumerate(smi):
                try:
                    _sample[j][self._trf_forward[token]] = 1
                    _sample[j][0] = 0
                except KeyError:
                    continue
            results.append(_sample)
        return np.asarray(results)

    def decode(self, X: 'numpy.array') -> List[str]:
        """
        Decodes vectorized SMILES strings to SMILES format

        Args:
            X (numpy.array): encoded SMILES, shape (n_samples, max_tokens,
                len(token vocab))

        Returns:
            List[str]: decoded SMILES strings
        """

        results = []
        for smi in X:
            smiles = ''
            smi = smi.argmax(axis=-1)
            for i in smi:
                smiles += self._trf_reverse[i]
            results.append(smiles)
        return results
