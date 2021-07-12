from typing import List
import collections
import re


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


class SMILESEmbedder(object):

    def __init__(self):

        return
