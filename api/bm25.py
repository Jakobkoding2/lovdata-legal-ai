import math
from typing import List

import numpy as np


class BM25Okapi:
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.corpus = corpus
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / self.corpus_size if self.corpus_size > 0 else 0.0
        self.k1 = k1
        self.b = b
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self._initialize()

    def _initialize(self) -> None:
        for document in self.corpus:
            frequencies = {}
            for word in document:
                frequencies[word] = frequencies.get(word, 0) + 1
            self.doc_freqs.append(frequencies)
            self.doc_len.append(len(document))
            for word in frequencies.keys():
                self.idf[word] = self.idf.get(word, 0) + 1

        for word, freq in self.idf.items():
            self.idf[word] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query: List[str]) -> np.ndarray:
        scores = np.zeros(self.corpus_size, dtype=float)
        if not query or self.corpus_size == 0 or self.avgdl == 0:
            return scores

        for token in query:
            if token not in self.idf:
                continue
            idf = self.idf[token]
            for idx, doc_freq in enumerate(self.doc_freqs):
                freq = doc_freq.get(token, 0)
                if freq == 0:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * self.doc_len[idx] / self.avgdl)
                score = idf * (freq * (self.k1 + 1)) / denom
                scores[idx] += score
        return scores

    def get_top_n(self, query: List[str], documents: List[str], n: int = 5) -> List[str]:
        scores = self.get_scores(query)
        top_indices = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_indices]
