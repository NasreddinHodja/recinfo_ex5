#!/usr/bin/env python3

"""
Author: Tomás Bizet de Barros
DRE: 116183736
"""

import numpy as np
import pandas as pd
import re
import math


def tokenize(s, separators):
    pattern = "|".join(map(re.escape, separators))
    tokens = re.split(pattern, s)
    if tokens[-1] == "":
        tokens.pop()

    return np.array([token for token in tokens if token])


def normalize(s):
    return s.lower().strip()


def remove_stopwords(tokens_list, stopwords):
    return [
        np.array([token for token in tokens if token not in stopwords])
        for tokens in tokens_list
    ]


class BM25:
    def __init__(self, documents, terms):
        self.K = 1.2
        self.b = 0.75
        self.N = len(documents)
        self.documents = documents

        self.matrix = pd.DataFrame(index=range(len(documents)), columns=terms)
        self.doclens = list(
            map(lambda doc: sum(map(lambda word: len(word), doc)), documents)
        )
        self.avg_doclen = np.mean(
            list(map(lambda doc: sum(map(lambda w: len(w), doc)), documents))
        )

    def weigh_term(frequency, K, b, N, doclens, avg_doclen, ni):
        return (
            ((K + 1) * frequency.iloc[0])
            / (
                K * ((1 - b) + b * (doclens[frequency.name] / avg_doclen))
                + frequency.iloc[0]
            )
            * np.log((N - ni + 0.5) / (ni + 0.5) + 1)
            if frequency.iloc[0] > 0
            else 0
        )

    def weigh_col(self, term):
        frequencies = np.array(
            list(map(lambda doc: np.count_nonzero(doc == term), self.documents))
        )
        ni = np.count_nonzero(list(map(lambda freq: freq > 0, frequencies)))
        weights = pd.DataFrame(frequencies).apply(
            BM25.weigh_term,
            args=(self.K, self.b, self.N, self.doclens, self.avg_doclen, ni),
            axis=1,
        )
        return weights

    def generate_bm_matrix(self):
        for term in self.matrix:
            self.matrix[term] = self.weigh_col(term)

        return self.matrix

    def rank(self, query):
        return self.matrix[query].T.sum().sort_values(ascending=False)


def main():
    # documentos
    dictionary = np.array(
        [
            "O peã e o caval são pec de xadrez. O caval é o melhor do jog.",
            "A jog envolv a torr, o peã e o rei.",
            "O peã lac o boi",
            "Caval de rodei!",
            "Polic o jog no xadrez.",
            "xadrez peã caval torr",
        ]
    )
    stopwords = ["a", "o", "e", "é", "de", "do", "no", "são"]  # lista de stopwords
    query = "xadrez peã caval torr"  # consulta
    separators = [" ", ",", ".", "!", "?"]  # separadores para tokenizacao

    # normalize / tokenize
    normalized = np.array([normalize(s) for s in dictionary])
    tokens_list = np.array([tokenize(s, separators) for s in normalized], dtype=object)
    tokens_list = remove_stopwords(tokens_list, stopwords)

    terms = np.array(sorted(list(set([term for l in tokens_list for term in l]))))
    query = np.array(query.split())

    # K = 1.2
    # b = 0.75
    # bm_matrix = generate_bm_matrix(tokens_list, terms, K, b)

    bm25 = BM25(tokens_list, terms)
    bm25.generate_bm_matrix()
    print(bm25.matrix)

    document_ranks = bm25.rank(query)
    print(document_ranks)


if __name__ == "__main__":
    main()
