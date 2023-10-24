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


def weigh_row(row, documents, term, K, b, N, doclens, avg_doclen):
    frequencies = np.array(
        list(map(lambda doc: np.count_nonzero(doc == term), documents))
    )
    N = len(documents)
    ni = np.count_nonzero(list(map(lambda freq: freq > 0, frequencies)))
    weights = pd.DataFrame(frequencies).apply(
        weigh_term, args=(K, b, N, doclens, avg_doclen, ni), axis=1
    )
    return weights


def generate_bm_matrix(documents, terms, K, b):
    bm_matrix = pd.DataFrame(index=terms, columns=range(len(documents)))
    N = len(documents)
    doclens = list(map(lambda doc: sum(map(lambda word: len(word), doc)), documents))
    avg_doclen = np.mean(
        list(map(lambda doc: sum(map(lambda w: len(w), doc)), documents))
    )
    for term, row in bm_matrix.iterrows():
        bm_matrix.loc[term] = weigh_row(
            row, documents, term, K, b, N, doclens, avg_doclen
        )

    return bm_matrix


def rank(documents, query):
    return documents.loc[query].sum().sort_values(ascending=False)


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

    K = 1.2
    b = 0.75
    bm_matrix = generate_bm_matrix(tokens_list, terms, K, b)

    document_ranks = rank(bm_matrix, query)
    print(document_ranks)


if __name__ == "__main__":
    main()
