#!/usr/bin/env python3

"""
Author: Tomás Bizet de Barros
DRE: 116183736
"""

import numpy as np
import pandas as pd
import re


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


class TFIDF:
    def __init__(self, documents, terms):
        self.documents = documents
        self.terms = terms

        self.matrix = pd.DataFrame(index=terms, columns=range(len(documents)))
        self.N = len(documents)

    @staticmethod
    def weigh_term(frequency, frequency_in_collection, N):
        return (
            1 + np.log2(frequency) * np.log2(N / frequency_in_collection)
            if frequency > 0
            else 0
        )

    def weigh_row(self, term):
        frequencies = np.array(
            list(map(lambda doc: np.count_nonzero(doc == term), self.documents))
        )
        frequency_in_collection = np.count_nonzero(
            np.concatenate(self.documents) == term
        )
        weights = pd.Series(frequencies).apply(
            TFIDF.weigh_term, args=(frequency_in_collection, self.N)
        )
        return weights

    def generate_matrix(self):
        for term, _ in self.matrix.iterrows():
            self.matrix.loc[term] = self.weigh_row(term)

        return self.matrix

    @staticmethod
    def similarity(document, query):
        return document.dot(query) / (np.linalg.norm(document) * np.linalg.norm(query))

    @staticmethod
    def rank(documents, query):
        ranked_documents = (
            documents.apply(TFIDF.similarity, args=(query,))
            .T[0]
            .sort_values(ascending=False)
        )
        ranked_documents = pd.DataFrame(
            {"document": ranked_documents.index, "score": ranked_documents.to_list()}
        )
        return ranked_documents


class BM25:
    def __init__(self, documents, terms, K=1.2, b=0.75):
        self.K = K
        self.b = b
        self.N = len(documents)
        self.documents = documents

        self.matrix = pd.DataFrame(index=range(len(documents)), columns=terms)
        self.doclens = list(
            map(lambda doc: sum(map(lambda word: len(word), doc)), documents)
        )
        self.avg_doclen = np.mean(
            list(map(lambda doc: sum(map(lambda w: len(w), doc)), documents))
        )

    @staticmethod
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
        ranked_documents = self.matrix[query].T.sum().sort_values(ascending=False)
        ranked_documents = pd.DataFrame(
            {"document": ranked_documents.index, "score": ranked_documents.to_list()}
        )
        return ranked_documents


class Evaluator:
    @staticmethod
    def recall(A, R):
        return len(np.intersect1d(A, R)) / len(R)

    @staticmethod
    def precision(A, R):
        return len(np.intersect1d(A, R)) / len(A)

    @staticmethod
    def recall_and_precision(A, R):
        rp = pd.DataFrame(columns=["recall", "precision"])

        for i in range(1, len(A) + 1):
            a = A[:i]
            if A[i - 1] not in R:
                continue

            recall = Evaluator.recall(a, R)
            precision = Evaluator.precision(a, R)
            new_row = {"recall": recall, "precision": precision}
            rp.loc[len(rp)] = new_row

        return rp.sort_values("recall")

    @staticmethod
    def interpolated_precision(rec_and_pre):
        index = np.array(range(10 + 1)) / 10
        ip = pd.DataFrame(columns=["precision"], index=index)

        for idx, _ in ip.iterrows():
            rp = rec_and_pre[rec_and_pre["recall"] >= idx]
            if not len(rp):
                ip.loc[idx] = 0
            else:
                ip.loc[idx] = rp["precision"].max()

        return ip

    @staticmethod
    def mean_average_precision(rec_and_pre, R_length):
        return rec_and_pre["precision"].sum() / R_length


def exec_example(dictionary, stopwords, query, separators, R):
    print(f"dictionary = {dictionary}")
    print(f"stopwords = {stopwords}")
    print(f"query = {query}")
    print(f"separators = {separators}\n")
    print(f"R = {R}")

    # normalize / tokenize
    normalized = np.array([normalize(s) for s in dictionary])
    tokens_list = np.array([tokenize(s, separators) for s in normalized], dtype=object)
    tokens_list = remove_stopwords(tokens_list, stopwords)

    terms = np.array(sorted(list(set([term for l in tokens_list for term in l]))))
    query = np.array(query.split())

    # retrieval
    tfidf = TFIDF(tokens_list, terms)
    tfidf.generate_matrix()
    query_weights = TFIDF(np.array([query]), terms)
    query_weights.generate_matrix()
    tfidf_ranks = TFIDF.rank(tfidf.matrix, query_weights.matrix)

    bm25 = BM25(tokens_list, terms)
    bm25.generate_bm_matrix()
    bm25_ranks = bm25.rank(query)

    # avaliation: tfidf
    print("\n===> Avaliating TF-IDF")
    print(f"\n+ Ranks:\n{tfidf_ranks}\n")
    A = np.array(tfidf_ranks["document"])
    recall_and_precision = Evaluator.recall_and_precision(A, R)
    print(f"\n+ Recall and precision:\n{recall_and_precision}")
    interpolated_precision = Evaluator.interpolated_precision(recall_and_precision)
    print(f"\n+ Interpolated precision:\n{interpolated_precision}")
    mean_average_precision = Evaluator.mean_average_precision(
        recall_and_precision, len(R)
    )
    print(f"\n+ Mean average precision:\n{mean_average_precision}\n")

    # avaliation: BM25
    print("\n===> Avaliating BM25")
    print(f"\n+ Ranks:\n{bm25_ranks}\n")
    A = np.array(bm25_ranks["document"])
    recall_and_precision = Evaluator.recall_and_precision(A, R)
    print(f"\n+ Recall and precision:\n{recall_and_precision}")
    interpolated_precision = Evaluator.interpolated_precision(recall_and_precision)
    print(f"\n+ Interpolated precision:\n{interpolated_precision}")
    mean_average_precision = Evaluator.mean_average_precision(
        recall_and_precision, len(R)
    )
    print(f"\n+ Mean average precision:\n{mean_average_precision}\n")


def ex1_input():
    # documentos
    dictionary = np.array(
        [
            "O peã e o caval são pec de xadrez. O caval é o melhor do jog.",
            "A jog envolv a torr, o peã e o rei.",
            "O peã lac o boi",
            "Caval de rodei!",
            "Polic o jog no xadrez.",
        ]
    )
    stopwords = ["a", "o", "e", "é", "de", "do", "no", "são"]  # lista de stopwords
    query = "xadrez peã caval torr"  # consulta
    separators = [" ", ",", ".", "!", "?"]  # separadores para tokenizacao
    R = np.array([1, 2])

    print("\n*** Exemplo 1:")

    return dictionary, stopwords, query, separators, R


def ex2_input():
    # documentos
    dictionary = np.array(
        [
            "Parasita é o grande vencedor do Oscar 2020, com quatro prêmios",
            "Green Book, Roma e Bohemian Rhapsody são os principais vencedores do Oscar 2019",
            "Oscar 2020: Confira lista completa de vencedores. Parasita e 1917 foram os grandes vencedores da noite",
            "Em boa fase, Oscar sonha em jogar a Copa do Mundo da Rússia",
            "Conheça os indicados ao Oscar 2020; Cerimônia de premiação acontece em fevereiro",
            "Oscar Schmidt receberá Troféu no Prêmio Brasil Olímpico 2019. Jogador de basquete com mais pontos em Jogos Olímpicos.",
            "Seleção brasileira vai observar de 35 a 40 jogadores para definir lista da Copa América",
            "Oscar 2020: saiba como é a escolha dos jurados e como eles votam",
            "Bem, Amigos! discute lista da Seleção, e Galvão dá recado a Tite: Cadê o Luan?",
            "IFAL-Maceió convoca aprovados em lista de espera do SISU para chamada oral",
            "Arrascaeta e Matías Viña são convocados pelo Uruguai para eliminatórias da Copa. Além deles, há outros destaques na lista.",
            "Oscar do Vinho: confira os rótulos de destaque da safra 2018",
            "Parasita é o vencedor da Palma de Ouro no Festival de Cannes",
            "Estatísticas. Brasileirão Série A: Os artilheiros e garçons da temporada 2020",
            "Setembro chegou! Confira o calendário da temporada 2020/2021 do futebol europeu",
        ]
    )
    stopwords = [
        "a",
        "o",
        "e",
        "é",
        "de",
        "do",
        "da",
        "no",
        "na",
        "são",
        "dos",
        "com",
        "como",
        "eles",
        "em",
        "os",
        "ao",
        "para",
        "pelo",
    ]  # lista de stopwords
    query = "oscar 2020"  # consulta
    separators = [
        " ",
        ",",
        ".",
        "!",
        "?",
        ":",
        ";",
        "/",
    ]  # separadores para tokenizacao
    R = np.array([1, 3, 5, 8])

    print("\n*** Exemplo 2:")

    return dictionary, stopwords, query, separators, R


def main():
    exec_example(*ex1_input())
    exec_example(*ex2_input())


if __name__ == "__main__":
    main()
