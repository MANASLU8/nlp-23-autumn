import numpy as np
import re
import json
from tokenizer import tokenize_text


def text_to_tfidf_vector(text, token_dictionary, term_document_matrix):
    # преобразуем текст в список токенов
    tokens = tokenize_text(text)

    # создаем пустой вектор tf-idf
    tfidf_vector = np.zeros(len(token_dictionary))

    # считаем частоту каждого токена в тексте
    token_count = {}
    for token in tokens:
        _, kind, part = token
        if kind == 'word':
            token = re.sub(r'[^\w\s]', '', part.lower())
            if token in token_dictionary:
                if token in token_count:
                    token_count[token] += 1
                else:
                    token_count[token] = 1

    # вычисляем tf-idf для каждого токена в тексте
    for token, frequency in token_count.items():
        token_index = token_dictionary[token]
        if token_index >= len(term_document_matrix):
            continue
        tfidf_vector[token_index] = frequency * np.log(
            1 + len(term_document_matrix[0]) / np.count_nonzero(term_document_matrix[token_index]))

    # нормализуем вектор tf-idf
    tfidf_vector = tfidf_vector / np.linalg.norm(tfidf_vector)

    return tfidf_vector


# произвольный текст, который нужно преобразовать в вектор значений tf-idf
text = "Dwyane Wade scored 25 points to inspire the Miami Heat to their seventh straight win, a 107-100 victory over the Denver Nuggets Friday"

# загрузка словаря из файла JSON
with open('token_dictionary.json', 'r') as f:
    token_dictionary = json.load(f)

# загрузка матрицы "термин-документ" из файла
term_document_matrix = np.loadtxt('term_document_matrix.txt')

# Вызов функции text_to_tfidf_vector
tfidf_vector = text_to_tfidf_vector(text, token_dictionary, term_document_matrix)

# вывод полученного вектора tf-idf
print(tfidf_vector)

# записываем tf-idf вектор в файл
np.savetxt('tfidf_vector.txt', tfidf_vector, delimiter=',')
