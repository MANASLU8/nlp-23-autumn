import os
import json
import re
import numpy as np
from tokenizer import tokenize_text

# определим путь к папке, содержащей файлы tsv
base_folder = 'C:/Users/n.strokova/Pictures/ITMO/nlp/assets/train'

# определим список категорий
categories = ['Business', 'Sci-Tech', 'Sports', 'World']

# создадим пустой словарь токенов
word_count = {}

# проходим по каждой категории
for category in categories:
    category_folder = os.path.join(base_folder, category)
    # проходим по каждому файлу внутри категории
    for file_name in os.listdir(category_folder):
        if file_name.endswith(".tsv"):
            file_path = os.path.join(category_folder, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    tokens = tokenize_text(line)
                    for token in tokens:
                        _, kind, part = token
                        if kind == 'word':
                            token = re.sub(r'[^\w\s]', '', part.lower())
                            if token in word_count:
                                word_count[token] += 1
                            else:
                                word_count[token] = 1

# сортировка словаря токенов
sorted_word_count = {k: v for k, v in sorted(word_count.items(), key=lambda item: item[1], reverse=True)}

# сохраняем словарь в отдельный файл
with open('token_dictionary.json', 'w') as f:
    json.dump(sorted_word_count, f, ensure_ascii=False, indent=4)

# создадим пустую матрицу термин-документ
term_document_matrix = np.zeros((len(sorted_word_count), len(categories)))

# проходим по каждой категории
for i, category in enumerate(categories):
    category_folder = os.path.join(base_folder, category)
    # проходим по каждому файлу tsv внутри категории
    for file_name in os.listdir(category_folder):
        if file_name.endswith(".tsv"):
            file_path = os.path.join(category_folder, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    tokens = tokenize_text(line)
                    token_set = set()
                    for token in tokens:
                        _, kind, part = token
                        if kind == 'word':
                            token = re.sub(r'[^\w\s]', '', part.lower())
                            token_set.add(token)
                    # увеличиваем значение в матрице термин-документ
                    for token in token_set:
                        if token in sorted_word_count:
                            term_document_matrix[list(sorted_word_count.keys()).index(token), i] += 1

# Сохраняем матрицу термин-документ в отдельный файл
np.savetxt('term_document_matrix.txt', term_document_matrix, fmt='%d')

# пример трактовки матрицы термин-документ
# 2900 2992 3971 3068
# слово встречается в 1-й категории 2900 раз, во второй 2991 и т.д.
