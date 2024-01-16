import os
import numpy as np
import nltk
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer


def vectorize_text(text, model):
    # сегментация текста на предложения и токены
    sentences = nltk.sent_tokenize(text)
    tokens = [nltk.word_tokenize(sentence) for sentence in sentences]

    # формирование векторных представлений каждого токена
    token_vectors = []
    for sentence_tokens in tokens:
        sentence_vectors = []
        for token in sentence_tokens:
            # проверка, что токен есть в словаре модели
            if token in model.wv.key_to_index.keys():
                sentence_vectors.append(model.wv[token])
        token_vectors.append(sentence_vectors)

    # подсчет среднего значения векторных представлений токенов каждого предложения
    sentence_vectors = []
    for sentence_tokens in token_vectors:
        if len(sentence_tokens) > 0:
            sentence_vectors.append(sum(sentence_tokens) / len(sentence_tokens))
        else:
            sentence_vectors.append([0] * len(model.wv[model.wv.index_to_key[0]]))

    # подсчет векторного представления документа
    document_vector = np.mean(sentence_vectors, axis=0)

    return document_vector


# Загрузка модели векторных представлений текста, обученной на большом корпусе
model = Word2Vec.load('w2v_model')

# Путь к папке с тестовой выборкой
test_folder = 'C:/Users/n.strokova/Pictures/ITMO/nlp/assets/test'

# создание списка для хранения результирующих векторных представлений документов
document_vectors = []

# обход всех папок в заданной папке с тестовой выборкой
for i, folder_name in enumerate(os.listdir(test_folder)):
    folder_path = os.path.join(test_folder, folder_name)

    # проверка, является ли путь папкой
    if os.path.isdir(folder_path):
        # создание списка для хранения векторных представлений документов внутри папки
        folder_document_vectors = []

        # обход всех файлов внутри папки
        for j, file_name in enumerate(os.listdir(folder_path)):
            if file_name.endswith('.tsv'):
                file_path = os.path.join(folder_path, file_name)

                # чтение содержимого файла
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()

                # векторизация текста
                document_vector = vectorize_text(text, model)

                # добавление идентификатора и векторного представления в список
                folder_document_vectors.append((file_name[:-4], document_vector))

            # вывод статуса выполнения
            progress = (j + 1) / len(os.listdir(folder_path)) * 100
            print(f"Обработано {progress:.2f}% из {folder_name}")

        # сохранение результатов в формате tsv для каждой папки
        output_file = f'C:/Users/n.strokova/Pictures/ITMO/nlp/assets/annotated-corpus/{folder_name}_vectors_test.tsv'
        with open(output_file, 'w', encoding='utf-8') as file:
            for document_id, document_vector in folder_document_vectors:
                line = f"{document_id}\t" + "\t".join(str(val) for val in document_vector)
                file.write(line + '\n')

        # добавление списка векторных представлений папки в общий список
        document_vectors.extend(folder_document_vectors)

    # Вывод статуса выполнения
    progress = (i + 1) / len(os.listdir(test_folder)) * 100
    print(f"Обработано {progress:.2f}% из всего набора данных")

print("Векторизация тестовой выборки выполнена.")
