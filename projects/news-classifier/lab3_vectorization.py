# %%
!python -m spacy download en_core_web_md
import os
import re
from collections import Counter
from scipy.sparse import lil_matrix, save_npz
import nltk
from nltk.corpus import stopwords
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import spacy

nltk.download('stopwords')
nltk.download('punkt')

# %%
assets_dir = os.path.join(os.getcwd(), 'drive', 'MyDrive', 'ITMO', 'sem_3', 'NLP', 'assets', 'annotated-corpus')
train_dir = os.path.join(assets_dir, "train")
assets_dir

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
topics = os.listdir(train_dir)
topics

# %%
# # # # # # #
# ДОБАВЛЕНО #
# Шаблон регулярного выражения для слова (\b - граница слова, \w+ буквенный символ (минимум 1 подряд))
word_regex = re.compile(r'\b\w+\b')
# # # # # # #

# %%
sentences = []
fnum = 0
for t in topics:
    # путь до конкретного класса новостей
    workdir = os.path.join(train_dir, t)
    if fnum >= 1000:
          break
    # перебор всех файлов tsv в директории
    for filename in os.listdir(workdir):
        fnum += 1
        if fnum >= 1000:
          break
        print(fnum)
        # чтение файла
        with open(os.path.join(workdir, filename)) as f:
            # объединение всех строк в файле в одну
            lines = "".join(f.readlines())
            # разделение строки на предложения
            sentences_raw = lines.split("\n\n")
            # перебор предложений
            for s in sentences_raw:
                # предложение разделяем на слова
                words = s.split("\n")
                if len(words) == 0 or words[0] == "":
                    continue
                # третий элемент (индекс 2 = стемма) каждого слова words, разделенного по символу табуляции
                stems_raw = list(map(lambda x: x.split("\t")[2], words))
                # четвертый элемент (индекс 3 = лемма) каждого слова words, разделенного по символу табуляции
                lemmas = list(map(lambda x: x.split("\t")[3], words))
                stems = []
                # перебор символов в стемме
                for i in range(len(stems_raw)):
                    # Если лемма не является стоп-словом,
                    # то соответствующий стем (stems_raw[i])
                    # добавляется в список stems

                    # # # # # # #
                    # ДОБАВЛЕНО #
                    # Если стем не является словом (не подходит под шаблон регулярного выражения), то пропускаем итеррацию (слово не добавляется)

                    if lemmas[i] not in stopwords.words("english") and word_regex.match(stems_raw[i]):
                        stems.append(stems_raw[i].lower())

                     # # # # # # #

                # добавление стемм в список предложений
                sentences.append(stems)

# %% [markdown]
# ## Task 1
# 
# 
# По сформированной в результате выполнения первой лабораторной работы аннотации обучающей выборки в формате `tsv` построить словарь токенов с указанием их частот (словарь должен содержать как сами токены, так и количество их употреблений в обучающей выборке) и матрицу "термин-документ" (`term-document matrix`).
# 
# Результаты необходимо сохранить во внешние файлы в произвольном формате. Использование стандартных библиотечных реализаций данных преобразований не разрешается. Также рекомендуется убирать из текста стоп-слова и пунктуацию, а также низкочастотные токены. Также для получения дополнительных баллов по данному пункту необходимо учитывать эффективность хранения разреженных структур данных на диске.

# %%
token_counter = Counter()
doc_token_matrix = []

for document_tokens in sentences:
    # Обновляем счетчик токенов
    token_counter.update(document_tokens)

# Убираем низкочастотные токены
filtered_tokens = {token for token, freq in token_counter.items() if freq >= 6}

# Сортируем токены для формирования индексов в матрице
sorted_tokens = sorted(filtered_tokens)

# Строим матрицу "термин-документ"
for document_tokens in sentences:
    row = np.zeros(len(sorted_tokens), dtype=int)
    for token in document_tokens:
        if token in filtered_tokens:
            # Заполняем матрицу только для уникальных токенов
            row[sorted_tokens.index(token)] += 1
    doc_token_matrix.append(row.tolist())

# %%
# Сохраняем словарь в файл
with open('token_dictionary.txt', 'w', encoding='utf-8') as dict_file:
    for token, count in token_counter.items():
            dict_file.write(f'{token}\t{count}\n')

# Сохраняем матрицу "термин-документ" в файл
np.savetxt('term_document_matrix.txt', doc_token_matrix, fmt='%d', delimiter='\t')

# %% [markdown]
# ## Task 2
# 
# Разработать метод, позволяющий преобразовать произвольный текст в вектор частот токенов, содержащихся в данном тексте, с использованием словаря токенов с указанием их частот, полученного на шаге 1.

# %%
def text_to_vector(text, token_dict):
    # Преобразование текста в вектор частот токенов
    text = re.sub(r'[^\w\s]', '', text).lower()  # Предварительная обработка текста
    tokens = text.split(' ')

    # Поиск соответствующих значений в словаре
    vector = [int(token_dict[token]) if token in token_dict.keys() else -1 for token in tokens]

    return vector

# %%
input_text = """It's a fish symbol, like the ones Christians stick on their cars, but with feet and the word "Darwin" written inside."""
with open('token_dictionary.txt', 'r') as file:
  token_dictionary = file.read()

token_dictionary = {row.split('\t')[0]: row.split('\t')[1] for row in token_dictionary.split('\n') if row != ''}

print(text_to_vector(input_text, token_dictionary))

# %% [markdown]
# ## Task 3
# 
# Реализовать метод, позволяющий векторизовать произвольный текст с использованием нейронных сетей (`w2v`). Выбранную модель необходимо обучить на обучающей выборке.

# %%
def train_word2vec_model(data):
    tokenized_data = [word_tokenize(text) for text in data.split(' ')]
    model = Word2Vec(sentences=tokenized_data, vector_size=100, window=5, min_count=1, workers=4)
    # min_count=1 - учитываются все слова, даже если они только один раз встречаются в тексте
    # vector_size=100 - каждое слово будет представлено в виде вектора размерностью 10
    # window=5 - пять слов до текущего и 5 после
    model.save("w2v_model")

    return model

# %%
with open(os.path.join(os.getcwd(), 'drive', 'MyDrive', 'ITMO', 'sem_3', 'NLP', 'assets', 'dataset', 'train.csv')) as file:
  data = file.read() #[:10000]
model = train_word2vec_model(data)

# %%
# # Как найти путь

# dataset_dir = os.path.realpath("../../assets/dataset")
# train_file_dir = os.path.join(dataset_dir, "train.csv")
# train_file_dir

# %% [markdown]
# ## Task 4
# 
# С использованием библиотечной реализации метода подсчета косинусного расстояния между векторными представлениями текста, продемонстрировать на примерах, что для семантически близких слов модель генерирует вектора, для которых косинусное расстояние меньше, чем для семантически далеких токенов. Демонстрация работы модели происходит в соответствии со сценарием:
# * Ручное выделение набора токенов (от 2 до 5) из датасета, для каждого токена определение 2-3 токенов с похожим значением, 2-3 токенов из той же предметной области и 2-3 токенов с совершенно другими семантическими свойствами. Например, если изначально взяли токен cat, то токенами с похожим значением могут быть tiger, felines, токенами из той же предметной области - animal, rabbit, токенами с соверешенно другими семантическими свойствами - sentence, creation. Необходимо получить векторное представление для каждого исходного токена, векторные представления токенов из 3 указанных групп и продемонстрировать в виде ранжированного списка с указанием косинусного расстояния, что между векторным представлением исходного токена и токенов с похожим значением косинусное расстояние меньше, чем между векторным представлением исходного токена и токенов из той же предметной области, которое в свою очередь меньше косинусного расстояния между векторным представлением исходного токена и векторными представлениями токенов с совершенно другими семантическими свойствами

# %%
# Функция для измерения косинусного расстояния и вывода ранжированного списка
def measure_cosine_similarity(reference_vector, comparison_vectors, labels):
    similarity_scores = cosine_similarity([reference_vector], comparison_vectors)[0]
    ranked_results = sorted(zip(labels, similarity_scores), key=lambda x: x[1], reverse=True)

    print(f"\nРанжированный список для '{tokens_to_test[0]}':")
    for token, score in ranked_results:
        print(f"{token}: {score}")

# %%
# Примеры токенов для тестирования
tokens_to_test = ["cat", "tiger", "felines", "animal", "rabbit", "sentence", "creation"]

# Векторизация исходных токенов
vectors_to_test = [model.wv[token] for token in tokens_to_test]

# Вывод векторных представлений
for i, token in enumerate(tokens_to_test):
    print(f"Векторное представление для '{token}': {vectors_to_test[i]}")

# %%
# Сравнение векторов для токенов с похожим значением
measure_cosine_similarity(vectors_to_test[0], vectors_to_test[1:3], tokens_to_test[1:3])

# Сравнение векторов для токенов из той же предметной области
measure_cosine_similarity(vectors_to_test[0], vectors_to_test[3:5], tokens_to_test[3:5])

# Сравнение векторов для токенов с совершенно другими семантическими свойствами
measure_cosine_similarity(vectors_to_test[0], vectors_to_test[5:], tokens_to_test[5:])

# %%
import numpy as np

vec = np.array(vectors_to_test)

# Визуализация результатов
plt.figure(figsize=(8, 6))
for i, label in enumerate(tokens_to_test):
    plt.scatter(vec[i, 0], vec[i, 1], label=label)

# Добавление меток для точек
for i, label in enumerate(tokens_to_test):
    plt.annotate(label, (vec[i, 0], vec[i, 1]))

plt.title('Результаты косинусного расстояние для векторов Word2Vec')
plt.legend()
plt.show()

# %% [markdown]
# ## Task 5
# 
# Применить какой-либо метод сокращения размерностей полученных одним из базовых способов векторизации, выбранным ранее (см. пункт 2), векторов (в простейшем случае можно использовать метод PCA, причем допускается использование библиотечной реализации, сокращенная размерность должна быть сопоставима с размерностью векторов, формируемых векторной моделью, примененной на шаге 3, поскольку далее будет предложено сравнить данный метод с подходом, основанным на использовании векторной модели), а именно кодированием текста в виде последовательности частот токенов

# %%
# Функция для применения PCA и визуализации результата
def apply_pca_and_visualize(vectors, labels):
    # Применение PCA
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(vectors)

    # Визуализация результатов
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(labels):
        plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1], label=label)

    # Добавление меток для точек
    for i, label in enumerate(labels):
        plt.annotate(label, (reduced_vectors[i, 0], reduced_vectors[i, 1]))

    plt.title('Результаты PCA для векторов Word2Vec')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()

# %%
# Применение PCA и визуализация результатов
apply_pca_and_visualize(vectors_to_test, tokens_to_test)

# %% [markdown]
# ## Task 6
# 
# С использованием разработанного метода подсчета косинусного расстояния сравнить эффективность метода векторизации с использованием нейронных сетей и эффективность базовых методов векторизации с последующим сокращением размерности. Сформулировать вывод о том, применение какого способа позволяет получить лучшие результаты на выбранном датасете.
# 

# %% [markdown]
# 
# ### Вывод:
# 
# Модель с использованием метода главных компонент (PCA) работает лучше.

# %% [markdown]
# ## Task 7
# 
# Реализовать метод, осуществляющий векторизацию произвольного текста по следующему алгоритму:
# 1. Сегментация текста на предложения и токены;
# 2. Формирование векторных представлений каждого токена по-отдельности с использованием выбранной модели векторных представлений текста, основанной на нейронных сетях;
# 3. Подсчет среднего значения векторных представлений токенов каждого предложения;
# 4. Подсчет векторного представления документа по векторным представлениям составляющих его предложений в соответствии с некоторым подходом (например, путем подсчета среднего значения).

# %%
import numpy as np
import nltk
from gensim.models import Word2Vec

def text_vectorization(text, model):
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
    # Если предложение содержит токены, их вектора суммируются и делятся на количество токенов в предложении. Если предложение пустое, для него создается вектор из нулей.
    sentence_vectors = []
    for sentence_tokens in token_vectors:
        if len(sentence_tokens) > 0:
            sentence_vectors.append(sum(sentence_tokens) / len(sentence_tokens))
        else:
            sentence_vectors.append([0] * len(model.wv[model.wv.index_to_key[0]]))

    # подсчет векторного представления документа
    document_vector = np.mean(sentence_vectors, axis=0)

    return document_vector

# %%
# Загрузка модели векторных представлений текста, обученной на большом корпусе
model = Word2Vec.load('w2v_model')

result_vector = text_vectorization(input_text, model)

print("Document Vector:", result_vector)

# %% [markdown]
# ## Task 8
# 
# Выполнить векторизацию тестовой выборки с использованием метода, реализованного на предыдущем шаге. Результаты сохранить в формате `tsv `

# %%
# Загрузка модели векторных представлений текста, обученной на большом корпусе
model = Word2Vec.load('w2v_model')

# Путь до директории с тестовой выборкой
test_dir = os.path.join(os.getcwd(), 'drive', 'MyDrive', 'ITMO', 'sem_3', 'NLP', 'assets', 'annotated-corpus', 'test')


# Путь до директории, в которой будет сохранен результат
output_dir = os.path.join(os.getcwd(), 'drive', 'MyDrive', 'ITMO', 'sem_3', 'NLP', 'assets', 'annotated-corpus', 'test-embeddings.tsv')


# Функция для чтения текста из файла
def read_text_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# Функция для сохранения векторных представлений в файл
def save_embeddings_to_tsv(file_path, embeddings):
    with open(file_path, "w", encoding="utf-8") as file:
        for doc_id, embedding in embeddings.items():
            line = [str(doc_id)] + [str(component) for component in embedding]
            file.write("\t".join(line) + "\n")



# Словарь для хранения векторных представлений каждого документа
embeddings_dict = {}

for topic_id, topic in enumerate(topics):
  # Получаем список файлов в тестовой выборке
  test_files = os.listdir(os.path.join(test_dir, topic))
  # Проходим по каждому файлу в тестовой выборке
  for file_name in test_files:
      # Полный путь к файлу
      file_path = os.path.join(test_dir, topic, file_name)

      # Извлекаем doc_id из названия файла (например, "001.txt" -> "001")
      doc_id = os.path.splitext(file_name)[0]

      # Читаем текст из файла
      text = read_text_from_file(file_path)

      # Выполняем векторизацию текста
      vector = text_vectorization(text, model)

      # Сохраняем вектор в словарь
      embeddings_dict[(doc_id, topic_id)] = vector

# Сохраняем векторные представления в файл
save_embeddings_to_tsv(output_dir, embeddings_dict)

print(f"Векторные представления сохранены в {output_dir}")

# %% [markdown]
# /content/drive/MyDrive/Colab Notebooks/assets/dataset/test.csv


