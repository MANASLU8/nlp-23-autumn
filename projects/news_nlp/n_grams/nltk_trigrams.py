import nltk
from nltk.collocations import *
from nltk.corpus import stopwords
import os
import string
from math import log2

# используется для вычисления метрики
trigram_measures = nltk.collocations.TrigramAssocMeasures()
# путь к датасету
directory = 'C:/Users/n.strokova/Pictures/ITMO/nlp/assets/raw-dataset/'


# предварительная обработка текста
def preprocess(directory):
    # просмотр всех файлов в папке
    filenames = os.listdir(directory)
    corpus = ""
    stop_words = set(stopwords.words('english'))
    for filename in filenames:
        with open(os.path.join(directory, filename), 'r') as file:
            f = file.read()
            # удаление знаков пунктуации
            f = f.translate(str.maketrans('', '', string.punctuation))
            # приведение к нижнему регистру
            f = f.lower()
            # удаление всех цифр из текста
            f = ''.join([c for c in f if not c.isdigit()])  # Удаление всех цифр
            # разделение текста на отдельные слова
            words = f.split()
            # удаление стоп-слов
            filtered_words = [word for word in words if word not in stop_words]  # Удаление стоп-слов
            f = ' '.join(filtered_words)
            corpus += f
    return corpus

# запись предварительно подготовленного текста в переменную
raw = preprocess(directory)

# токенизация
tokens = nltk.word_tokenize(raw)
text = nltk.Text(tokens)

# извлечение триграмм, используя библиотеку
finder_thr = TrigramCollocationFinder.from_words(text)

# Расчет метрики MI
trigram_scores = finder_thr.score_ngrams(trigram_measures.pmi)
# добавление значения MI к триграмме
score_with_mi = [(score[0], score[1], log2(score[1])) for score in trigram_scores]
# сортировка количества упоминаний по убыванию
sorted_trigram_scores = sorted(score_with_mi, key=lambda x: x[1], reverse=True)

# формирование итогов: триграмма + кол-во упоминаний + MI
for score in sorted_trigram_scores:
    trigram = ' '.join(score[0])
    frequency = int(round(score[1]))
    mi = score[2]
    print(f"{trigram} ({frequency} mentions) - MI: {mi}")

# Сохранение результатов в файл
with open('nltk_trigram_measures.txt', 'w') as file:
    for score in sorted_trigram_scores:
        trigram = ' '.join(score[0])
        frequency = int(round(score[1]))
        mi = score[2]
        file.write(f"{trigram} ({frequency} mentions) - MI: {mi}\n")