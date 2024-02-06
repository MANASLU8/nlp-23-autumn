# %%
import os
import re
from math import sqrt
import nltk
from nltk.corpus import stopwords
from nltk.collocations import  *
from nltk import Text

nltk.download('stopwords')

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
!unzip drive/MyDrive/ITMO/sem_3/NLP/assets/annotated-corpus/train.zip -d train
assets_dir = os.path.join(os.getcwd(), 'drive', 'MyDrive', 'ITMO', 'sem_3', 'NLP', 'assets', 'annotated-corpus')
train_dir = os.path.join(assets_dir, "train")
assets_dir

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
for t in topics:
    # путь до конкретного класса новостей
    workdir = os.path.join(train_dir, t)
    # перебор всех файлов tsv в директории
    for filename in os.listdir(workdir):
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
                        stems.append(stems_raw[i])

                     # # # # # # #

                # добавление стемм в список предложений
                sentences.append(stems)

# %%
len(sentences)

# %%
# посследовательность из 3-ех слов
ngram_length = 3

# %%
# для найденных n-gram
ngrams = []
# кол-во упоминаний каждого слова
word_count = {}
# кол-во встречаемости каждой n-граммы
ngrams_count = {}
# перебор предложений
for s in sentences:
    # общее кол-во слов в предложении
    counter = 0
    # перебор слов в предложении
    for w in s:
        # проверка, есть ли слово в словаре
        if w not in word_count.keys():
            # добавляем со значением 0
            word_count[w] = 0
        # увеличиваем значение слова
        word_count[w] += 1
        counter += 1
    # если слов в предложении меньше чем длина n-gram
    if counter < ngram_length:
        continue
    # Цикл for, который перебирает каждый индекс (i)
    # в диапазоне от 0 до (длина предложения - длина n-граммы + 1).
    # Внутри цикла формируется n-грамма из текущего подпространства
    # слов (s[i:i+ngram_length]), преобразуется в кортеж (tuple),
    # и если данная n-грамма отсутствует в словаре ngrams_count,
    # она добавляется со значением 0. Затем значение данной n-граммы
    # в словаре ngrams_count увеличивается на 1.
    # Наконец, n-грамма добавляется в список ngrams.
    for i in range(len(s) - ngram_length + 1):
        ngram = tuple(s[i:i+ngram_length])
        if ngram not in ngrams_count.keys():
            ngrams_count[ngram] = 0
        ngrams_count[ngram] += 1
        ngrams.append(ngram)

# %%
len(ngrams)

# %%
ngrams

# %%
word_count

# %%
# Сортировка словаря word_count по убыванию значений (-x[1]) и
# выбор первых 30 элементов. Это возвращает отсортированный
# список пар (слово, количество упоминаний) с наибольшими значениями
sorted(word_count.items(), key=lambda x: -x[1])[:30]

# %%
# Сортировка словаря ngrams_count по убыванию значений (-x[1])
# и выбор первых 30 элементов. Это возвращает отсортированный
# список пар (n-грамма, количество упоминаний) с наибольшими значениями
sorted(ngrams_count.items(), key=lambda x: -x[1])[:30]

# %%
# сумма всех значений (количества упоминаний) в словаре word_count
total_words = sum(word_count.values())
total_words

# %%
# для оценок для каждой н-граммы
ngram_score = {}
# перебор ункальных н-грамм
for ngram in set(ngrams):
    count_mul_result = 1
    # перебор каждого слова н-граммы
    for word in ngram:
        # кол-во упоминаний каждого слова word в н-грамме во всем тексте из словаря word_count
        count_mul_result *= word_count[word]
    # оценка для н-граммы: вычитание относительного произведения count_mul_result и total_words
    # от количества упоминаний данной н-граммы из словаря ngrams_count, деленного на корень квадратный из количества упоминаний данной н-граммы
    ngram_score[ngram] = (ngrams_count[ngram] - (count_mul_result / (total_words ** (ngram_length - 1)))) / sqrt(ngrams_count[ngram])
ngram_score

# %%
# сортировка оценок по убыванию и выбор первых 30 эл-ов
sorted(ngram_score.items(), key=lambda x: -x[1])[0:30]

# %%
# перебор всех слов в каждом предложении в список text
text = []
for s in sentences:
    text += s

# %%
# нахождение и анализ коллокаций (частотных словосочетаний) в тексте
finder_thr = TrigramCollocationFinder.from_words(Text(text))

# %%
# 30 лучших коллокаций на основе метода статистического метода student_t
# насколько сильно отличаются средние значения частоты появления триграмм в тексте от ожидаемых значений при условии независимости слов
finder_thr.nbest(nltk.collocations.TrigramAssocMeasures().student_t, 30)


