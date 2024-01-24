import os
import string
from collections import Counter
from nltk.corpus import stopwords
import nltk
import math
from nltk.stem import WordNetLemmatizer

# модуль для разделение текста на отдельные слова
nltk.download('punkt')
# база для лемматизации
nltk.download('wordnet')
nltk.download('stopwords')


# Очистка от знаков пунктуации и приведение к нижнему регистру
def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    return text


# Извлечение триграмм из текста на основании лексем
def extract_trigrams(text):
    trigrams = []
    # разделение на лексемы
    tokens = nltk.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    # ограничение, чтобы триграмма не брала последнее слово в предложении в качестве начального
    for i in range(len(lemmas) - 2):
        # текущая лексема + следующая лексема + лексема через одну
        trigram = ' '.join([lemmas[i], lemmas[i+1], lemmas[i+2]])
        trigrams.append(trigram)
    return trigrams


# Считывание содержимого файла, предварительная обработка
# Подсчет кол-ва триграмм
def process_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    # вызов функции препроцессинга (регистр, знаки препинания)
    preprocessed_content = preprocess_text(content)
    # поиск стоп-слов
    stop_words = set(stopwords.words('english'))
    # разбивка на отдельные слова
    words = preprocessed_content.split()
    # исключение стоп-слов
    filtered_words = [word for word in words if word not in stop_words]
    filtered_text = ' '.join(filtered_words)
    # непосредственно извлечение триграмм и их подсчет
    trigrams = extract_trigrams(filtered_text)
    trigram_counts = Counter(trigrams)
    return trigram_counts


# Обработка всех файлов в директории, откуда извлекают триграммы
def process_folder(folder_path):
    # подсчет триграмм
    trigram_counts = Counter()
    # подсчет отдельных слов
    unigram_counts = Counter()
    # перебираем все файлы в директории
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            trigram_counts += process_file(file_path)
            unigram_counts += process_unigrams(file_path)
    # возвращаем кол-во слов и триграмм
    return trigram_counts, unigram_counts


# Обработка отдельного файла и подсчет слов
def process_unigrams(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    # вызов функции предобработки текста
    preprocessed_content = preprocess_text(content)
    stop_words = set(stopwords.words('english'))
    words = preprocessed_content.split()
    # исключаются стоп-слова
    filtered_words = [word for word in words if word not in stop_words]
    unigram_counts = Counter(filtered_words)
    # возвращение количества слов
    return unigram_counts


# Сохранение всех триграмм в файл
def save_trigrams(trigram_counts):
    n_grams_folder = 'n_grams'
    os.makedirs(n_grams_folder, exist_ok=True)
    with open(os.path.join(n_grams_folder, 'all_trigrams.txt'), 'w') as file:
        for trigram, count in trigram_counts.items():
            file.write(f'{trigram}\t{count}\n')


# Сохранение ТОП-30 триграмм в файл
def save_top_n_trigrams(trigram_counts, n):
    n_grams_folder = 'n_grams'
    os.makedirs(n_grams_folder, exist_ok=True)
    top_n_trigrams = trigram_counts.most_common(n)
    with open(os.path.join(n_grams_folder, 'top_30_trigrams.txt'), 'w') as file:
        for trigram, count in top_n_trigrams:
            file.write(f'{trigram}\t{count}\n')


# Подсчет MI (взаимной информации)
# На вход - кол-во триграмм, текущая триграмма, кол-во слов
# Измеряет степень зависимости между двумя переменным, основываясь на изменении вероятности одной переменной
# при известной информации о другой переменной
def calculate_mutual_information(trigram_counts, trigram, unigram_counts):
    # для каждого слова из триграммы берется частота из счетчика слов и сохраняется в trigram_count
    word1, word2, word3 = trigram.split()
    unigram1_count = unigram_counts[word1]
    unigram2_count = unigram_counts[word2]
    unigram3_count = unigram_counts[word3]
    trigram_count = trigram_counts[trigram]
    total_count = sum(trigram_counts.values())
    # формула для расчета
    # Взаимная информация - оценка степени связи между триграммой и контекстом
    mutual_information = math.log2((trigram_count*total_count) / (unigram1_count*unigram2_count*unigram3_count))
    return mutual_information

# Mutual Information > 0: появление триграммы в этом контексте является вероятным, связь есть
# Mutual Information = 0: отсутствие связи между триграммой и контекстом
# Mutual Information < 0: негативная связь между триграммой и контекстом


# Расчет log-likelihood (логарифмическая правдоподобность)
# На вход принимает тоже самое, что и в предыдущей функции
# Основывается на методе максимального правдоподобия
def calculate_log_likelihood(trigram_counts, trigram, unigram_counts):
    word1, word2, word3 = trigram.split()
    unigram1_count = unigram_counts[word1]
    unigram2_count = unigram_counts[word2]
    unigram3_count = unigram_counts[word3]
    trigram_count = trigram_counts[trigram]
    total_count = sum(trigram_counts.values())
    expected_count = (unigram1_count*unigram2_count*unigram3_count) / total_count
    log_likelihood = 2 * (trigram_count * math.log2(trigram_count / expected_count))
    return log_likelihood


# Сохранение значений метрик в отдельный файл
def save_measures(trigram_counts, unigram_counts, n):
    n_grams_folder = 'n_grams'
    os.makedirs(n_grams_folder, exist_ok=True)
    top_n_trigrams = trigram_counts.most_common(n)
    with open(os.path.join(n_grams_folder, 'measures.txt'), 'w') as file:
        for trigram, count in top_n_trigrams:
            mutual_information = calculate_mutual_information(trigram_counts, trigram, unigram_counts)
            log_likelihood = calculate_log_likelihood(trigram_counts, trigram, unigram_counts)
            file.write(f'{trigram}\t{count}\tMutual Information: {mutual_information}\tLog-likelihood: {log_likelihood}\n')

folder_path = 'C:/Users/n.strokova/Pictures/ITMO/nlp/assets/raw-dataset'
trigram_counts, unigram_counts = process_folder(folder_path)

save_trigrams(trigram_counts)
save_top_n_trigrams(trigram_counts, 30)
save_measures(trigram_counts, unigram_counts, 30)

