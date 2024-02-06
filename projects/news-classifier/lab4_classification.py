# %%
import os
import time
import random
import warnings
import numpy as np
from sklearn.svm import SVC
from ast import literal_eval
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
documents_vectors_path = os.path.join(os.getcwd(), 'drive', 'MyDrive', 'ITMO', 'sem_3', 'NLP', 'assets', 'annotated-corpus', 'test-embeddings.tsv')

# %% [markdown]
# # Task 1

# %%
# Чтение файла
with open(documents_vectors_path, 'r') as file:
  content = file.read()

# Перевод контента в словарь
vec_dict = {tuple(map(int, literal_eval(line.split('\t')[0]))): list(map(float, line.split('\t')[1:])) for line in content.split('\n') if line != ''}
print(vec_dict)

# %%
# Функция подсчета метрик
def calculate_metrics(true_labels, predicted_labels):
    tp = tn = fp = fn = 0
    for true, pred in zip(true_labels, predicted_labels):
        if true == 1 and pred == 1:
            tp += 1
        elif true == 0 and pred == 0:
            tn += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 1 and pred == 0:
            fn += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return accuracy, recall, precision, f1_score


# %%
# Получение X (вектора) и y (класс документа)
X = np.array(list(vec_dict.values()))
y = np.array([elem[1] for elem in vec_dict.keys()])

# Разбиение выборки на тестовую и тренировочную
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Проход по нескольким kernel functions
kernel_functions = ['linear', 'rbf', 'poly']

for function in kernel_functions:
  print(f"Эксперимент с использованием Kernel Function: {function}")

  # Создание модели
  svm_model = SVC(kernel=function)

  # Засечка веремни обучения
  start_time = time.time()

  # Обучение модели
  svm_model.fit(X_train, y_train)

  # Подсчет времени обучения
  training_time = time.time() - start_time

  # Предсказание на тестовой выборке
  y_pred = svm_model.predict(X_test)

  # Расчет метрик
  accuracy, recall, precision, f1_score = calculate_metrics(y_test, y_pred)

  print(f"Accuracy score is: {accuracy}\nPrecision score is: {precision}\nRecall score is: {recall}\nF1 score is: {f1_score}"
        f"\nTraining time is: {training_time}s\n\n")


# %% [markdown]
# # Task 2

# %% [markdown]
# У SVM нет понятия эпох. Это только у нейронных сетей. Их не было. На нашем датасете наилучшие результаты по качеству показала линейная (linear) функция. По времени обучения она немногим медленнее poly (незначительно). На бо́льших сетах данных ситуация может измениться.

# %% [markdown]
# # Task 3

# %% [markdown]
# 2. Сократить размерность векторных представлений до некоторого значения, зафиксировать характер зависимости значений метрик от новой размерности;

# %%
# Изменение размерности методом главных компонент
def reduce_dimensions(vector_representation, new_dimension):
  pca = PCA(n_components=new_dimension)
  reduced_vectors = pca.fit_transform(vector_representation)

  return reduced_vectors

# %%
# Изменение X по новой размерности
X_upd = reduce_dimensions(np.array(list(vec_dict.values())), 50)

# Все то же самое, что и в task 1
X_train, X_test, y_train, y_test = train_test_split(X_upd, y, test_size=0.2, random_state=42)

kernel_functions = ['linear', 'rbf', 'poly']

for function in kernel_functions:
  print(f"Эксперимент с использованием Kernel Function: {function}")

  svm_model = SVC(kernel=function)

  start_time = time.time()

  svm_model.fit(X_train, y_train)

  training_time = time.time() - start_time

  y_pred = svm_model.predict(X_test)

  accuracy, recall, precision, f1_score = calculate_metrics(y_test, y_pred)

  print(f"Accuracy score is: {accuracy}\nPrecision score is: {precision}\nRecall score is: {recall}\nF1 score is: {f1_score}"
        f"\nTraining time is: {training_time}s\n\n")



