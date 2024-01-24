import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# загрузка обученной модели
model = Word2Vec.load("w2v_model")

# примеры токенов для построения диаграммы
initial_tokens = ["basketball"]
similar_tokens = ["football", "soccer"]
same_field_tokens = ["sport", "players"]
other_tokens = ["yesterday", "fresh", "rules"]

tokens = {
    "initial": initial_tokens,
    "similar": similar_tokens,
    "same_field": same_field_tokens,
    "other": other_tokens
}


# функция для получения векторного представления токена
def get_token_vector(token):
    if token in model.wv:
        return model.wv[token]
    else:
        return np.zeros(model.vector_size)


# вычисление косинусного расстояния между векторными представлениями
results = {}

for token, token_group in tokens.items():
    token_vectors = [get_token_vector(t) for t in token_group]
    cos_sim = [cosine_similarity([get_token_vector(token)], [tv])[0][0] for tv in token_vectors]
    results[token] = list(zip(token_group, cos_sim))

# вывод ранжированного списка
for token, sim_tokens in results.items():
    sorted_sim_tokens = sorted(sim_tokens, key=lambda x: x[1], reverse=True)
    print(f"Token: {token}")
    for sim_token, cos_sim in sorted_sim_tokens:
        print(f"Similar Token: {sim_token}, Cosine Similarity: {cos_sim}")
    print()

# преобразование векторов в двумерные массивы
vectors = np.array([get_token_vector(token) for group in tokens.values() for token in group])
labels = [token for group in tokens.values() for token in group]

# построение точечной диаграммы
plt.scatter(vectors[:, 0], vectors[:, 1])

# добавляем текстовые метки для каждого токена
for i, label in enumerate(labels):
    plt.annotate(label, (vectors[i, 0], vectors[i, 1]))

# сохранение диаграммы в файл
plt.savefig("basketball_example_plot.png")
