from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import os

# пути к папкам с категориями
folder_paths = ['C:/Users/n.strokova/Pictures/ITMO/nlp/assets/train/Business',
                'C:/Users/n.strokova/Pictures/ITMO/nlp/assets/train/Sci-Tech',
                'C:/Users/n.strokova/Pictures/ITMO/nlp/assets/train/Sports',
                'C:/Users/n.strokova/Pictures/ITMO/nlp/assets/train/World']

# инициализация списка предложений
sentences = []

# проход по каждой папке
for folder_path in folder_paths:
    category_sentences = []

    # получение списка файлов в папке
    files = os.listdir(folder_path)

    # проход по каждому файлу в папке
    for file in files:
        file_path = os.path.join(folder_path, file)

        # загрузка и токенизация файла
        with open(file_path, "r") as f:
            tokens = [simple_preprocess(line.strip()) for line in f]
            category_sentences.extend(tokens)

    # добавление списка токенов к общему списку предложений
    sentences.extend(category_sentences)

# обучение модели word2vec
model = Word2Vec(sentences, min_count=1, vector_size=100)
# min_count=1 - учитываются все слова, даже если они только один раз встречаются в тексте
# vector_size=100 - каждое слово будет представлено в виде вектора размерностью 100
# при обучении используется метод "скользящего окна"

# сохранение обученной модели
model.save("w2v_model")
