import nltk
from scipy.constants import degree
from nltk.tokenize import word_tokenize
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
from gensim import corpora, models
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import os
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

term_document_matrix = np.load('C:/Users/n.strokova/Pictures/ITMO/nlp/assets/train-td-matrix.npz')


folders = ['C:/Users/n.strokova/Pictures/ITMO/nlp/assets/train/Business', 'C:/Users/n.strokova/Pictures/ITMO/nlp/assets/train/Sci-Tech', 'C:/Users/n.strokova/Pictures/ITMO/nlp/assets/train/Sports', 'C:/Users/n.strokova/Pictures/ITMO/nlp/assets/train/World']
documents = []
labels = []

for folder in folders:
    files = os.listdir(folder)
    for file in files:
        file_path = os.path.join(folder, file)
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                tokens = word_tokenize(line.strip())
                documents.append(tokens)
                labels.append(folder)

dictionary = corpora.Dictionary(documents)
corpus = [dictionary.doc2bow(doc) for doc in documents]

# Загрузка тестовых документов и их классов
test_folders = ['C:/Users/n.strokova/Pictures/ITMO/nlp/assets/test/Business', 'C:/Users/n.strokova/Pictures/ITMO/nlp/assets/test/Sci-Tech', 'C:/Users/n.strokova/Pictures/ITMO/nlp/assets/test/Sports', 'C:/Users/n.strokova/Pictures/ITMO/nlp/assets/test/World']
test_documents = []
test_labels = []

for test_folder in test_folders:
    files = os.listdir(test_folder)
    for file in files:
        file_path = os.path.join(test_folder, file)
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                tokens = word_tokenize(line.strip())
                test_documents.append(tokens)
                test_labels.append(None)  # Здесь можно задать метку класса тестовых документов, если она есть
def get_top_keywords(model, num_keywords=10):
    topics = model.show_topics(num_topics=-1, num_words=num_keywords, formatted=False)
    top_keywords = []
    for topic in topics:
        keywords = [word for word, prob in topic[1]]
        top_keywords.append(keywords)
    return top_keywords

def save_document_probabilities(model):
    document_probas = []
    for doc_bow in corpus:
        doc_proba = np.zeros(model.num_topics)
        for topic, prob in model[doc_bow]:
            doc_proba[topic] = prob
        document_probas.append(doc_proba.tolist())
    np.save('document_probabilities.npy', document_probas)

num_clusters = [2, 5, 10, 20, 40]
perplexity_values = []

for num_topics in num_clusters:
    print(f"Experiment for {num_topics} topics is in progress...")
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary)

    top_keywords = get_top_keywords(lda_model)
    print(f'Top keywords for {num_topics} topics:')
    for topic, keywords in enumerate(top_keywords):
        print(f'Topic {topic + 1}: {keywords}')

    test_corpus = [dictionary.doc2bow(doc) for doc in test_documents]
    perplexity = lda_model.log_perplexity(test_corpus)
    print(f'Perplexity for {num_topics} topics: {perplexity}')

    perplexity = lda_model.log_perplexity(test_corpus)
    perplexity_values.append(perplexity)
    print(f'Perplexity for {num_topics} topics: {perplexity}')

    save_document_probabilities(lda_model)

    print(f"Experiment for {num_topics} topics is completed.")


# Построение графика perplexity от количества тем
plt.plot(num_clusters, perplexity_values, 'o-')
plt.xlabel('Number of Topics')
plt.ylabel('Perplexity')
plt.title('Perplexity vs Number of Topics')
plt.savefig('perplexity_graph.png')

# Полиномиальная аппроксимация графика
coefficients = np.polyfit(num_clusters, perplexity_values, degree)
polynomial = np.polyval(coefficients, num_clusters)
plt.plot(num_clusters, perplexity_values, 'o-', label='Perplexity')
plt.plot(num_clusters, polynomial, '-', label='Polynomial Approximation')
plt.xlabel('Number of Topics')
plt.ylabel('Perplexity')
plt.title('Perplexity vs Number of Topics with Polynomial Approximation')
plt.legend()
plt.savefig('polynomial_approximation.png')