import os
import math
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


#Count cos distance 
def cos_distance(vector1, vector2):
    vectors_dot_product = np.dot(vector1, vector2)
    vec1_sq = np.sqrt(np.dot(vector1, vector1).sum())
    vec2_sq = np.sqrt(np.dot(vector2, vector2).sum())
    cos_similarity = math.fabs(np.divide(vectors_dot_product, vec1_sq * vec2_sq))
    cos_distance = 1 - cos_similarity
    return cos_distance

def get_token_vector(model, token):
    print(f'Get vector for {token}')
    if token in model.wv:
        return model.wv[token]
    else:
        return np.zeros(model.vector_size)

def save_plot(vectors, labels):
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(vectors)

    print('Create a plots')
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1])
    for i, label in enumerate(labels):
        plt.annotate(label, (reduced_vectors[i, 0], reduced_vectors[i, 1]))
    plt.savefig("assets/lab_3/PCA_plot.png")
    plt.clf()

    plt.scatter(vectors[:, 0], vectors[:, 1])
    for i, label in enumerate(labels):
        plt.annotate(label, (vectors[i, 0], vectors[i, 1]))
    plt.savefig("assets/lab_3/plot.png")


def work_with_model():
    print('W2V set up')
    sentences = []

    print('Get tokens from files')
    for category in ['1', '2', '3', '4']:
        category_sentences = []
        files = os.listdir(f'D:/labs/nlp/assets/train/{category}')
        for file in files:
            file_path = os.path.join(category, file)

            with open(file_path, "r") as f:
                tokens = [simple_preprocess(line.strip()) for line in f]
                category_sentences.extend(tokens)
        sentences.extend(category_sentences)
    print('Save w2v model')
    model = Word2Vec(sentences, min_count=1, vector_size=100)
    model.save("assets/lab_3/w2v_model")

    print('Tokens example')
    results = {}
    tokens = {
        "initial": ["robbed"],
        "similar": ["jailed", "arrest"],
        "same_field": ["kidnapping", "crime"],
        "other": ["race", "man", "yesterday"]
    }
    print(tokens)

    print('Calculate cosine similarity')
    for token, token_group in tokens.items():
        token_vectors = [get_token_vector(model,t) for t in token_group]
        cos_sim = [cosine_similarity([get_token_vector(model,token)], [tv])[0][0] for tv in token_vectors]
        results[token] = list(zip(token_group, cos_sim))
    #Print it to console
    for token, data in results.items():
        print(f"Token: {token}")
        for token_group, cos_sim in data:
            print(f"   Token Group: {token_group}, Cosine Similarity: {cos_sim}")
        print()

    # Vec to array
    vectors = np.array([get_token_vector(model,token) for group in tokens.values() for token in group])
    labels = [token for group in tokens.values() for token in group]
    
    #Plot create 
    save_plot(vectors, labels)