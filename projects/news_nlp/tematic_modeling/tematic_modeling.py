#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import make_multilabel_classification


# In[ ]:


all_lemmas_test = []

# обработка файлов в папке с каждой категорией
for folder in tqdm(['1', '2', '3', '4']):
    folder_path = os.path.join('assets/annotated_corpus/test/', folder)
    for file in os.listdir(folder_path):
        try:
            # проверяем, является ли файл tsv
            if file.endswith('.tsv') and file.startswith('annotation'):
                file_path = os.path.join(folder_path, file)
                df = pd.read_csv(file_path, sep='\t', header=None)
                lemma_list = df[0].tolist()
                # список для хранения лемм текущего предложения
                sentence_lemmas = []
                for lemma in lemma_list:
                    if str(lemma) != 'nan':
                        sentence_lemmas.append(lemma)
                    else:
                        all_lemmas.append(sentence_lemmas)
                        sentence_lemmas = []

                if len(sentence_lemmas) > 0:
                    all_lemmas.append(sentence_lemmas)
        except: Exception


# In[ ]:


# обращение к term_document_matrix и terms
with open("term_document_matrix", "rb") as fp:
    term_document_matrix = pickle.load(fp)
    
with open("terms", "rb") as fp: 
    terms = pickle.load(fp)


# In[ ]:


# создаем аннотируемый корпус на основе распределения тем для каждого документа
def create_annotation(doc_topic_dist, n_comps):
    # doc_topic - распределение тем для каждого документа
    # n_comps - кол-во тем, которые должны быть использованы для аннотации
    output_filename = f"assets/annotated_corpus/test/probs_topics_{n_comps}.tsv"
    with open(output_filename, "w", encoding="utf-8") as f:
        for index, probs in zip(range(len(doc_topic_dist)),doc_topic_dist):
                    # запись индекса
                    f.write(f"{index}\t") 
                    for prob in probs:
                        # запись вероятности
                        f.write(f"{prob}\t")
                    f.write("\n")  


# In[ ]:


# непосредственно тематическое модедирование с помощью LDA
def lda_for_comps(lda_comp, term_document_matrix, max_iter):
    # lda_comp - кол-во компонент для модели
    # term_document_matrix - матрица из 3-ей ЛР (коллекция документов в виде частот)
    # max_iter - максимольное кол-во итераций для обучения LDA
    lda = LatentDirichletAllocation(n_components=lda_comp, random_state=0, max_iter=max_iter)
    # обучение модели
    lda.fit(term_document_matrix)
    # словарь топ-слов для каждой темы
    topic_words = {}
    n_top_words= 10
    # расчет распределения вероятностей тем для каждого документа
    doc_topic_dist = lda.transform(term_document_matrix)
    # выбор топ-10 слов из каждой темы
    for topic, comp in enumerate(lda.components_):  
        word_idx = np.argsort(comp)[::-1][:n_top_words]
        topic_words[topic] = [terms[i] for i in word_idx]
        # запись в файл
        with open(f'lab4/top_words_{lda_comp}_maxiter_{max_iter}', "w", encoding="utf-8") as f:
            top_docs = np.argmax(doc_topic_dist, axis=0)
            for topic, words in topic_words.items():
                f.write('Topic: %d \n' % topic)
                f.write('Top words: %s' % ', '.join(words))
                f.write('\n')
                f.write('Top text:\n'+' '.join(all_lemmas_test[top_docs[topic]]) +'\n\n')
    create_annotation(doc_topic_dist, lda_comp)  
    # перплексия измеряет ожидаемую сложность модели в предсказании новых наблюдений
    # чем более низкая перплексия, тем лучше модель
    return lda.perplexity(term_document_matrix)


# In[ ]:


# задание входных параметров
lda_comps = [2, 5, 10, 20, 40]
perplexity = [lda_for_comps(lda_comp, term_document_matrix, max_iter=10) for lda_comp in tqdm(lda_comps)]
perplexity


# In[ ]:


# построение графика
plt.figure(dpi=280, figsize=(8,4))
plt.plot(lda_comps, perplexity, color='red')
pol = np.poly1d(np.polyfit(lda_comps, perplexity, 5))
plt.plot(lda_comps, [pol(lda_comp) for lda_comp in lda_comps], color='blue')
plt.xlabel("n_comps")
plt.ylabel("perplexity")
plt.show()


# ### 5 итераций (в первом случае было 10) 

# In[ ]:


perplexity = [lda_for_comps(lda_comp, term_document_matrix, max_iter=5) for lda_comp in tqdm(lda_comps)]


# In[ ]:


plt.figure(dpi=280, figsize=(8,4))
plt.plot(lda_comps, perplexity, color='red')
pol = np.poly1d(np.polyfit(lda_comps, perplexity, 5))
plt.plot(lda_comps, [pol(lda_comp) for lda_comp in lda_comps], color='blue')
plt.xlabel("n_comps")
plt.ylabel("perplexity")
plt.show()


# ### 20 итераций (в первом случае было 10)

# In[ ]:


perplexity = [lda_for_comps(lda_comp, term_document_matrix, max_iter=20) for lda_comp in tqdm(lda_comps)]


# In[ ]:


plt.figure(dpi=280, figsize=(8,4))
plt.plot(lda_comps, perplexity, color='red')
pol = np.poly1d(np.polyfit(lda_comps, perplexity, 5))
plt.plot(lda_comps, [pol(lda_comp) for lda_comp in lda_comps], color='blue')
plt.xlabel("n_comps")
plt.ylabel("perplexity")
plt.show()

