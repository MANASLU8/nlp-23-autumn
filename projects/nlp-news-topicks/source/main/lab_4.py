import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import LatentDirichletAllocation

#Set up
all_lemmas_test = []
with open("D:/labs/nlp/assets/annotated-corpus/terms", "r", encoding="utf-8") as f:
    terms = f.read()
with open("D:/labs/nlp/assets/annotated-corpus/term_document_matrix", "r", encoding="utf-8") as f:
    term_document_matrix = f.read()


def get_annotation(doc_topic, n_comps):
    output_filename = f"D:/labs/nlp/assets/annotated_corpus/test/probs_topics_{n_comps}.tsv"
    with open(output_filename, "w", encoding="utf-8") as f:
        for index, probs in zip(range(len(doc_topic)),doc_topic):
                    f.write(f"{index}\t") 
                    for prob in probs:
                        f.write(f"{prob}\t")
                    f.write("\n")  


def get_lda_for_comps(lda_comp, term_doc_matrix, max_iter):
    lda = LatentDirichletAllocation(n_components=lda_comp, random_state=0, max_iter=max_iter)
    lda.fit(term_doc_matrix)
    topic_words = {}
    n_top_words= 10
    doc_topic = lda.transform(term_doc_matrix)

    for topic, comp in enumerate(lda.components_):  

        word_idx = np.argsort(comp)[::-1][:n_top_words]
        topic_words[topic] = [terms[i] for i in word_idx]

        with open(f'D:/labs/nlp/assets/lab4/top_words_{lda_comp}_{max_iter}', "w", encoding="utf-8") as f:

            top_docs = np.argmax(doc_topic, axis=0)
            for topic, words in topic_words.items():
                f.write('Topic: %d \n' % topic)
                f.write('\nTop words: %s' % ', '.join(words))
                f.write('\nTop text:\n'+' '.join(all_lemmas_test[top_docs[topic]]) +'\n\n')
            
    get_annotation(doc_topic, lda_comp)
    return lda.perplexity(term_doc_matrix)


def show(lda_comps, perplexity,max_iter):
    plt.figure(dpi=280, figsize=(8,4))
    plt.plot(lda_comps, perplexity, color='red')   
    #approximate that minimises the squared error
    pol = np.poly1d(np.polyfit(lda_comps, perplexity, 5))
    plt.plot(lda_comps, [pol(lda_comp) for lda_comp in lda_comps], color='blue')
    plt.xlabel("n_comps")
    plt.ylabel("perplexity")
    plt.savefig(f'D:/labs/nlp/assets/lab4/top_itter_{max_iter}')


def lab_4(max_iter):
    print('Start get data from files')
    print('Number of iterrations: ', max_iter)

    lda_comps = [5, 8, 10, 15, 20, 25, 30, 35, 40]
    perplexity = [get_lda_for_comps(lda_comp, term_document_matrix, max_iter) for lda_comp in tqdm(lda_comps)]
    
    show(lda_comps, perplexity, max_iter)

    print('Save results to plot')