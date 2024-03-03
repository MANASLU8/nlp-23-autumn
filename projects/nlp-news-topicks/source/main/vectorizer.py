import os
import re
import numpy as np
import nltk
from tokenizer import tokenize_text


#Create term_document_matrix and sorted_tokens
def term_doc_matrix_create(path_folder):
    print('Create term doc matrix from: ', path_folder) 
    tokens_list = {}

    for category in ['1', '2', '3', '4']:
        join_folder = os.path.join(path_folder, category)
        for file_name in os.listdir(join_folder):
            if file_name.endswith(".tsv"):
                file_path = os.path.join(join_folder, file_name)
                with open(file_path, 'r', encoding='utf-8') as fp:
                    for line in fp:
                        tokens = tokenize_text(line)
                        for token in tokens:
                            nan, kind, w = token
                            if kind == 'word':
                                token = re.sub(r'[^\w\s]', '', w.lower())
                                if token in tokens_list:
                                    tokens_list[token] += 1
                                else:
                                    tokens_list[token] = 1
    sorted_tokens = dict(sorted(tokens_list.items(), key=lambda item: item[1], reverse=True))
    
    print('Save to: sorted tokens')
    with open('D:/labs/nlp/assets/lab_3/term', 'w') as file:
        for key, value in sorted(tokens_list.items(), key=lambda item: item[1], reverse=True):
            file.write(f'{key}: {value}\n')

    term_doc_matrix = np.zeros((len(sorted_tokens), 4))
    for i, category in ['1', '2', '3', '4']:
        join_folder = os.path.join(path_folder, category)
        for file_name in os.listdir(join_folder):
            if file_name.endswith(".tsv"):
                file_path = os.path.join(join_folder, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        tokens = tokenize_text(line)
                        token_set = set()
                        for token in tokens:
                            nan, kind, part = token
                            if kind == 'word':
                                token = re.sub(r'[^\w\s]', '', part.lower())
                                token_set.add(token)

                        for token in token_set:
                            if token in sorted_tokens:
                                term_doc_matrix[list(sorted_tokens.keys()).index(token), i] += 1

    print('Save to: term_document_matrix')
    with open('D:/labs/nlp/assets/lab_3/term_document_matrix', 'w') as file:
        for row in term_doc_matrix:
            line = ' '.join(map(str, row)) + '\n'
            file.write(line)
    return 0


#Get tfidf from text sample 
def text_to_tfidf(text, token_dict, term_doc_matrix):
    print('Get tfidf from text')
    tokens = tokenize_text(text)
    tfidf_vector = np.zeros(len(token_dict))
    token_count = {}

    for token in tokens:
        nan, kind, part = token
        if kind == 'word':
            cleaned_token = re.sub(r'[^\w\s]', '', part.lower())
            if cleaned_token in token_dict:
                if cleaned_token in token_count:
                    token_count[token] += 1
                else:
                    token_count[token] = 1

    for token, frequency in token_count.items():
        token_index = token_dict[token]
        if token_index >= len(term_doc_matrix):
            continue
        tfidf_vector[token_index] = frequency * np.log(1 + len(term_doc_matrix[0]) / np.count_nonzero(term_doc_matrix[token_index]))

    return tfidf_vector / np.linalg.norm(tfidf_vector)


#Vectirize text sample
def vectorize_text(text, model):
    sentences = nltk.sent_tokenize(text)
    tokens_sentences = [tokenize_text(sent) for sent in sentences]
    token_vectors = []

    for sentence_tokens in tokens_sentences:
        sentence_vectors = []
        for token in sentence_tokens:
            if token in model.wv.key_to_index.keys():
                sentence_vectors.append(model.wv[token])
        token_vectors.append(sentence_vectors)
    sentence_vectors = []

    for sentence_tokens in token_vectors:
        if len(sentence_tokens) > 0:
            sentence_vectors.append(sum(sentence_tokens) / len(sentence_tokens))
        else:
            sentence_vectors.append([0] * len(model.wv[model.wv.index_to_key[0]]))

    doc_vector = np.mean(sentence_vectors, axis=0)
    return doc_vector
    

#Vectorize test samples
def vectorize_test_sampes(test_folder, model_path):
    print("Start test samples vectorization")
    doc_vectors = []

    for category in ['1', '2', '3', '4']:
        join_folder = os.path.join(test_folder, category)
        result_folder = []
        for file_name in os.listdir(join_folder):
            if file_name.endswith(".tsv"):
                file_path = os.path.join(join_folder, file_name)
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()

                vect = vectorize_text(text, model_path)
                result_folder.append((os.path.splitext(file_name)[0], vect))
        
        output_file = 'D:/labs/nlp/assets/lab_3/test/all_vectors_test.tsv'
        with open(output_file, 'w', encoding='utf-8') as file:
            for category, vect in result_folder:
                line = f"{category}\t" + "\t".join(str(val) for val in vect)
                file.write(line + '\n')

        doc_vectors.extend(result_folder)

    print("Finished test samples vectorization")