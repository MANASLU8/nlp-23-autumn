import os
import string
import nltk
import pandas as pd
import math
import re
from nltk.collocations import TrigramCollocationFinder
import warnings
from collections import Counter
from pathlib import Path
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.collocations import *
from tokenizer import get_wordnet_pos
import pandas as pd


#Set up 
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
warnings.filterwarnings('ignore')
lemmatizer = WordNetLemmatizer()
all_lemmas = []
stop_words = set(stopwords.words('english'))


#Return lemmas from a text sample
def preprocess_text(text):
    list_lemmas = []
    text=text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    for word in text.split():
            list_lemmas.append(lemmatizer.lemmatize(word, get_wordnet_pos(word)))
    return list_lemmas


#Generate n-gams for a text sample (n = 1,2,3)
def generate_ngrams(words_list, n):
    ngrams_list = []
    stop = 0
    if (n != 1):
        stop = n-1
    for num in range(0, len(words_list)-stop):
        ngram = ' '.join(words_list[num:num + n])
        ngrams_list.append(ngram)
    return ngrams_list


#Generate and calculate a trigrams. 
def process_trigrams(file_name):
    print('Get 3-grams from: ', file_name)
    trigrams = []
    df = pd.read_csv(file_name, sep=',', header=None)
    data = df.values
    trigram_df = pd.DataFrame(columns=['trigram', 'count'])
    count = 0

    for row in data:
        for i in range(1, len(row)-1):
            text = row[i]
            preproc_txt = preprocess_text(text)
            all_lemmas.append(preproc_txt)
            trigrams = generate_ngrams(preproc_txt, 3)
            for tgram in trigrams:
                if (tgram in trigram_df['trigram'].values):
                    count = trigram_df.loc[trigram_df['trigram'] == tgram, 'count'] + 1
                    trigram_df.loc[trigram_df['trigram'] == tgram] = [tgram, count]
                else:
                    trigram_df.loc[len(trigram_df.index)] = [tgram, 1] 
        trigrams = []
    #top 30 print 
    final_df = trigram_df.sort_values(by=['count'], ascending=False)
    print("Top 30 trigrams: ")
    print(final_df.head(30))

    try:
        dir_path = "D:/labs/nlp/assets/n-grams/" + Path(file_name).name.split('.')[0] + "/"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        f = open(dir_path + '3_count.csv', 'w+')
        f.truncate(0)
        print('Write to file')
        final_df.to_csv(dir_path + '3_count.csv')
    except Exception as e:
        print(e)
        print([text, trigrams])
        pass

    return final_df



def count_trigrams(text):
    words = re.findall(r'\b\w+\b', text.lower())
    trigrams = zip(words, words[1:], words[2:])
    trigram_counts = Counter(trigrams)
    counts_list = list(trigram_counts.items())
    counts_list.sort(key=lambda x: x[1], reverse=True)
    return counts_list

def count_unigrams(text):
    words = re.findall(r'\b\w+\b', text.lower())
    word_counts = Counter(words)
    counts_list = list(word_counts.items())
    counts_list.sort(key=lambda x: x[1], reverse=True)
    return counts_list

def calculate_loglike(text, trigram):
    trigram_count = count_trigrams(text)
    unigram_count = count_unigrams(text)
    total_count = sum(count for _, count in trigram_count)
    found_count = 0
    for count in trigram_count:
        if count[0] == trigram:
            found_count = count[1]
            print("This trigram founds: " + str(found_count))
            break
    trig_q = found_count / total_count
    x_q = unigram_count[0][1] / total_count
    y_q = unigram_count[1][1] / total_count
    z_q = unigram_count[2][1] / total_count
    x_q_z = x_q * z_q
    y_q_z = y_q * z_q
    trig_q_ind = x_q_z * y_q_z
    if trig_q_ind == 0:
        return float('-inf')  
    llr_value = 2 * total_count * (trig_q * math.log2(trig_q / trig_q_ind))
    return llr_value

def calculate_t_score(text):
    trigram= count_trigrams(text)
    unigram = count_unigrams(text)
    total_count = sum(count for _, count in unigram)
    unigram_count = sum(count for _, count in unigram)
    trigram_count = sum(count for _, count in trigram)
    return (trigram_count  - (unigram_count / (total_count ** 2))) / (trigram_count + 1e-5) ** 0.5

#To test some tasks from lab
def lab_2():
    file_name = 'D:/labs/nlp/assets/raw-dataset/test.csv'
    print('STUDENT')
    trigram_df = process_trigrams(file_name)
    
    print("T-SCORE")
    t_score_student = []
    for sentence in all_lemmas: 
        t_score_student.append(calculate_t_score(str(sentence)))
    print(t_score_student)
        
    print('NLTK')
    #Get trigrams
    finder = TrigramCollocationFinder.from_documents(all_lemmas)
    print(finder.nbest(nltk.collocations.TrigramAssocMeasures().student_t, 30))
    #Get trigram and score 
    for i in finder.score_ngrams(nltk.collocations.TrigramAssocMeasures().student_t):
        print(i)