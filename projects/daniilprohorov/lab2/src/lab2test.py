import nltk
from nltk.collocations import *
from nltk.corpus import PlaintextCorpusReader
import pandas as pd
from math import pow
import csv
import glob
import hashlib
import matplotlib.pyplot as plt



# df = pd.read_csv('../assets/test.csv', header=None)
df = pd.read_csv('../assets/train.csv', header=None)


def lines(df):
    return df[2].count()


def str_by_id(id, df):
    if id < df[2].count():
        return (df[0].values[id], df[2].values[id] + ' ')
    else:
        raise Exception("id > items in file")


def t_score(threegram, freqs_dict, dictionary):
    n = 3
    count = freqs_dict[threegram]
    multiplication = 1
    for lemma in threegram:
        lemma_id = dictionary.token2id[lemma]
        multiplication *= dictionary.cfs[lemma_id]
    part = multiplication / pow(dictionary.num_pos, n - 1)
    return (count - part) / pow(count, 1 / n)


classes = ['0', '1', '2', '3']
baseDir = '../../lab1/assets/annotated-corpus/train/'
words = []
for cls in classes:
    filenames = glob.glob(baseDir + cls + '/*.tsv')
    for filename in filenames:
        with open(filename) as fd:
            rd = csv.reader(fd, delimiter='\t')
            for row in rd:
                words.append(row[0].lower())
#print(words)
trigrams = []
for i in range(len(words) - 2):
    trigrams.append([words[i], words[i + 1], words[i + 2]])

#print(trigrams[-1])


def frq_calculation(trigrams):
    storage = {}
    md = hashlib.md5()
    for trigram in trigrams:
        trigramConcat = ''.join(trigram)
        #md.update(trigramConcat.encode('UTF-8'))
        #hash = md.hexdigest()
        if trigramConcat in storage:
            count, trigram_ = storage[trigramConcat]
            storage[trigramConcat] = (count + 1, trigram_)
        else:
            storage[trigramConcat] = (1, trigram)
    return storage


storage = frq_calculation(trigrams)
storage_values = list(storage.values())
storage_sorted = sorted(storage_values, key=lambda x: x[0], reverse=True)

fig, ax = plt.subplots()

fruits = ['apple', 'blueberry', 'cherry', 'orange']
counts, trigrams_list = zip(*storage_sorted[:30])
trigrams_str = [' '.join(trg) for trg in trigrams_list]
#
print(counts)
print(trigrams_str)
ax.bar(trigrams_str, counts, width=0.2)
#
# ax.set_ylabel('fruit supply')
# ax.set_title('Fruit supply by kind and color')
# ax.legend(title='Fruit color')
#
plt.show()
# testText = ' '.join([str_by_id(i, df)[1] for i in range(100)])
#
# bigram_measures = nltk.collocations.BigramAssocMeasures()
# trigram_measures = nltk.collocations.TrigramAssocMeasures()
#
# f = open('text.txt')
# raw = f.read()
#
# tokens = nltk.word_tokenize(testText, 'english', True)
#
# text = nltk.Text(tokens)
# print(text)
#
# http://www.nltk.org/_modules/nltk/collocations.html
# finder_bi = BigramCollocationFinder.from_words(text)
# finder_thr = TrigramCollocationFinder.from_words(text)
#
# print(finder_bi.nbest(bigram_measures.pmi, 10))
# print(finder_thr.nbest(trigram_measures.pmi, 10))
