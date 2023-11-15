from re import Pattern
from typing import AnyStr
import pandas as pd
import re
from numpy import random
from pipe import dedup, where, select, sort, chain, map
from pathlib import Path
import sys
import csv
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

trainFile = Path('.') / 'resources' / 'train.csv'
testFile = Path('.') / 'resources' / 'test.csv'
nltk.download('wordnet')

lemmer = WordNetLemmatizer()
stemmer = SnowballStemmer('english')


def replace_str(patterns_replace, input_str):
    process_str = input_str
    for p, r_str in patterns_replace:
        process_str = re.sub(p, r_str, process_str)
    return process_str


random.seed(1070)
patterns: [Pattern[AnyStr]] = [
    # '(?:(?:[A-Z][a-z]+\s){2,})'  # Имена компаний
    '[a-zA-Z0-9]+-[a-zA-Z0-9\']+',  # word with -
    '(?:[A-Z]+_?){2,}',  # specialized tokens
    '[a-zA-Z\']+',  # word
    '[+-]?[0-9]+[,.][0-9]+',  # float number
    '[+-]?[0-9]+'  # int number
]
patterns_delete: [Pattern[AnyStr]] = [
    '\.,'
]
patterns_replace: [(Pattern[AnyStr], str)] = [
    (r'\.\s', ' PUNCTUATION_MARK_POINT '),
    # (r'\.', ' PUNCTUATION_MARK_POINT '),
    (r',\s', ' PUNCTUATION_MARK_COMMA '),
    (r'\?', ' PUNCTUATION_MARK_QUESTION '),
    (r'\!', ' PUNCTUATION_MARK_EXCLAMATION '),
    ('\s-\s', ' PUNCTUATION_MARK_IS ')
]
pattern = re.compile('|'.join(patterns))
pattern_delete = re.compile('|'.join(patterns_delete))
df = pd.read_csv(trainFile, header=None)


# df = pd.read_csv(testFile, header=None)


def lines(df):
    return df[2].count()


def str_by_id(id, df):
    if id < df[2].count():
        return (df[0].values[id], df[2].values[id] + ' ')
    else:
        raise Exception("id > items in file")


def show(x, label=None):
    if label is not None:
        print(label)
    print(x)
    return x


def sstate(label=None):
    return map(lambda x: show(x, label))


def to_tokens(input_str):
    return list([input_str]
                # | sstate('Start:')
                | map(lambda x: pattern_delete.sub('', x))
                # | sstate('Deleted:')
                # | map(lambda x: replace_str(patterns_replace, x))
                # | sstate('Replaced tokens:')
                | map(lambda x: pattern.findall(x))
                # | sstate('Result tokens:')

                # | map(lambda lst: [(x, stemmer.stem(x), lemmer.lemmatize(x)) for x in lst])
                | map(lambda lst: [[x.lower(), stemmer.stem(x).lower(), lemmer.lemmatize(x).lower()] for x in lst])
                # | sstate('Result:')
                )


loading_state = 0


def loading_indication(msg):
    global loading_state
    if loading_state < 3:
        loading_state += 1
    else:
        loading_state = 0
    symbols = ['|', '/', '-', '\\']
    sys.stdout.write("\033[F")  # Cursor up one line
    sys.stdout.write("\033[K")  # clear
    print(msg + ' ' + symbols[loading_state])


str_id = random.randint(df[2].count() - 1)
input_str = df[2].values[str_id] + ' '

# dict creation for lab 2
dictLst = [[] for i in range(50)]


def add_to_dict(s):
    sLength = len(s)
    if len(dictLst) < sLength:
        raise Exception('Super large word')
    dictLst[sLength].append(s)
    loading_indication('Add words to dict...')


def dict_to_files():
    basePath = Path('.') / 'resources' / 'words'
    for s in [s for s in dictLst if s]:
        lst = list(set(s))
        filename = str(len(lst[0])) + '.csv'
        print(filename)
        fullPath = basePath / filename
        for w in lst:
            with open(fullPath, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([w])
            loading_indication('Write words to files...')


def add_tokens_to_file(tokens, class_id, i):
    basePath = Path('.') / 'resources' / str(class_id)
    for token in tokens:
        filename = str(i) + '.tsv'
        full_path = basePath / filename
        token_str = '\t'.join(token)
        with open(full_path, 'w', newline='') as f:
            f.write(token_str)
            f.write('\n')


for i in range(lines(df)):
    class_id, input_str = str_by_id(i, df)
    tokens = to_tokens(input_str)[0]
    add_tokens_to_file(tokens, class_id, i)
    loading_indication(str(i) + ' Write words to files...')
