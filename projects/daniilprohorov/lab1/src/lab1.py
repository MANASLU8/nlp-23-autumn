import pandas as pd
import pathlib
from pathlib import Path
import random
import sys
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from typing import NamedTuple
import re
import argparse

nltk.download('wordnet')

lemmer = WordNetLemmatizer()
stemmer = SnowballStemmer('english')

parser = argparse.ArgumentParser(description='Lab 1 NLP Prohorov Daniil')
parser.add_argument('filename', type=str,
                    help='dataset file path')
parser.add_argument('-n', dest='n', type=int,
                    help='number of lines to process from dataset')

args = parser.parse_args()


class Token(NamedTuple):
    type: str
    value: str
    line: int
    column: int


def tokenize(code):
    token_specification = [
        ('TIME_12H', r'([0]?[0-9]|1[0-1]):[0-5][0-9]([ \t]*[aApP][ \.]*[mM][ \.]*)*'),  # with AM PM
        ('TIME_24H', r'([0-1]?[0-9]|2[0-3]):[0-5][0-9]'),
        ('NUMBER', r'\d+(\.\d*)?'),  # Integer or decimal(only with .)
        ('ASSIGN', r':='),  # Assignment operator
        ('COLON', r':'),  # Statement terminator
        ('SEMICOLON', r';'),  # Statement terminator
        ('DASH', r'-'),  # DASH
        ('WORD', r'[a-zA-Z\'-]+'),  # Identifiers
        ('OP', r'[+\-*/]'),  # Arithmetic operators
        ('HYPHEN_SHORT', r' - '),  # HYPHEN
        ('HYPHEN_LONG', r' – '),  # HYPHEN
        ('LEFT_ROUND_BRACKET', r'\('),  # (
        ('RIGHT_ROUND_BRACKET', r'\)'),  # )
        ('LEFT_SQUARE_BRACKET', r'\['),  # [
        ('RIGHT_SQUARE_BRACKET', r'\]'),  # ]
        ('LEFT_CURLY_BRACKET', r'\{'),  # {
        ('RIGHT_CURLY_BRACKET', r'\}'),  # }
        ('LEFT_ANGLE_BRACKET', r'\<'),  # <
        ('RIGHT_ANGLE_BRACKET', r'\>'),  # >
        ('QUOTATION_MARK_SINGLE', r'\''),  # '
        ('QUOTATION_MARK_DOUBLE', r'\"'),  # "
        ('NEWLINE', r'\n'),  # Line endings
        ('SKIP', r'[ \t]+'),  # Skip over spaces and tabs
        ('SKIP_POINT_COMMA', r'\.\,'),  # Skip .,
        ('SKIP_SLASHES', r'\\'),  # Skip \\
        ('POINT', r'\.'),  # .
        ('COMMA', r'\,'),  # ,
        ('UNDEFINED', r'.'),  # Any other character
    ]
    tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)
    line_num = 1
    line_start = 0
    for mo in re.finditer(tok_regex, code):
        kind = mo.lastgroup
        value = mo.group()
        column = mo.start() - line_start
        if kind == 'NUMBER':
            value = float(value) if '.' in value else int(value)
        elif kind == 'NEWLINE':
            line_start = mo.end()
            line_num += 1
            continue
        elif kind == 'SKIP':
            continue
        # elif kind == 'SKIP_SLASHES':
        #     continue
        elif kind == 'SKIP_POINT_COMMA':
            continue
        elif kind == 'UNDEFINED':
            continue
            # raise RuntimeError(f'{value!r} unexpected on line {line_num}')
        yield Token(kind, value, line_num, column)


df = pd.read_csv(args.filename, header=None)


def lines(df):
    return df[2].count()


def str_by_id(id, df):
    if id < df[2].count():
        return (df[0].values[id], df[2].values[id] + ' ')
    else:
        raise Exception("id > items in file")


# for debug purposes
# index = random.randrange(1000, 10000)
# print(index)
# statement = str_by_id(index, df)[1]
#
#
# statement = '11:59 14:25 78:99 12:00am 10:00pm 11:11 am 11:11 pm 11:11 \\a.m 11:11 AM 11:13 p.m. lol kek (cheburek)'
# print(statement)
# for token in tokenize(statement):
#     print(token)

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


def add_tokens_to_file(tokens, class_id, i):
    dataset_name = args.filename.split('/')[-1].split('.')[0]
    basePath = Path('..') / 'assets' / 'annotated-corpus' / dataset_name / str(class_id)
    pathlib.Path(basePath).mkdir(parents=True, exist_ok=True)
    for token in tokens:
        filename = str(i) + '.tsv'
        full_path = basePath / filename
        word = token.value
        stem = stemmer.stem(word)
        lem = lemmer.lemmatize(word)
        token_str = '\t'.join([word, stem, lem])
        with open(full_path, 'a', newline='') as f:
            f.write(token_str)
            f.write('\n')


count = None

if args.n is None:
    count = lines(df)
elif args.n < 0 or args.n > lines(df):
    parser.error("n < 0 or n > {}=lines in dataset".format(lines(df)))
else:
    count = args.n

for i in range(count):
    class_id, input_str = str_by_id(i, df)
    tokens = tokenize(input_str)
    words = [token for token in tokens if token.type == 'WORD']
    add_tokens_to_file(words, class_id, i)
    loading_indication(str(i+1) + ' Write words to files...')
