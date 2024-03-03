import re
import os


def split_to_sentences(text):
    sentences = re.split(
        r"(((?<!\w\.\w.)(?<!\s\w\.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s(?=[A-Z]))|((?<![\,\-\:])\n(?=[A-Z]|\" )))", text)[
                ::4]
    return sentences


def split_to_words(sentence):
    words = re.findall(r"\w+@\w+\.\w+|\+\d{1,3}-\d{3}-\d{3}-\d{2}-\d{2}|\w+", sentence)
    return words


def save_to_file(original, lemmatized, stemmed, id, path):
    with open(os.path.join(path, id), "w") as f:
        for i in range(len(original)):
            if original[i] == "\n":
                print("", file=f)
            else:
                print(original[i], stemmed[i], lemmatized[i], sep="\t", file=f)
