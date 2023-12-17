import nltk
from nltk.stem import SnowballStemmer
nltk.download('snowball_data')
stemmer = SnowballStemmer("english")
def get_stemm(word):
    return stemmer.stem(word)