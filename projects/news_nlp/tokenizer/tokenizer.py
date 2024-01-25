# импорт библиотек
import re
import os
import nltk
import pandas as pd
from pathlib import Path
from nltk import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import wordnet

# указание языка датасета
stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()

# знаки, которые указывают на окончание предложения
end_of_clause = ['.', '?', '!', '...']

# функция принимает на вход слово и возвращает тег части речи
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

# сокращения, которые были найдены при просмотре датасета и замена их на полной слово
knownAbbrevs = {
    "st.": "saint",
    "dr.": "doctor",
    "tel.": "telephone",
    "no.": "number",
    "u.s.": "United States",
    "u.k.": "United Kingdom",
    "prof.": "professor",
    "inc.": "incorporated",
    "ltd.": "limited",
    "corp.": "corporation",
    "co.": "corporation",
    "mr.": "mister",
    "plc.": "Public Limited Company",
    "assn.": "association",
    "univ.": "university",
    "intl.": "international",
    "sys.": "system",
    "est.": "Eastern Standard Time",
    "ext.": "extention",
    "sq.": "square",
    "jr.": "junior",
    "sr.": "senior",
    "bros.": "brothers",
    "ed.d.": "Doctor of Education",
    "ph.d.": "Doctor of Phylosophy",
    "sci.": "Science",
    "etc.": "Et Cetera",
    "al.": "al",
    "seq.": "sequence",
    "orig.": "original",
    "incl.": "include",
    "eg.": "eg",
    "avg.": "average",
    "pl.": "place",
    "min.": "min",
    "max.": "max",
    "cit.": "citizen",
    "mrs.": "mrs",
    "mx.": "mx",
    "miss.": "miss",
    "atty.": "attorney",
    "col.": "college",
    "messrs.": "messieurs",
    "gov.": "government",
    "adm.": "admiral",
    "rev.": "revolution",
    "fr.": "french",
    "maj.": "major",
    "sgt.": "sergeant",
    "cpl.": "corporal",
    "pvt.": "private",
    "capt.": "captain",
    "ave.": "avenue",
    "pres.": "president",
    "brig.": "brigadier",
    "cmdr.": "commander",
    "asst.": "assistant",
    "assoc.": "associate",
    "insp.": "inspiration"
}

# регулярные выражения
tokens = [
    # аббревиатуры
    ["abbrev", "|".join(map(lambda kv: "(?i:" + re.escape(kv[0]) + ")", knownAbbrevs.items()))],
    # IP-адреса
    ["ipaddress", "[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+"],
    # Российский номер телефона
    # +7-901-000-00-00, 8-901-000-00-00, 8(901)000-00-00, 89010000000
    ["mobile_phone", "^(?:\+7|8)(?:(?:-\d{3}-|\(\d{3}\))\d{3}-\d{2}-\d{2}|\d{10})"],
    # адрес электронной почты, пример abc@abc.abc
    ["mail", "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"],
    # числительные (например, the 4th)
    ["numeral", "[0-9]+((th)|(\\'s))"],
    # числа
    ["number", "[0-9]+(.|,|-)[0-9]*"],
    # пробелы, новая строка, слеш, табуляция
    ["whitespace", "\\s|\\n|\\\\|\\t"],
    # скобки
    ["braces", "\\(|\\)"],
    # строки в кавычках
    ["quoted", "(\\\")[^\\\"]*(\\\")"],
    # пункктуация
    ["punct", ",|\\.|\\?|\\!|(\\.\\.\\.)"],
    # слова (обычные, слова с апострофом, с дефисом) - word, don't,
    ["word", "[A-Za-z][A-Za-z\\']*(-[A-Z\\']?[A-Za-z\\']+)*"],
    # остальное (неалфавитные, нецифровые символы, например, хештеги)
    ["other", ".[^a-zA-Z0-9]*"]
]

# сопоставление слова и примененного токена
regex = re.compile("^(" + "|".join(map(lambda t: "(?P<" + t[0] + ">" + t[1] + ")", tokens)) + ")")

# задание классов новости
classes = dict([
    (1, 'World'),
    (2, 'Sports'),
    (3, 'Business'),
    (4, 'Sci-Tech')
])

# непосредственно токенизация
def tokenize_text(text):
    # отслеживание текущей позиции в тексте
    pos = 0
    # копирование входного текста в переменную s, которая будет изменяться во время токенизации
    s = text
    # инициализация пустого списка line, в котором будут храниться найденные токены
    line = []
    while len(s) > 0:
        # поиск совпадений слова с регулярным выражением
        match = regex.search(s)
        if match and match.endpos > match.pos:
            for gr in tokens:
                tt = list(filter(lambda kv: kv[1] is not None, match.groupdict().items()))
                # получили совпадение с регулярным выражением
                if len(tt) == 1:
                    # получение типа токена
                    kind = tt[0][0]
                    # получение фрагмента текста
                    part = tt[0][1]
                    # замена слова, если это аббревиатура
                    if kind == 'abbrev':
                        kind = 'word'
                        part = knownAbbrevs[part.lower()]
                    line.append([pos, kind, part])
                    # обновление pos на основе длины найденного фрагмента
                    pos += len(tt[0][1])
                    s = s[len(tt[0][1]):]
                    break
                else:
                    print('failed to tokenize: ' + s)
        else:
            print('failed to tokenize: ' + s)
    return line

# запись токенизации в файл tsv
def process_file(fname):
    print('working on ', fname)
    # считываем содержимое файла
    df = pd.read_csv(fname, sep=',', header=None)
    # массив данных из DataFrame
    data = df.values
    # число строк в массиве
    data_count = len(data)
    n = 0
    for row in data:
        # значение первого элемента текущей строки (класс новости)
        class_id = row[0]
        try:
            # путь к директории, куда будут записываться данные
            dir_path = "../assets/" + Path(fname).name.split('.')[0] + "/" + classes[class_id] + '/'
            # если директории нет, то она создается
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            # если файл есть и он не пустой, всё в нём обнуляется
            f = open(dir_path + str(n) + '.tsv', 'w+')
            f.truncate(0)
            # непосредственно запись токенизации
            for i in range(1, len(row)):
                text = row[i]
                tokens = tokenize_text(text)
                prev = [0, '', '']
                for w in tokens:
                    # идет "разделение" пробелами (определение кол-ва слов в одном токене)
                    if w[1] != 'whitespace':
                        f.write(w[1] + '\t' + w[2] + '\t' + stemmer.stem(w[2]) + "\t" + lemmatizer.lemmatize(w[2], get_wordnet_pos(w[2])) + '\n')
                    # конец строки в датасете
                    elif prev[2] in end_of_clause:
                        f.write('\n')
                    prev = w
                f.write('\n')
            f.close()
        # вывод на консоль, если что-то пойдет не так
        except Exception as e:
            print(e)
            print([n, text, tokens])
            pass
        n = n + 1
        # выводит процент готовности
        if n % 1000 == 0:
            print(int(n * 100 / data_count), '%')

def main():
    fname_train = '../assets/raw-dataset/train.csv'
    fname_test = '../assets/raw-dataset/test.csv'
    process_file(fname_train)
    process_file(fname_test)


if __name__ == "__main__":
    main()