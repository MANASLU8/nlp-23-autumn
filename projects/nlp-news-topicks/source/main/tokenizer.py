import re
import os
import nltk
import pandas as pd
from pathlib import Path
from nltk import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import wordnet


# Set up values  
stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()
end_sign = ['.', '?', '!', '...', ';']

#Set and tokens patterns
tokens = [
    ["ipaddress", "[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+"],
    ["mobile_phone", "^(?:\+7|8)(?:(?:-\d{3}-|\(\d{3}\))\d{3}-\d{2}-\d{2}|\d{10})"],
    ["mobile_phone_usa", "^(?:\+1)(?:(?:-\d{3}-|\(\d{3}\))\d{3}-\d{4}|\d{7})"],
    ["mobile_phone_china", "^(?:\+86)(?:(?:-\d{3}-|\(\d{3}\))\d{4}-\d{4}|\d{11})"],
    ["mail", "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"],
    ["whitespace", "\\s|\\n|\\\\|\\t"],
    ["braces", "\\(|\\)"],
    ["quoted", "(\\\")[^\\\"]*(\\\")"],
    ["punct", ",|\\.|\\?|\\!|(\\.\\.\\.)"],
    ["word", "[A-Za-z][A-Za-z\\']*(-[A-Z\\']?[A-Za-z\\']+)*"],
    ["other", ".[^a-zA-Z0-9]*"]
]
regex = re.compile("^(" + "|".join(map(lambda t: "(?P<" + t[0] + ">" + t[1] + ")", tokens)) + ")")

#Get a part-of-speech tagging
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


#Tokenize text by sample
def tokenize_text(text):
    position = 0
    tmp_text = text
    token_list = []

    while len(tmp_text) > 0:
        match = regex.search(tmp_text)

        if match and match.endpos > match.pos:
            for gr in tokens:
                token_text = list(filter(lambda i: i[1] is not None, match.groupdict().items()))
                
                if len(token_text) == 1:
                    #Set up token type
                    type = token_text[0][0]
                    part = token_text[0][1]
                    token_list.append([position, type, part])
                    
                    #Shift position
                    position += len(token_text[0][1])
                    tmp_text = tmp_text[len(token_text[0][1]):]
                    break
                else:
                    print('Failed to tokenize: ' + tmp_text)
        else:
            print('Failed to tokenize: ' + tmp_text)
    return token_list


#Save tokens to existing file
def save_tokens(file_name):
    print('Save tokens to file: ', file_name)
    df = pd.read_csv(file_name, sep=',', header=None)
    data = df.values
    name, format= name.rpartition('.')
    count = 0 

    for row in data:
        class_id = row[0]
        try:
            # dir structure ./assets/{test or train}/{list of .tsv files}
            dir_path = f"D:/labs/nlp/assets/{name}/{class_id}/"

            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            f = open(dir_path + str(count) + '.tsv', 'w+')
            f.truncate(0)
            
            for i in range(1, len(row)):
                text = row[i]
                tokens = tokenize_text(text)
                prev = [0, '', '']
                for w in tokens:
                    if w[1] != 'whitespace':
                        f.write(w[1] + '\t' + w[2] + '\t' + stemmer.stem(w[2]) + "\t" + lemmatizer.lemmatize(w[2], get_wordnet_pos(w[2]))+'\n')
                    elif (prev[2] in end_sign) :
                        f.write('\n')
                    prev = w
                if i != len(row)-1:
                    f.write('\n')
            
            f.close()
        except Exception as e:
            print(e)
            pass
        count = count + 1