import re

def tokenize(text):
    numbers = r'\d+[,|.]\d+'                                                    # числа разделенные точкой или запятой
    websites = r'|https?:\/\/\S+'                                               # сайты
    dates = r'|\d{1,3}[\.|\/]\d{1,4}[\.|\/]\d{1,4}'                             # даты
    abbreviations = r'|\bU.N|U.S|corp.|Corp.|inc.|Inc.\b'                       # некоторые абббривеатуры
    junk = r'|\/b&gt;|\/p&gt;|p&gt;|p&gt|&lt;|&gt;|br|\.\.\.|#36;|#39;s|#39;'   # нежелательные символы
    words_with_dash = r"|\b\w+-\w+?\b"                                          # слова с дефисом
    words = r"|\b\w+(?:'s)?\b"                                                  # слова на 's
    time = r'|\d{1,2}\:\d{2}'                                                   # время
    special_symbols = r'|[^\w\s]'                                               # спец символы - не буква/цифра/пробел
    words_left = r'|[A-Za-z]+'
    email = r'|[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+'                                 # почта
    hashtags = r'|\#\b\w\w+\b'                                                  # хэштеги
    mentions = r'|\@\b\w\w+\b'                                                  # упоминания
    emoji = r'|[\U0001F300-\U0001F5FF]'                                         # смайлики
    acronyms = r'|\b(?:[a-zA-Z]\.){2,}'                                         # акронимы
    tokens = re.findall(numbers+websites+dates+time+abbreviations+junk+words_with_dash+words+special_symbols+email+hashtags+mentions+emoji+acronyms+words_left, text)
    return tokens
