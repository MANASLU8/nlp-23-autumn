import re
def tokenize(text):
    numbers = r'\d+[,|.]\d+'
    # seq_in_brackets = r'\((?:.*?)\)'
    websites = r'|https?:\/\/\S+'
    dates = r'|\d{1,3}[\.|\/]\d{1,4}[\.|\/]\d{1,4}'
    abbreviations = r'|\bU.N|U.S|corp.|Corp.|inc.|Inc.\b'
    junk = r'|\/b&gt;|\/p&gt;|p&gt;|p&gt|&lt;|&gt;|br|\.\.\.|#36;|#39;s|#39;'
    words_with_dash = r"|\b\w+-\w+?\b"
    words = r"|\b\w+(?:'s)?\b"  # with 's ending 
    time = r'|\d{1,2}\:\d{2}'
    special_symbols = r'|[^\w\s]'
    words_left = r'|[A-Za-z]+'
    tokens = re.findall(numbers+websites+dates+time+abbreviations+junk+words_with_dash+words+special_symbols+words_left, text)
    return tokens

#    abbreviations = r'|\bU.S|#36;|#39;s|#39;|corp.|Corp.|inc.|Inc.\b'
