import re
from typing import List

from patterns import normal_email_pattern, normal_phone_pattern, normal_url_pattern

alphabets = "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"
multiple_dots = r'\.{2,}'
newlines = "\n+"


def split_into_sentences(text: str) -> List[str]:
    """
    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead
    to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """
    text = " " + text + "  "

    text = re.sub("^\s+", " ", text)

    email_iter = re.finditer(normal_email_pattern, text)
    for m in email_iter:
        text = text.replace(m.group(), m.group().replace(".", "<prd>"))

    phone_iter = re.finditer(normal_phone_pattern, text)
    for m in phone_iter:
        text = text.replace(m.group(), m.group().replace(".", "<prd>"))

    url_iter = re.finditer(normal_url_pattern, text)
    for m in url_iter:
        text = text.replace(m.group(), m.group().replace(".", "<prd>"))

    text = re.sub(r"(\,|\:|\;)*((\?|\!|\.)+)(\,|\:|\;)*", "\\2", text)
    text = re.sub(r"(\.+)[\?\!\.]*", "\\1<stop>", text)
    text = re.sub(r"((\?|\!)+)[\?\!\.]*", "\\1<stop>", text)
    text = re.sub(r"((\?+\!)|(\!+\?))[\?\!\.]*", "?!<stop>", text)

    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    text = re.sub(digits + "[.]" + digits, "\\1<prd>\\2", text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)

    text = re.sub(newlines, "<stop>", text)

    if "Ph.D" in text: text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
    text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
    text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
    if "”" in text: text = text.replace(".”", "”.")
    if "\"" in text: text = text.replace(".\"", "\".")
    if "!" in text: text = text.replace("!\"", "\"!")
    if "?" in text: text = text.replace("?\"", "\"?")
    text = text.replace(".", ".<stop>")
    text = text.replace("(?|!|?!)", "\\1<stop>")

    text = text.replace("<prd>", ".")

    text = re.sub("(\s*<stop>\s*){2,}", "<stop>", text)

    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]
    return sentences
