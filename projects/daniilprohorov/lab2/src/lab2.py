from typing import Dict, Any

from lib.typodistance import euclideanKeyboardDistance

dst = euclideanKeyboardDistance(' ', 's')


class Word:

    def __init__(self, str):
        self.charsCount = {}
        for c in str:
            if c in self.charsCount:
                self.charsCount[c] += 1
            else:
                self.charsCount[c] = 1

    def __str__(self):
        return str(self.charsCount)

    def __sub__(self, other):
        charsCountRes = {}
        for k in self.charsCount.keys():
            charsCountRes[k] = self.charsCount[k]
        for k in other.charsCount.keys():
            if k in charsCountRes:
                charsCountRes[k] -= other.charsCount[k]
                charsCountRes[k] = abs(charsCountRes[k])
                if charsCountRes[k] == 0:
                    charsCountRes.pop(k)
            else:
                charsCountRes[k] = other.charsCount[k]
        return charsCountRes



def levenstein(str_1, str_2):
    n, m = len(str_1), len(str_2)
    if n > m:
        str_1, str_2 = str_2, str_1
        n, m = m, n

    current_row = range(n + 1)
    for i in range(1, m + 1):
        previous_row, current_row = current_row, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete, change = previous_row[j] + 1, current_row[j - 1] + 1, previous_row[j - 1]
            if str_1[j - 1] != str_2[i - 1]:
                change += 1
            current_row[j] = min(add, delete, change)

    return current_row[n]


test = {'a': 1}

print(dst)
print(levenstein('lol', 'klo'))
fst = Word('lol')
snd = Word('lola')
print(fst - snd)
# сравнивать два слова, если есть разница, то брать эти два символа и смотреть какие из них наиболее близки по qwerty
