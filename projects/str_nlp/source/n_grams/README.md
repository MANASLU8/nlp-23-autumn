## Лабораторная работа №2. N-grams
**Задачи:**
<li> Извлечь триграммы из текста на основании своего алгоритма и на основании скрипта, основанного на библиотеке `nltk` <br>
<li> Реализовать алгоритм расчета меры ассоциации лексемы (`MI` и `log-likelihood`)<br>

**Меры ассоциативной связанности (`association measures`)** - меры, вычисляющие силу связи между элементами в составе коллокации (параметры: частота совместной встречаемости, частота слова в корпусе, размер корпуса, и др.).  

**MI**
![MI](https://github.com/kivirciks/nlp/blob/main/assets/MI.png)
`n` — ключевое слово (`node`);  
`c` — коллокат (`collocate`);  
`f (n, c)` — частота встречаемости ключевого слова `n` в паре с коллокатом `c`;  
`f(n)`, `f(c)` — абсолютные (независимые) частоты ключевого слова `n` и слов `c` в корпусе (тексте) соответственно;  
`N` — общее число словоупотреблений в корпусе (тексте);  
`ngram` - количество слов в `n`-грамме (например, для триграмм `ngram = 3`);  
`f(u_i)` - абсолютная частота `i`-й униграммы в `n`-грамме.  

**Log-likelihood**
![log-likelihood](https://github.com/kivirciks/nlp/blob/main/assets/log-likelihood.png)
`O_ij`, `E_ij` - наблюдаемая и ожидаемая частоты соответственно;  
`ngram` - количество слов в `n`-грамме (например, для триграмм `ngram = 3`).  

**N-grams без использования `nltk`**
Пример кода для извлечение триграмм из текста на основании лексем

```python
def extract_trigrams(text):
    trigrams = []
    tokens = nltk.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    for i in range(len(lemmas) - 2):
        trigram = ' '.join([lemmas[i], lemmas[i+1], lemmas[i+2]])
        trigrams.append(trigram)
    return trigrams
```

Пример кода для подсчета MI

```python
def calculate_mutual_information(trigram_counts, trigram, unigram_counts):
    word1, word2, word3 = trigram.split()
    unigram1_count = unigram_counts[word1]
    unigram2_count = unigram_counts[word2]
    unigram3_count = unigram_counts[word3]
    trigram_count = trigram_counts[trigram]
    total_count = sum(trigram_counts.values())
    mutual_information = math.log2((trigram_count*total_count) / (unigram1_count*unigram2_count*unigram3_count))
    return mutual_information
```
Пример кода для подсчета log likelihood

```python
def calculate_log_likelihood(trigram_counts, trigram, unigram_counts):
    word1, word2, word3 = trigram.split()
    unigram1_count = unigram_counts[word1]
    unigram2_count = unigram_counts[word2]
    unigram3_count = unigram_counts[word3]
    trigram_count = trigram_counts[trigram]
    total_count = sum(trigram_counts.values())
    expected_count = (unigram1_count*unigram2_count*unigram3_count) / total_count
    log_likelihood = 2 * (trigram_count * math.log2(trigram_count / expected_count))
    return log_likelihood
```

ТОП-30 триграмм 

| Триграмма | MI | Log-likelihood |
|----------|----------|----------|
| new york reuters (2265)  | -7.310437798121416   | -33116.28322549001   |
| quote profile research (1092)   | -0.23204100962071586   | -506.77756501164345   |
| york reuters u (561)  | -0.8479176193695295   | -951.3635689326121   |
| canadian presscanadian press (479)   | 1.0848918260589877   | 1039.3263693645101   |
| boston red sox (478)   | -1.9462391857329742   | -1860.6046615607233   |
| world 39s largest (474)   | -8.567686594885021   | -8122.166891951   |
| faceverdanams sans serifarialhelvetica (439)   | 3.974822766798887   | 3489.894389249423   |
| sans serifarialhelvetica size (439)   | 3.974822766798887   | 3489.894389249423   |
| george w bush (429)   | 0.34802133270720864   | 298.6023034627852   |
| new york yankee (423)    | -3.5918229643470037   | -3038.6822278375653   |
| security exchange commission (357)  | -2.331585381951799   | -1664.7519627135848   |
| prime minister ariel (357)    | -1.885251327799964   | -1346.0694480491743   |
| reuters u stock (340)    | 0.18836739744741643   | 128.08983026424298   |
| minister ariel sharon (334)    | 0.29044934524576943   | 194.020162624174   |
| minister tony blair (312)    | -0.4148057630631898   | -258.8387961514304   |
| president george w (302)    | -1.23524674663915   | -746.0890349700464   |
| prime minister tony (276)    | -2.624157131799956   | -1448.5347367535758   |
| president vladimir putin (273)    | -0.1129479307844089   | -61.66957020828726   |
| san francisco reuters (267)    | -4.75417754643795   | -2538.7308097978653   |
| international space station (264)    | -2.9771165986244474   | -1571.9175640737083   |
| ltfont faceverdanams sans (258)    | 3.8671089494986335   | 1995.4282179412949   |
| reuters oil price (247)    | -6.984263667100394   | -3450.226251547594   |
| un security council (247)    | -3.354618101834084   | -1657.1813423060375   |
| initial public offering (236)    | 0.18756600726582623   | 88.53115542946998   |
| major league baseball (222)    | -3.2985355962675946   | -1464.549804742812   |
| world 39s biggest (220)    | -9.191657165195993   | -4044.3291526862367   |
| state colin powell (219)    | 0.6468358646207671   | 283.314108703896   |
| serifarialhelvetica size color (214)    | 4.057832543258923   | 1736.7523285148188   |
| size color washington (212)    | 1.714492226676719   | 726.9447041109289   |
| color washington postltbgtltfontgt (212)    | 2.6345883610845053   | 1117.0654650998301  |


**N-grams с использованием `nltk`**
Пример кода для извлечение триграмм из текста и подсчет метрик на основании лексем

```python
trigram_measures = nltk.collocations.TrigramAssocMeasures()
tokens = nltk.word_tokenize(raw)
text = nltk.Text(tokens)
finder_thr = TrigramCollocationFinder.from_words(text)
trigram_scores = finder_thr.score_ngrams(trigram_measures.pmi)
```

ТОП-30 триграмм 

| Триграмма | MI | Column 3 |
|----------|----------|----------|
| abend einem offiziellen    | 5.4297629888288625   | 
| bainum outspent mufi    | 5.4297629888288625   | 
| circusif roone arledge    | 5.4297629888288625   |
| claptons laylarecently guitarworld    | 5.4297629888288625   |
| desa rueng bakjok    | 5.4297629888288625   | 
| dropper downloader bagleaq    | 5.4297629888288625   | 
| easybus easypizza easycar    | 5.4297629888288625   |
| electionbefore shugofa beheshti    | 5.4297629888288625   | 
| fadhil muhsen salom    | 5.4297629888288625   | 
| gimm mur schwg    | 5.4297629888288625   | 
| hanno rilasciato congiuntamente    | 5.4297629888288625   | 
| jasbir kang yuba    | 5.4297629888288625   | 
| lifesized jabba hutt    | 5.4297629888288625   | 
| madame edmey cimeus    | 5.4297629888288625   | 
| namesas bassist popfunk    | 5.4297629888288625   | 
| newburg bahar uttam    | 5.4297629888288625   | 
| olympicsby miron varouhakis    | 5.4297629888288625   | 
| parmi lesquelles figurent    | 5.4297629888288625   | 
| quebeckers guylaine dumont    | 5.4297629888288625   | 
| raad altamimi newlylaunched    | 5.4297629888288625   |
| staphylococcus aureus mrsa    | 5.4297629888288625   | 
| technogadgets holidaystalk overachiever    | 5.4297629888288625   | 
| undoing flintoffbrian laras    | 5.4297629888288625   |
| utils bsd licenseand    | 5.4297629888288625   | 
| waisted emeraldeyed brunette    | 5.4297629888288625   | 
| xwiki aspwiki snipsnap    | 5.4297629888288625   | 
| yearjune kamila vodichkova    | 5.4297629888288625   | 
| zhah stoyah kohvich    | 5.4297629888288625   | 
| zihdroo nuhs ihlgows    | 5.4297629888288625   | 
| zservices clientsofts servicebuilder    | 5.4297629888288625   | 
