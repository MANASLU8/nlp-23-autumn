## Лабораторная работа №1. Токенизация
<li> Токенизация - разбиение текста на осмысленные элементы (слова, фразы, символы), называемые токенами. <br>
<li> Стемминг - это процесс нахождения основы слова для заданного исходного слова. Основа слова не обязательно совпадает с морфологическим корнем слова.<br>
<li> Лемматизация - процесс приведения словоформы к лемме — её нормальной (словарной) форме.<br>
  
**Задачи:**
<li> Реализовать сегментацию текста на предложения и их токенизацию на основе регулярных выражений
<li> Выполнить стемминг (SnowballStemmer) и лемматизацию (WordNetLemmatizer) текста
<li> Сформировать аннотаию в формате tsv
<br>
  <br>
  
  **Пример собственного токенизатора:**
```python
def tokenize_text(text):
    pos = 0
    s = text
    line = []
    while len(s) > 0:
        match = regex.search(s)
        if match and match.endpos > match.pos:
            for gr in tokens:
                tt = list(filter(lambda kv: kv[1] is not None, match.groupdict().items()))
                if len(tt) == 1:
                    kind = tt[0][0]
                    part = tt[0][1]
                    if kind == 'abbrev':
                        kind = 'word'
                        part = knownAbbrevs[part.lower()]
                    line.append([pos, kind, part])
                    pos += len(tt[0][1])
                    s = s[len(tt[0][1]):]
                    break
                else:
                    print('failed to tokenize: ' + s)
        else:
            print('failed to tokenize: ' + s)
    return line  
```
  **Пример результата токенизации:**
<li> Первый столбик - исходное слово
<li> Второй столбик - стемминг 
<li> Третий столбик - лемматизация
    <br>

  ![Результат токенизации](https://github.com/kivirciks/nlp-23-autumn/blob/main/projects/news_nlp/utils/example_tokenizer.PNG)

