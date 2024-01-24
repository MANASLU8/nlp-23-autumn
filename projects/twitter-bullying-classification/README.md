# Cyberbullying classification

В рамках работы был выбран [датасет по классификации постов с кибербуллингом](https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification) в Твиттере (X).

Лабораторные работы выполнены в файлах:
1. [`Lab12.ipynb`](source/Lab12.ipynb) - лабораторные 1 и 2
2. [`Lab3.ipynb`](source/Lab3.ipynb) - лабораторная 3
3. [`Lab4_1.ipynb`](source/Lab4_1.ipynb) - лабораторная 4 вариант 1
4. [`Lab5.ipynb`](source/Lab5.ipynb) - лабораторная 5
5. [`Lab6.ipynb`](source/Lab6.ipynb) - лабораторная 6

Также представлены дополнительные файлы:
1. [`patterns.py`](source/patterns.py) - паттерны для поиска ссылок, почт и телефонов в тексте. Изначально представлены в файле `Lab12.ipynb`
2. [`utils.py`](source/utils.py) - содержит функцию для разбивки текста на предложения. Изначально представлена в файле `Lab12.ipynb`, используется в файлах `Lab5.ipynb` и `Lab6.ipynb`
3. [`chroma_db.py`](source/chroma_db.py) - класс для работы с векторной базой данных Chroma. Используется в файлах `Lab5.ipynb` и `Lab6.ipynb`