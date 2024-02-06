# %%
!pip install langchain==0.0.300
!pip install chromadb==0.4.12

# %%
import chromadb
from langchain.docstore.document import Document
import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

# %%
from google.colab import drive
drive.mount('/content/drive')

# %% [markdown]
# # Task 1
# 
# Разбиение текстовых документов на фрагменты

# %%
SIZE = 250
OVERLAP = 50

# %%
# Класс загрузчика документов
# row - строка

class Loader:

    # Функция загрузки нескольких документов
    def load_documents(self, path: str):
        result = []
        dataframe = pd.read_csv(path, header=None, sep=",")

        for index, row in dataframe.iterrows():
          result.append(Document(page_content=row.iloc[2], metadata={'source': path, "row": index, "class": row.iloc[0], "topic": row.iloc[1]}))

        return result

# %%
loader = Loader()

docs = loader.load_documents(os.path.join(os.getcwd(), 'drive', 'MyDrive', 'ITMO', 'sem_3', 'NLP', 'assets', 'dataset', 'test.csv'))

print(docs[0])

# %% [markdown]
# ## 1.i + 1.ii

# %%
# Класс разбиения документов на фрагменты
class Splitter:
    def __init__(self, chunk_size, chunk_overlap):
        # Инициализация размера фрагментов
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = [' ', '.', '!', '?']

    # Функция разбиения
    def split_documents(self, documents):
        result = []
        for document in documents:
          page_content = document.page_content
          for sep in self.separators:
            # замена '.', '!', '?' на пробелы
            page_content = page_content.replace(sep, ' ')
          # сплит
          words = page_content.split()
          start = 0

          while start < len(words):
              end = start + self.chunk_size
              fragment = ' '.join(words[start:end])

              result.append(Document(page_content=fragment, metadata={'source': document.metadata['source'], "class": document.metadata['class'],
                                                                      "topic": document.metadata['topic']}))
              start = end - self.chunk_overlap # чтобы слово бралось целиком

        return result

# %%
splitter = Splitter(SIZE, OVERLAP)

splited_data = splitter.split_documents(docs)

# %%
splited_data[0]

# %% [markdown]
# # Task 2
# 
# Векторизация фрагментов текста

# %%
class Embedder:

  #Mean Pooling - Take attention mask into account for correct averaging
  @staticmethod
  # вычисление векторных представлений токенов с учетом маски внимания
  def mean_pooling(model_output, attention_mask):
      token_embeddings = model_output[0] #First element of model_output contains all token embeddings
      input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
      return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

  @staticmethod
  def embed_documents(documents):

    # загрузка модели с HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

    # токенизация предложений
    encoded_input = tokenizer([document.page_content for document in documents], padding=True, truncation=True, return_tensors='pt')

    # вычисление векторных представлений токенов
    with torch.no_grad():
        model_output = model(**encoded_input)

    # применение пулинга для получения векторных представлений предложений на основе векторных представлений токенов
    sentence_embeddings = Embedder.mean_pooling(model_output, encoded_input['attention_mask'])

    # вектора слов
    return sentence_embeddings.tolist()

# %%
embeddings = Embedder.embed_documents(splited_data[:100])
len(embeddings)

# %% [markdown]
# # Task 3
# 
# Создание Векторной Базы Данных

# %%
# Векторная БД
client = chromadb.Client()

collection_name = "myCol3"

collection = client.create_collection(collection_name)

# сколько данных берём
range_for = 1000

collection.add(
    # вектора
    embeddings=Embedder.embed_documents(splited_data[:range_for]),
    # порядковый номер строки в документе
    ids=[str(elem) for elem in range(len(splited_data[:range_for]))],
    # метадата
    metadatas=[doc.metadata for doc in splited_data[:range_for]],
    # page_content
    documents=[doc.page_content for doc in splited_data[:range_for]]
)

# %% [markdown]
# # Task 4
# 
# Поиск схожих фрагментов текста

# %%
class ChromaCollection():
    def __init__(self, collection_name, similarity, client):
      self.collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": similarity})

    def get(self):
      return self.collection

# %%
# Поиск фрагмента и вывод первых 10 подходящих
def search_fragment(real_id, query, similarity, collection_name, client):
  chc = ChromaCollection(collection_name, similarity, client)

  # поиск по нашей коллекции запроса
  documents_found = chc.get().query(Embedder.embed_documents([Document(page_content=query)]))

  # список 10 идентификаторов; реальный id, где есть этот фрагмент, либо 11; запросы
  return documents_found['ids'][0], documents_found['ids'][0].index(str(real_id)) + 1 if str(real_id) in documents_found['ids'][0] else 11, documents_found['documents']

# %%
search_fragment(1, 'rocketeers competing for the', 'cosine', collection_name, client)

# %% [markdown]
# # Task 5
# 
# Оценка качества поиска

# %%
# число - из какого документа взят фрагмент
sf1 = (0, "Turner Newall say they are")
sf2 = (1, 'rocketeers competing for the')
sf3 = (2, 'A company founded')
sf4 = (3, 'shift with a blur of')
sf5 = (-1, 'ha ha ha')

# %%
sr1_cos = search_fragment(sf1[0], sf1[1], 'cosine', collection_name, client)
sr2_cos = search_fragment(sf2[0], sf2[1], 'cosine', collection_name, client)
sr3_cos = search_fragment(sf3[0], sf3[1], 'cosine', collection_name, client)
sr4_cos = search_fragment(sf4[0], sf4[1], 'cosine', collection_name, client)
sr5_cos = search_fragment(sf5[0], sf5[1], 'cosine', collection_name, client)

sr1_l2 = search_fragment(sf1[0], sf1[1], 'l2', collection_name, client)
sr2_l2 = search_fragment(sf2[0], sf2[1], 'l2', collection_name, client)
sr3_l2 = search_fragment(sf3[0], sf3[1], 'l2', collection_name, client)
sr4_l2 = search_fragment(sf4[0], sf4[1], 'l2', collection_name, client)
sr5_l2 = search_fragment(sf5[0], sf5[1], 'l2', collection_name, client)

sr1_ip = search_fragment(sf1[0], sf1[1], 'ip', collection_name, client)
sr2_ip = search_fragment(sf2[0], sf2[1], 'ip', collection_name, client)
sr3_ip = search_fragment(sf3[0], sf3[1], 'ip', collection_name, client)
sr4_ip = search_fragment(sf4[0], sf4[1], 'ip', collection_name, client)
sr5_ip = search_fragment(sf5[0], sf5[1], 'ip', collection_name, client)



print(f"\nCредний порядковый номер требуемого фрагмента (cos): {np.mean([sr1_cos[1], sr2_cos[1], sr3_cos[1], sr4_cos[1], sr5_cos[1]])}")
print(f"\nCредний порядковый номер требуемого фрагмента (l2): {np.mean([sr1_l2[1], sr2_l2[1], sr3_l2[1], sr4_l2[1], sr5_l2[1]])}")
print(f"\nCредний порядковый номер требуемого фрагмента (ip): {np.mean([sr1_ip[1], sr2_ip[1], sr3_ip[1], sr4_ip[1], sr5_ip[1]])}")


