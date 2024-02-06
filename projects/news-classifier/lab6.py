# %%
!pip install langchain==0.0.300
!pip install chromadb==0.4.12
!pip install bert-score
!pip install ctransformers

# %%
from langchain.docstore.document import Document
from ctransformers import AutoModelForCausalLM
from bert_score import score
import pandas as pd
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import chromadb
import warnings

warnings.filterwarnings('ignore')

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
SIZE = 250
OVERLAP = 50

# %%
# Класс загрузчика документов

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
            page_content = page_content.replace(sep, ' ')
          words = page_content.split()
          start = 0

          while start < len(words):
              end = start + self.chunk_size
              fragment = ' '.join(words[start:end])

              result.append(Document(page_content=fragment, metadata={'source': document.metadata['source'], "class": document.metadata['class'],
                                                                      "topic": document.metadata['topic']}))
              start = end - self.chunk_overlap

        return result

# %%
splitter = Splitter(SIZE, OVERLAP)

splited_data = splitter.split_documents(docs)

# %%
class Embedder:

  #Mean Pooling - Take attention mask into account for correct averaging
  @staticmethod
  def mean_pooling(model_output, attention_mask):
      token_embeddings = model_output[0] #First element of model_output contains all token embeddings
      input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
      return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

  @staticmethod
  def embed_documents(documents):

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

    # Tokenize sentences
    encoded_input = tokenizer([document.page_content for document in documents], padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, average pooling
    sentence_embeddings = Embedder.mean_pooling(model_output, encoded_input['attention_mask'])

    return sentence_embeddings.tolist()

# %%
# Векторная БД
client = chromadb.Client()

collection_name = "myCol"

collection = client.create_collection(collection_name)

range_for = 1000

collection.add(
    embeddings=Embedder.embed_documents(splited_data[:range_for]),
    ids=[str(elem) for elem in range(len(splited_data[:range_for]))],
    metadatas=[doc.metadata for doc in splited_data[:range_for]],
    documents=[doc.page_content for doc in splited_data[:range_for]]
)

# %%
class ChromaCollection():
    def __init__(self, collection_name, similarity, client):
      self.collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": similarity})

    def get(self):
      return self.collection

# %%
# Поиск фрагмента и вывод первых 10 подходящих
def search_fragment(query, similarity, collection_name, client):
  chc = ChromaCollection(collection_name, similarity, client)

  documents_found = chc.get().query(Embedder.embed_documents([Document(page_content=query)]))

  return documents_found['ids'][0], documents_found['documents']

# %%
def answer_question(question, context_list, model):
    # Формирование контекста для модели
    context = '\n'.join(context_list)

    # Получение ответа от модели
    output = model(f"The context is {context}. Using this context answer the question: {question}")

    return output

# %%
# Инициализация модели
model = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-OpenOrca-GGUF", model_file="mistral-7b-openorca.Q3_K_M.gguf", model_type="mistral", gpu_layers=50)

# %%
# Вопросы
queries = [
  "What happened to money market funds in the past week?", # левый вопрос "Что произошло с фондами денежного рынка за последнюю неделю?"
  "What events might impact the stock market in the near future?", # Какие события могут повлиять на фондовый рынок в ближайшем будущем?
  "Why did Iraq halt oil exports from the main southern pipeline?", # Почему Ирак прекратил экспорт нефти по главному южному трубопроводу?
  "What is the butterfly?", # левый вопрос "Что такое бабочка?""
  "How did the dollar respond to the record U.S. trade deficit in June?", # Как доллар отреагировал на рекордный дефицит торгового баланса США в июне?
  "Does wall St. Bears refers to short-sellers, Wall Street's dwindling band of ultra-cynics, who are seeing gains again?", # ответ да/нет "Относится ли wall St. Bears к продавцам коротких позиций, редеющей группе ультрациников Уолл-стрит, которые снова демонстрируют рост?"
  "What impact do oil prices have on stock outlooks?", # "Какое влияние оказывают цены на нефть на биржевые прогнозы?"
  "What measures can non-OPEC countries consider to reduce record oil prices?", # Какие меры могут рассмотреть страны, не входящие в ОПЕК, для снижения рекордных цен на нефть?
  "How did the auction for Google Inc.'s initial public offering (IPO) start?", # Как начался аукцион по первичному публичному размещению акций Google Inc. (IPO)?
  "What recommendations provide regarding financial discussions?" # Какие рекомендации содержатся в отношении финансовых обсуждений?
]

# %%
# Предсказание ответов
predicted_answers = []
for query in queries:
  # Получаем топ ответов
  top_documents = search_fragment(query, 'cosine', collection_name, client)[1][0]
  # Передаем их как контекст
  predicted_answer = answer_question(query, top_documents, model)
  predicted_answers.append(predicted_answer)

predicted_answers

# %%
# Мои ответы
my_answers = [
  "The article does not provide information about recent events related to money market funds.", # В статье не приводится информация о недавних событиях, связанных с фондами денежного рынка.
  "Soaring crude oil prices, concerns about the economy, and outlook for earnings are expected to impact the stock market in the near future.", # Ожидается, что растущие цены на сырую нефть, опасения по поводу экономики и перспективы прибыли окажут влияние на фондовый рынок в ближайшем будущем.
  "Iraq halted oil exports due to intelligence suggesting a rebel militia could strike the infrastructure of the main southern pipeline.", # Ирак приостановил экспорт нефти из-за разведывательных данных, свидетельствующих о том, что повстанческое ополчение может нанести удар по инфраструктуре главного южного трубопровода.
  "A flying insect belonging to the order Lepidoptera, the origin of this term is associated with scales and wings.", # Летающее насекомое, относящееся к отряду чешуекрылых, происхождение этого термина связано с чешуей и крыльями.
  "The dollar tumbled broadly after data showing a record U.S. trade deficit in June, casting fresh doubts on the economy's recovery.", # Доллар резко упал после того, как данные показали рекордный дефицит торгового баланса США в июне, что вызвало новые сомнения в восстановлении экономики.
  "Yes", # Да
  "Soaring crude oil prices are expected to hang over the stock market, affecting its outlook.", # Ожидается, что растущие цены на сырую нефть будут оказывать давление на фондовый рынок, влияя на его перспективы.
  "Non-OPEC countries should consider increasing output to cool record crude prices, as suggested by OPEC President Purnomo Yusgiantoro.", # Странам, не входящим в ОПЕК, следует рассмотреть возможность увеличения добычи, чтобы снизить рекордные цены на сырую нефть, как предложил президент ОПЕК Пурномо Юсгианторо.
  "The auction for Google Inc.'s IPO got off to a rocky start after the company sidestepped a bullet from U.S. securities regulators.", # Аукцион по IPO Google Inc. начался неудачно после того, как компания избежала критики со стороны американских регуляторов по ценным бумагам.
  "The article advises not to be shy about discussing finances with elderly relatives, especially if there's a need for financial assistance." # В статье советуется не стесняться обсуждать финансы с пожилыми родственниками, особенно если есть необходимость в финансовой помощи.
]

# %%
# Сравниение ответов по bertscore
P, R, F1 = score(predicted_answers, my_answers, lang="en", verbose=True)
print(f"Precision: {P.mean():.4f}")
print(f"Recall: {R.mean():.4f}")
print(f"F1 score: {F1.mean():.4f}")


