import chromadb
from sentence_transformers import SentenceTransformer


class EmbeddingFunction:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    def __call__(self, input):
        return self.model.encode(input).tolist()


class OldEmbeddingFunction:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

    def __call__(self, input):
        return self.model.encode(input).tolist()


class DB:
    def __init__(self, alg_name, root_path, ef):
        self.ef = ef()
        self.client = chromadb.PersistentClient(path=root_path)
        self.distance_function = "cosine"
        self.alg_name = alg_name
        self.collection = self.client.get_or_create_collection(self.alg_name,
                                                               metadata={"hnsw:space": self.distance_function},
                                                               embedding_function=self.ef)

    def add(self, items):
        old_batch = 0
        new_batch = 1000
        while True:
            if new_batch > len(items["fragments"]):
                break
            self.collection.add(
                documents=items["fragments"][old_batch:new_batch],
                ids=items["ids"][old_batch:new_batch])
            old_batch = new_batch
            new_batch += 1000
        self.collection.add(
            documents=items["fragments"][old_batch:],
            ids=items["ids"][old_batch:])

    def query(self, query, n_results):
        return self.collection.query(query_embeddings=self.ef(query), n_results=n_results)

    def clear(self):
        self.client.delete_collection("sem_split_" + self.alg_name)
        self.collection = self.client.get_or_create_collection("sem_split_" + self.alg_name,
                                                               metadata={"hnsw:space": self.distance_function},
                                                               embedding_function=self.ef)
