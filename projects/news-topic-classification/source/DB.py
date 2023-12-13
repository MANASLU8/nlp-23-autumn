from sentence_transformers import SentenceTransformer
import chromadb


class EmbeddingFunction:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

    def __call__(self, input):
        return self.model.encode(input).tolist()


class DB:

    def __init__(self, distance_function, root_path):
        self.ef = EmbeddingFunction()
        self.client = chromadb.PersistentClient(path=root_path)
        self.distance_function = distance_function
        assert distance_function in ["l2", "ip", "cosine"], "Distance function should be 'l2' or 'ip' or 'cosine'"
        self.collection = self.client.get_or_create_collection("lab5_" + self.distance_function,
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
                metadatas=items["metadata"][old_batch:new_batch],
                ids=items["ids"][old_batch:new_batch])
            old_batch = new_batch
            new_batch += 1000
        self.collection.add(
            documents=items["fragments"][old_batch:],
            metadatas=items["metadata"][old_batch:],
            ids=items["ids"][old_batch:])

    def query(self, query, n_results):
        return self.collection.query(query_embeddings=self.ef(query), n_results=n_results)

    def clear(self):
        self.client.delete_collection("lab5_" + self.distance_function)
        self.collection = self.client.get_or_create_collection("lab5_" + self.distance_function,
                                                               metadata={"hnsw:space": self.distance_function},
                                                               embedding_function=self.ef)
