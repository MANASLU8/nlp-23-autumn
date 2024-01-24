import chromadb
from chromadb.utils import embedding_functions


class ChromaDB:
    class ChromaCollection():
        def __init__(self, collection_name, similarity, client, model_name=None):
            if model_name is None:
                model_name = "BAAI/bge-large-en-v1.5"
            self.collection_name = collection_name
            self.similarity = similarity
            self.client = client
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": similarity},
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
            )

        def add(self, embeddings, texts, metadatas, ids):
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )

        def query(self, n_results, query_texts=None, query_embeddings=None, where=None, where_document=None):
            assert (query_texts or query_embeddings) and not (query_texts and query_embeddings)
            return self.collection.query(
                query_texts=query_texts,
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                where_document=where_document
            )

    def __init__(self):
        self.client = chromadb.PersistentClient(path="./ChromaDB_client")

    def clear(self, name):
        self.client.delete_collection(name=name)
        return self.client.list_collections()

    def get_collection(self, name, similarity=None):
        return self.ChromaCollection(name, similarity, self.client)

    def get_collections(self):
        return self.client.list_collections()
