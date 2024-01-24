import chromadb
from chromadb.utils import embedding_functions


class ChromaDataBase:
    def __init__(self, 
                 root_path='../ChromaDataBase', 
                 similarity_function='cosine', 
                 model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.client = chromadb.PersistentClient(path=root_path)
        self.collection_name = f'db_{similarity_function}'
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={'hnsw:space': similarity_function},
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
        )

    def add(self, embeddings, texts, metadatas, ids, batch_size=4000):
        assert len(embeddings) == len(texts) == len(metadatas) == len(ids)
        total = len(embeddings)
        for start_idx in range(0, total, batch_size):
            end_idx = min(start_idx + batch_size, total)
            batch_embeddings = embeddings[start_idx:end_idx]
            batch_texts = texts[start_idx:end_idx]
            batch_metadatas = metadatas[start_idx:end_idx]
            batch_ids = ids[start_idx:end_idx]

            self.collection.add(
                embeddings=batch_embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )

    def query(self, n_results, query_texts=None, query_embeddings=None, where=None, where_document=None):
        return self.collection.query(
            n_results=n_results,
            query_texts=query_texts,
            query_embeddings=query_embeddings,
            where=where,
            where_document=where_document
        )

    def clear(self):
        self.client.delete_collection(name=self.collection_name)
        return self.client.list_collections()
