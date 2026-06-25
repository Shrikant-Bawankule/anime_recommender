import os

import certifi
import truststore
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings


from dotenv import load_dotenv

load_dotenv()

truststore.inject_into_ssl()
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())


class VectorStoreBuilder:
    def __init__(self, csv_path: str, persist_dir: str = "chroma_db"):
        self.csv_path = csv_path
        self.persist_dir = persist_dir
        self.embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def build_and_save_vectorstore(self):
        loader = CSVLoader(
            file_path=self.csv_path, encoding="utf-8", metadata_columns=[]
        )

        data = loader.load()

        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = splitter.split_documents(data)
        db = Chroma.from_documents(
            texts, self.embedding, persist_directory=self.persist_dir
        )
        db.persist()

    def load_vector_store(self):
        return Chroma(
            persist_directory=self.persist_dir, embedding_function=self.embedding
        )

    def build_and_save_vectorestore(self):
        return self.build_and_save_vectorstore()


Vector_store_builder = VectorStoreBuilder
