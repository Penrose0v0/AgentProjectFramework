import os
import json
import hashlib
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
from langchain_core.embeddings.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

TEXT_EXTENSIONS = (
    '.txt', '.md', '.py', '.json', '.js', '.ts', '.java',
    '.c', '.cpp', '.h', '.go', '.rb', '.rs',
    '.yaml', '.yml', '.toml', '.cfg', '.ini',
    '.csv', '.log', '.env', '.gitignore', 'Dockerfile'
)


def compute_file_hash(content: str, algo: str = "sha256") -> str:
    hasher = hashlib.new(algo)
    hasher.update(content.encode('utf-8'))
    return hasher.hexdigest()

def make_embedding(config: dict) -> Embeddings:
    provider = config.get("embedding_provider", "ollama")

    if provider == "openai":
        api_key = config.get("openai_api_key")
        if not api_key and config.get("api_key_path"):
            with open(config["api_key_path"], 'r', encoding='utf-8') as f:
                api_key = f.read().strip()
        
        return OpenAIEmbeddings(
            model=config.get("embedding_model", "text-embedding-3-small"),
            openai_api_key=api_key,
            openai_api_base=config.get("embedding_base_url")
        )
    
    elif provider == "huggingface":
        return HuggingFaceEmbeddings(
            model_name=config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
            model_kwargs={'device': config.get("device", "cpu")}
        )
    
    elif provider == "ollama":
        return OllamaEmbeddings(
            model=config.get("embedding_model", "nomic-embed-text"),
            base_url=config.get("embedding_base_url", "http://localhost:11434")
        )
    
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")
    

class DocumentLoader(ABC):
    @abstractmethod
    def load(self, source: str) -> List[Document]:
        pass


class TextFileLoader(DocumentLoader):
    def load(self, filepath: str) -> Document:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return Document(
            page_content=content,
            metadata={
                "source": filepath,
                "hash": compute_file_hash(content)
            }
        )


class JSONFileLoader(DocumentLoader):
    def load(self, filepath: str) -> Document:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            data = json.loads(content)
        doc_text = json.dumps(data, ensure_ascii=False, indent=2)
        return Document(
            page_content=doc_text,
            metadata={
                "source": filepath,
                "hash": compute_file_hash(doc_text)
            }
        )
    

class RAGComponent:
    def __init__(self, config: dict):
        self.config = config
        self.embedding = make_embedding(config)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.get("chunk_size", 1000),
            chunk_overlap=config.get("chunk_overlap", 200),
            separators=config.get("separators", ["\n\n", "\n", " ", ""])
        )
        self.rag_dir = config.get("rag_dir", "./rag_documents")
        
        # Init vector database
        self.vectorstore = None
        self.initialize_vectorstore()
        
        # Document loader
        self.loaders = {
            "text": TextFileLoader(),
            "json": JSONFileLoader(),
        }

    def initialize_vectorstore(self):
        vectorstore_path = self.config.get("vectorstore_path", "./vectorstore")
        
        self.vectorstore = Chroma(
            persist_directory=vectorstore_path,
            embedding_function=self.embedding
        )
        print(f"Initialized Chroma vectorstore at {vectorstore_path}")
        

    def sync_documents(self):
        current_files = set()
        for root, _, files in os.walk(self.rag_dir):
            for fname in files:
                full_path = os.path.join(root, fname)
                should_sync, source_type = self._sync_rule(full_path)
                if should_sync:
                    current_files.add(full_path)
                    self._index_document(full_path, source_type=source_type)
        self._remove_deleted_files(current_files)

    def list_documents(self, limit: int = 10):
        if self.vectorstore is None:
            print("Vectorstore is not initialized.")
            return []

        try:
            results = self.vectorstore._collection.get(include=['metadatas', 'documents'])

            documents = results.get("documents", [])
            metadatas = results.get("metadatas", [])

            output = []
            for i in range(min(limit, len(documents))):
                output.append({
                    "content": documents[i],
                    "metadata": metadatas[i]
                })

            print(f"列出了 {len(output)} 条向量数据库中的内容。")
            return output

        except Exception as e:
            print(f"Error retrieving documents from vectorstore: {e}")
            return []
        
    def _index_document(self, filepath: str, source_type: str = "text"):
        loader = self.loaders.get(source_type)
        if not loader:
            raise ValueError(f"Unsupported source type: {source_type}")

        if not os.path.isfile(filepath):
            print(f"Skipped non-file path: {filepath}")
            return

        try:
            document = loader.load(filepath)
            doc_hash = document.metadata["hash"]
            existing_hashes = self._get_existing_hashes()

            if filepath in existing_hashes:
                if existing_hashes[filepath] == doc_hash:
                    print(f"[Skip] File unchanged: {filepath}")
                    return True
                else:
                    print(f"[Update] File changed: {filepath}")
                    self._remove_document(filepath)
            else:
                print(f"[New] Indexing file: {filepath}")

            chunks = self.text_splitter.split_documents([document])
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_index": i,
                    "total_chunks_in_doc": len(chunks),
                    "chunk_size": len(chunk.page_content),
                    "word_count": len(chunk.page_content.split())
                })

            self.vectorstore.add_documents(chunks)
            if hasattr(self.vectorstore, 'persist'):
                self.vectorstore.persist()
            print(f"Indexed {len(chunks)} chunks for {filepath}")
            return True

        except Exception as e:
            print(f"Error indexing file {filepath}: {e}")
            return False
    
    def _sync_rule(self, filepath: str) -> tuple[bool, Optional[str]]:
        fname = os.path.basename(filepath)
        if fname in ('.gitignore', 'Dockerfile', '.env'):
            return True, "text"
        return any(fname.endswith(ext) for ext in TEXT_EXTENSIONS), "text"

    def _get_existing_hashes(self) -> Dict[str, str]:
        collection = self.vectorstore._collection.get(include=["metadatas"])
        hashes = {}
        for meta in collection.get("metadatas", []):
            src = meta.get("source")
            hash_val = meta.get("hash")
            if src and hash_val:
                hashes[src] = hash_val

        return hashes
    
    def _remove_document(self, source: str):
        try:
            collection = self.vectorstore._collection
            results = collection.get(include=["metadatas"])
            
            ids = results.get("ids", [])
            metadatas = results.get("metadatas", [])

            ids_to_delete = [
                doc_id for doc_id, meta in zip(ids, metadatas)
                if meta.get("source") == source
            ]

            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                print(f"Deleted {len(ids_to_delete)} chunks from: {source}")
        except Exception as e:
            print(f"Error deleting documents for {source}: {e}")

    def _remove_deleted_files(self, current_file_set: set):
        try:
            collection = self.vectorstore._collection
            results = collection.get(include=["metadatas"])

            ids = results.get("ids", [])
            metadatas = results.get("metadatas", [])

            ids_to_delete = []
            for doc_id, meta in zip(ids, metadatas):
                source = meta.get("source")
                if source and not os.path.exists(source) and source not in current_file_set:
                    ids_to_delete.append(doc_id)

            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                print(f"[Cleanup] Removed {len(ids_to_delete)} chunks from deleted files.")
            else:
                print("[Cleanup] No deleted files to remove.")
        except Exception as e:
            print(f"Error during cleanup of deleted files: {e}")
        

if __name__ == "__main__": 
    config = {
        "embedding_provider": "ollama",  # openai, huggingface, ollama
        "embedding_model": "nomic-embed-text",
        
        "vectorstore_path": "./vectorstore",

        "chunk_size": 1000,
        "chunk_overlap": 200,
    }
    rag = RAGComponent(config)
    rag.sync_documents()
    docs = rag.list_documents(limit=100)
    for i, d in enumerate(docs):
        print(f"\n--- Document {i+1} ---")
        print("Content:", d["content"])
        print("Metadata:", d["metadata"])
