import os
import shutil
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
import pandas as pd

from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss



@dataclass
class RAGConfig:
    EMBEDDING_MODEL: str = "nomic-embed-text:latest"
    EMBEDDING_DIM: int = 768
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_PROVIDER: str = "anthropic"
    LLM_MODEL: str = "claude-sonnet-4-5-20250929"
    TEMPERATURE: float = 0.3
    STORAGE_DIR: str = "./storage"
    FAISS_INDEX_FILE: str = "./storage/faiss_index.faiss"
    DOCSTORE_FILE: str = "./storage/docstore.json"
    ANTHROPIC_API_KEY: str = "x"

def clean_storage(storage_dir: str):
    p = Path(storage_dir)
    if p.exists():
        shutil.rmtree(p)
        print(f"Storage temizlendi: {storage_dir}")

def load_and_prepare_documents_v2(base_path: str = "veri/42bin_haber/news") -> List[Document]:
    rows = []
    error_count = 0
    base_path_obj = Path(base_path)
    if not base_path_obj.exists():
        print(f" Klasör bulunamadı: {base_path}")
        return []

    for category_path in base_path_obj.iterdir():
        if not category_path.is_dir():
            continue
        category = category_path.name
        print(f" Kategori işleniyor: {category}")

        for file_path in category_path.glob("*.txt"):
            try:
                text = file_path.read_text(encoding="utf-8").strip()
                if len(text) > 30:
                    rows.append({"kategori": category, "metin": text, "dosya": file_path.name})
            except Exception as e:
                print(f"⚠ Dosya okunamadı: {file_path} - {e}")
                error_count += 1

    if not rows:
        print(f"✗ Haber bulunamadı. Yol doğru mu? {base_path}")
        return []

    df = pd.DataFrame(rows)
    documents = [Document(text=row['metin'], metadata={"kategori": row['kategori'], "dosya": row['dosya']}) for _, row in df.iterrows()]
    print(f"\n✓ {len(documents)} belge başarıyla Document formatına dönüştürüldü\n")
    return documents

def setup_global_embedding(settings: Settings, cfg: RAGConfig):
    Settings.embed_model = OllamaEmbedding(model_name=cfg.EMBEDDING_MODEL, base_url=cfg.OLLAMA_BASE_URL, ollama_additional_kwargs={"mirostat":0})
    print(" Global Ollama embedding ayarlandı")

def create_faiss_index(documents: List[Document], cfg: RAGConfig) -> Optional[VectorStoreIndex]:
    try:
        faiss_index = faiss.IndexFlatIP(cfg.EMBEDDING_DIM)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)
        Path(cfg.STORAGE_DIR).mkdir(parents=True, exist_ok=True)
        faiss.write_index(faiss_index, cfg.FAISS_INDEX_FILE)
        print(f" FAISS index dosyası kaydedildi: {cfg.FAISS_INDEX_FILE}")
        return index
    except Exception as e:
        print(f" İndeks oluşturma hatası: {e}")
        return None

def load_or_create_index(documents: List[Document], cfg: RAGConfig) -> Optional[VectorStoreIndex]:
    storage_dir = Path(cfg.STORAGE_DIR)
    setup_global_embedding(Settings, cfg)

    if storage_dir.exists() and (storage_dir / "docstore.json").exists():
        try:
            vector_store = FaissVectorStore.from_persist_dir(str(storage_dir))
            storage_context = StorageContext.from_defaults(persist_dir=str(storage_dir), vector_store=vector_store)
            index = load_index_from_storage(storage_context=storage_context)
            print(" İndeks başarıyla yüklendi")
            return index
        except Exception as e:
            print(f" İndeks yüklenemedi: {e} — storage temizleniyor")
            clean_storage(str(storage_dir))

    return create_faiss_index(documents, cfg)

def call_llm(prompt: str, cfg: RAGConfig, max_tokens: int = 512) -> str:
    from anthropic import Anthropic
    client = Anthropic(api_key=cfg.ANTHROPIC_API_KEY)

    try:
        resp = client.messages.create(
            model=cfg.LLM_MODEL,
            max_tokens=max_tokens,
            temperature=cfg.TEMPERATURE,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return resp.content[0].text
    except Exception as e:
        print(f"✗ LLM çağrı hatası: {e}")
        return "[LLM cevap hatası]"

class AgenticRAG:
    def __init__(self, index: VectorStoreIndex, cfg: RAGConfig):
        self.index = index
        self.cfg = cfg
        self.retriever = index.as_retriever(similarity_top_k=5)

    def retrieve_context(self, question: str, k: int = 5) -> List[Document]:
        return self.retriever.retrieve(question)

    def plan(self, question: str, contexts: List[Document]) -> str:
        snippets = "\n---\n".join([c.get_content()[:600] for c in contexts])
        prompt = f"Aşağıdaki bağlam parçaları verildi. Kısa plan oluştur.\nBAĞLAM:\n{snippets}\nSORU: {question}\nPLAN:"
        return call_llm(prompt, self.cfg, max_tokens=150)

    def generate_answer(self, question: str, contexts: List[Document], plan: str) -> str:
        context_text = "\n---\n".join([c.get_content() for c in contexts])
        prompt = f"Aşağıdaki bağlam ve planı kullanarak cevap ver.\nBAĞLAM:\n{context_text}\nPLAN:\n{plan}\nSORU: {question}\nCEVAP:"
        return call_llm(prompt, self.cfg, max_tokens=512)

    def run(self, question: str) -> str:
        contexts = self.retrieve_context(question)
        if not contexts:
            return "Maalesef uygun bir bilgi bulunamadı."
        plan = self.plan(question, contexts)
        return self.generate_answer(question, contexts, plan)

def main():
    cfg = RAGConfig()
    print("🚀 Agentic RAG v2 başlatılıyor")

    documents = load_and_prepare_documents_v2(base_path="veri/42bin_haber/news")
    if not documents:
        print("✗ Belge yüklenemedi. Çalışma dizinini kontrol et.")
        return

    index = load_or_create_index(documents, cfg)
    if not index:
        print("✗ Index oluşturulamadı")
        return

    agent = AgenticRAG(index, cfg)

    test_queries = [
        "Son futbol transfer haberleri hakkında bilgi ver",
        "Türkiye ekonomisinde son gelişmeler neler?",
        "Yapay zeka alanındaki önemli konferanslar hangileri?"
    ]

    for q in test_queries:
        print("\n---\nSoru:", q)
        answer = agent.run(q)
        print("Cevap:\n", answer[:1500], "...\n")

if __name__ == "__main__":
    main()
