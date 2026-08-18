"""Vector store manager and RAG query pipeline using FastEmbed, ChromaDB, and Groq."""

import logging
import os
from typing import Any

import chromadb
from fastembed import TextEmbedding
from groq import Groq

from app.config import Settings, get_settings
from app.core.loader import DocumentChunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Você é um especialista em arquitetura de backend e documentação técnica corporativa.
Sua missão é responder perguntas dos desenvolvedores com base ESTRITAMENTE no contexto documental fornecido abaixo.

Diretrizes obrigatórias:
1. Baseie-se apenas nas informações presentes no contexto fornecido.
2. Seja direto, técnico e preciso em suas explicações.
3. Quando houver trechos de código, comandos, endpoints HTTP ou diagramas, inclua-os formatados em Markdown limpo.
4. Cite explicitamente as fontes e páginas consultadas (ex: `[Fonte: arquitetura.pdf, Página 2]`) no corpo da resposta ou ao final.
5. Se o contexto não contiver informações suficientes para responder com certeza à pergunta, declare educadamente que a documentação técnica disponível não cobre o tópico solicitado e evite alucinações.

---
Contexto Documental Relevante:
{context}
---
"""


class RAGPipeline:
    """Orchestrates document embedding, ChromaDB vector indexing, semantic search and Groq LLM generation."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

        # Initialize FastEmbed Embedding Model
        logger.info(f"Initializing FastEmbed model: {self.settings.EMBEDDING_MODEL}")
        self.embedding_model = TextEmbedding(model_name=self.settings.EMBEDDING_MODEL)

        # Initialize ChromaDB persistent client
        try:
            os.makedirs(self.settings.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
        except OSError as e:
            logger.warning(
                f"Não foi possível criar o diretório '{self.settings.CHROMA_PERSIST_DIRECTORY}' ({e}). "
                "Tentando conectar ao ChromaDB diretamente..."
            )

        self.chroma_client = chromadb.PersistentClient(
            path=self.settings.CHROMA_PERSIST_DIRECTORY
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.settings.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        # Initialize Groq Client
        self._groq_client: Groq | None = None
        if self.settings.GROQ_API_KEY:
            self._groq_client = Groq(api_key=self.settings.GROQ_API_KEY)
        else:
            logger.warning("GROQ_API_KEY not set. LLM generation endpoints will require API key.")

    @property
    def groq_client(self) -> Groq:
        """Get or initialize Groq client dynamically."""
        if not self._groq_client:
            api_key = os.getenv("GROQ_API_KEY") or self.settings.GROQ_API_KEY
            if not api_key:
                raise ValueError(
                    "GROQ_API_KEY não foi configurada. Configure a variável no arquivo .env ou no ambiente."
                )
            self._groq_client = Groq(api_key=api_key)
        return self._groq_client

    def _generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate normalized vector embeddings for a list of text strings using FastEmbed."""
        embeddings_generator = self.embedding_model.embed(texts)
        return [embedding.tolist() for embedding in embeddings_generator]

    def index_chunks(self, chunks: list[DocumentChunk], batch_size: int = 64) -> dict[str, Any]:
        """Generate embeddings and upsert chunks into ChromaDB collection in batches."""
        if not chunks:
            return {"status": "empty", "indexed_count": 0, "total_collection_count": self.collection.count()}

        total_chunks = len(chunks)
        logger.info(f"Indexing {total_chunks} chunks into ChromaDB collection '{self.settings.COLLECTION_NAME}'")

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]
            ids = [chunk.id for chunk in batch]
            documents = [chunk.content for chunk in batch]
            metadatas = [chunk.metadata for chunk in batch]

            embeddings = self._generate_embeddings(documents)

            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )

        collection_count = self.collection.count()
        logger.info(f"Successfully indexed {total_chunks} chunks. Collection total: {collection_count}")
        return {
            "status": "success",
            "indexed_count": total_chunks,
            "total_collection_count": collection_count,
        }

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Retrieve most semantically similar document chunks for a given query."""
        k = top_k or self.settings.TOP_K
        total_count = self.collection.count()
        if total_count == 0:
            logger.warning("Vector collection is empty. Cannot retrieve documents.")
            return []

        actual_k = min(k, total_count)
        query_embedding = self._generate_embeddings([query])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved_docs: list[dict[str, Any]] = []
        if results and results["documents"] and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
            distances = results["distances"][0] if results["distances"] else [0.0] * len(docs)
            ids = results["ids"][0] if results["ids"] else [""] * len(docs)

            for doc_id, doc_text, meta, dist in zip(ids, docs, metas, distances):
                # Cosine distance in Chroma: similarity = 1 - distance
                similarity_score = round(1.0 - float(dist), 4)
                retrieved_docs.append(
                    {
                        "id": doc_id,
                        "content": doc_text,
                        "metadata": meta,
                        "similarity_score": similarity_score,
                        "source": meta.get("source", "unknown"),
                    }
                )

        return retrieved_docs

    def _build_context_string(self, retrieved_docs: list[dict[str, Any]]) -> str:
        """Format retrieved documents into structured context text for prompt insertion."""
        if not retrieved_docs:
            return "Nenhum documento relevante encontrado na base de conhecimento."

        context_blocks = []
        for i, doc in enumerate(retrieved_docs, start=1):
            source = doc.get("source", "Documento")
            score = doc.get("similarity_score", 0.0)
            meta = doc.get("metadata", {})
            page = meta.get("page")
            page_info = f" | Pág. {page}" if page else ""
            content = doc.get("content", "")
            context_blocks.append(
                f"[Documento {i} | Fonte: {source}{page_info} | Relevância: {score:.2f}]\n{content}"
            )

        return "\n\n".join(context_blocks)

    def answer_query(
        self,
        query: str,
        chat_history: list[dict[str, str]] | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Execute the full RAG pipeline: retrieval, context construction, and Groq LLM completion."""
        # 1. Retrieve relevant chunks
        retrieved_docs = self.retrieve(query=query, top_k=top_k)

        # 2. Build context
        context_str = self._build_context_string(retrieved_docs)

        # 3. Construct prompt messages
        system_content = SYSTEM_PROMPT.format(context=context_str)
        messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]

        # Include prior conversation history if provided (limiting to last 6 turns)
        if chat_history:
            for turn in chat_history[-6:]:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": query})

        # 4. Generate answer via Groq
        try:
            client = self.groq_client
            response = client.chat.completions.create(
                model=self.settings.GROQ_MODEL,
                messages=messages,
                temperature=self.settings.GROQ_TEMPERATURE,
                max_tokens=self.settings.GROQ_MAX_TOKENS,
            )
            answer_text = response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Error calling Groq API: {e}")
            answer_text = (
                f"⚠️ Ocorreu um erro ao comunicar com o modelo LLM (Groq): {str(e)}. "
                "Verifique se sua `GROQ_API_KEY` está configurada corretamente no arquivo `.env`."
            )

        # 5. Extract sources metadata
        sources = []
        seen_sources = set()
        for doc in retrieved_docs:
            src_name = doc.get("source", "desconhecido")
            meta = doc.get("metadata", {})
            page = meta.get("page")
            source_key = f"{src_name}_p{page}" if page else src_name
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                sources.append(
                    {
                        "source": src_name,
                        "page": page,
                        "similarity_score": doc.get("similarity_score", 0.0),
                        "snippet": doc.get("content", "")[:180] + "...",
                        "metadata": meta,
                    }
                )

        return {
            "query": query,
            "answer": answer_text,
            "model": self.settings.GROQ_MODEL,
            "sources": sources,
            "chunks_retrieved": len(retrieved_docs),
        }

    def get_stats(self) -> dict[str, Any]:
        """Return operational statistics about the vector collection."""
        count = self.collection.count()
        return {
            "collection_name": self.settings.COLLECTION_NAME,
            "persist_directory": self.settings.CHROMA_PERSIST_DIRECTORY,
            "total_vectors": count,
            "embedding_model": self.settings.EMBEDDING_MODEL,
            "groq_model": self.settings.GROQ_MODEL,
            "has_groq_api_key": bool(self.settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY")),
        }
