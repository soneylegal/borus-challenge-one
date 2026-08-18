"""FastAPI main application module for Borus."""

from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import get_settings
from app.core.loader import PDFDocLoader
from app.core.rag import RAGPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("borus")

settings = get_settings()
rag_pipeline: RAGPipeline | None = None
doc_loader: PDFDocLoader | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup and shutdown procedures."""
    global rag_pipeline, doc_loader
    logger.info("Initializing Borus...")

    # Initialize components
    doc_loader = PDFDocLoader(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    rag_pipeline = RAGPipeline(settings=settings)

    # Auto-ingest docs if ChromaDB collection is empty and docs_source exists
    stats = rag_pipeline.get_stats()
    if stats["total_vectors"] == 0:
        docs_dir = Path(settings.DOCS_SOURCE_DIR)
        if docs_dir.exists():
            logger.info(f"Empty collection detected. Auto-ingesting documents from {docs_dir}...")
            chunks = doc_loader.load_directory(docs_dir)
            if chunks:
                rag_pipeline.index_chunks(chunks)
                logger.info(f"Auto-ingested {len(chunks)} chunks on startup.")
    else:
        logger.info(f"ChromaDB ready with {stats['total_vectors']} vector embeddings.")

    yield
    logger.info("Shutting down Borus.")


app = FastAPI(
    title="Borus",
    description="Borus — RAG Agent para consulta e recuperação semântica de documentação técnica de backend",
    version=settings.VERSION,
    lifespan=lifespan,
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static assets
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# Pydantic Request/Response Models
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message text content")


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Developer question to query the knowledge base")
    history: list[ChatMessage] | None = Field(
        default=None, description="Optional conversational history"
    )
    top_k: int | None = Field(
        default=None, ge=1, le=20, description="Optional override for number of retrieved chunks"
    )


class SourceItem(BaseModel):
    source: str
    page: int | None = Field(default=None, description="PDF page number if available")
    similarity_score: float
    snippet: str
    metadata: dict[str, Any]


class ChatResponse(BaseModel):
    query: str
    answer: str
    model: str
    sources: list[SourceItem]
    chunks_retrieved: int


class IngestResponse(BaseModel):
    status: str
    message: str
    indexed_chunks: int
    total_collection_count: int


class HealthResponse(BaseModel):
    status: str
    version: str
    groq_configured: bool
    total_vectors: int
    collection: str
    embedding_model: str
    groq_model: str


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
async def serve_index():
    """Serve the web UI single page application."""
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Borus API is running. Access /docs for API documentation."}


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Healthcheck endpoint reporting status of Groq API, ChromaDB vector store, and embeddings."""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline not yet initialized.")

    stats = rag_pipeline.get_stats()
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        groq_configured=stats["has_groq_api_key"],
        total_vectors=stats["total_vectors"],
        collection=stats["collection_name"],
        embedding_model=stats["embedding_model"],
        groq_model=stats["groq_model"],
    )


@app.get("/stats", tags=["Monitoring"])
async def get_stats():
    """Get vector store index details, configuration, and document counts."""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized.")
    return rag_pipeline.get_stats()


@app.post("/ingest", response_model=IngestResponse, tags=["Knowledge Ingestion"])
async def ingest_documents():
    """Scan and index all PDF (.pdf) documents from the configured `docs_source` directory."""
    if not rag_pipeline or not doc_loader:
        raise HTTPException(status_code=503, detail="Service not initialized.")

    docs_dir = Path(settings.DOCS_SOURCE_DIR)
    if not docs_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Diretório de documentos '{settings.DOCS_SOURCE_DIR}' não foi encontrado.",
        )

    chunks = doc_loader.load_directory(docs_dir)
    if not chunks:
        return IngestResponse(
            status="warning",
            message=f"Nenhum arquivo .pdf encontrado em '{docs_dir}'.",
            indexed_chunks=0,
            total_collection_count=rag_pipeline.collection.count(),
        )

    result = rag_pipeline.index_chunks(chunks)
    return IngestResponse(
        status="success",
        message=f"Ingestão concluída com sucesso. {result['indexed_count']} chunks de páginas PDF foram processados.",
        indexed_chunks=result["indexed_count"],
        total_collection_count=result["total_collection_count"],
    )


@app.post("/ingest/upload", response_model=IngestResponse, tags=["Knowledge Ingestion"])
async def upload_document(file: UploadFile = File(...)):
    """Upload and index an individual PDF (.pdf) document directly."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Apenas arquivos PDF (.pdf) são aceitos para ingestão.",
        )

    if not rag_pipeline or not doc_loader:
        raise HTTPException(status_code=503, detail="Service not initialized.")

    content_bytes = await file.read()
    chunks = doc_loader.load_pdf_bytes(content_bytes, filename=file.filename)
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="Arquivo PDF vazio, protegido por senha ou sem texto extraível.",
        )

    result = rag_pipeline.index_chunks(chunks)
    return IngestResponse(
        status="success",
        message=f"PDF '{file.filename}' indexado com sucesso ({len(chunks)} chunks gerados).",
        indexed_chunks=len(chunks),
        total_collection_count=result["total_collection_count"],
    )


@app.post("/chat", response_model=ChatResponse, tags=["RAG Query"])
async def chat_query(request: ChatRequest):
    """Query the technical documentation using vector similarity search and Groq LLM answer generation."""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized.")

    history_payload = None
    if request.history:
        history_payload = [{"role": msg.role, "content": msg.content} for msg in request.history]

    result = rag_pipeline.answer_query(
        query=request.query,
        chat_history=history_payload,
        top_k=request.top_k,
    )

    return ChatResponse(
        query=result["query"],
        answer=result["answer"],
        model=result["model"],
        sources=[
            SourceItem(
                source=s["source"],
                page=s.get("page"),
                similarity_score=s["similarity_score"],
                snippet=s["snippet"],
                metadata=s.get("metadata", {}),
            )
            for s in result["sources"]
        ],
        chunks_retrieved=result["chunks_retrieved"],
    )
