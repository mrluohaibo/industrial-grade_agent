"""
FastAPI application entry point for RAG Document Processing API.

Author: RAG Team
Created: 2026-03-02
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from bz_agent.api.document_routes import document_router
from bz_agent.api.rag_routes import rag_router
from utils.config_init import application_conf
from utils.logger_config import get_logger

logger = get_logger(__name__)

# Global document processor (initialized on startup)
_processor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting RAG Document Processing API...")
    logger.info("API Version: 1.0.0")

    # Initialize document processor
    from bz_agent.rag.document_processor import DocumentProcessor

    global _processor
    try:
        _processor = DocumentProcessor()
        logger.info("Document processor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize document processor: {e}")
        raise

    # Configure routers with shared processor
    document_router.get_processor = lambda: _processor
    rag_router.get_processor = lambda: _processor

    logger.info("API startup completed")

    yield

    # Shutdown
    logger.info("Shutting down RAG Document Processing API...")


# Create FastAPI application
app = FastAPI(
    title="RAG Document Processing API",
    description="Enterprise-grade document processing API for RAG applications",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ============================================================================
# Middleware
# ============================================================================


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    # Get current time for logging
    import time
    request_start = time.time()

    # Process request
    response = await call_next(request)

    # Log request
    process_time = (time.time() - request_start) * 1000
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.2f}ms"
    )

    return response


# ============================================================================
# Exception Handlers
# ============================================================================


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "code": 422,
            "message": "Validation error",
            "data": {"errors": exc.errors()}
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "code": 500,
            "message": "Internal server error",
            "data": {"detail": str(exc)}
        }
    )


# ============================================================================
# Routes
# ============================================================================


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns the status of the API and its dependencies.
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "rag-document-api"
    }


# Include routers
app.include_router(document_router)
app.include_router(rag_router)


# ============================================================================
# Startup
# ============================================================================


if __name__ == "__main__":
    import uvicorn

    # Get API configuration
    api_host = application_conf.get_properties("api.host", "0.0.0.0")
    api_port = application_conf.get_properties("api.port", 8000)
    api_debug = application_conf.get_properties("api.debug", False)

    logger.info(f"Starting API server on {api_host}:{api_port}")

    uvicorn.run(
        "api.main:app",
        host=api_host,
        port=api_port,
        reload=api_debug,
        log_config=None,  # Use our custom logger
    )
