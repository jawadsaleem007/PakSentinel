"""
Task 7 — Middleware [30 Marks]

- Request logging middleware to console and rotating file
- Rate limiting (100 req/min for /classify, 10 req/min for /classify/batch)
"""

import time
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from slowapi import Limiter
from slowapi.util import get_remote_address

# ──────────────────────────────────────────────────────────
#  Logging Setup
# ──────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Configure logger
api_logger = logging.getLogger("paksentinel_api")
api_logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(console_format)
api_logger.addHandler(console_handler)

# Rotating file handler (10MB max, keep 5 backups)
file_handler = RotatingFileHandler(
    LOG_DIR / "api_requests.log",
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=5,
    encoding='utf-8',
)
file_handler.setLevel(logging.INFO)
file_format = logging.Formatter(
    '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(file_format)
api_logger.addHandler(file_handler)


# ──────────────────────────────────────────────────────────
#  Request Logging Middleware
# ──────────────────────────────────────────────────────────
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that logs all incoming requests and response times."""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Log request
        client_ip = request.client.host if request.client else "unknown"
        api_logger.info(
            f"REQUEST | {request.method} {request.url.path} | "
            f"Client: {client_ip} | "
            f"Query: {dict(request.query_params)}"
        )
        
        # Process request
        try:
            response = await call_next(request)
            elapsed = (time.time() - start_time) * 1000  # ms
            
            # Log response
            api_logger.info(
                f"RESPONSE | {request.method} {request.url.path} | "
                f"Status: {response.status_code} | "
                f"Time: {elapsed:.1f}ms"
            )
            
            # Add processing time header
            response.headers["X-Processing-Time-Ms"] = f"{elapsed:.1f}"
            
            return response
            
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            api_logger.error(
                f"ERROR | {request.method} {request.url.path} | "
                f"Error: {str(e)} | Time: {elapsed:.1f}ms"
            )
            raise


# ──────────────────────────────────────────────────────────
#  Rate Limiter
# ──────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)
