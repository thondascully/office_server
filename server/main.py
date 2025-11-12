"""
Office Map Server - Main Application
Modular FastAPI application with organized routers
"""
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import config
from routers import dashboard, rpi_control, rpi_communication, face_api, database_api

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        from database import metadata_db, vector_db
        
        # Create thread pool executor for CPU-intensive tasks (face recognition)
        # Limit to 2 workers to prevent CPU overload while allowing some concurrency
        executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="face_recognition")
        app.state.face_recognition_executor = executor
        
        print("=" * 70)
        print("  Office Map Server - Starting")
        print("=" * 70)
        print(f"  People in database: {len(metadata_db.people)}")
        print(f"  Vectors in database: {len(vector_db.vectors)}")
        print(f"  Unlabeled people: {len(metadata_db.get_unlabeled())}")
        print(f"  Face recognition executor: {executor._max_workers} workers")
        print("=" * 70)
        print("[Startup] Server ready and accepting connections")
    except Exception as e:
        print(f"[Startup] ERROR during startup: {e}")
        import traceback
        traceback.print_exc()
        raise
    yield
    # Shutdown
    print("[Shutdown] Shutting down face recognition executor...")
    if hasattr(app.state, 'face_recognition_executor'):
        app.state.face_recognition_executor.shutdown(wait=True)
    print("[Shutdown] Server shutting down")

app = FastAPI(title="Office Map Dashboard", lifespan=lifespan)

# Health check endpoint for Railway
@app.get("/health")
async def health_check():
    """Health check endpoint for Railway"""
    return {"status": "ok", "service": "office-map-server"}

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files & templates
app.mount("/static", StaticFiles(directory=str(config.BASE_DIR / "static")), name="static")

# Include routers
app.include_router(dashboard.router, tags=["Dashboard"])
app.include_router(rpi_control.router, prefix="/api/rpi", tags=["RPi Control"])
app.include_router(rpi_communication.router, prefix="/api/rpi", tags=["RPi Communication"])
app.include_router(face_api.router, prefix="/api", tags=["Face Recognition"])
app.include_router(database_api.router, prefix="/api", tags=["Database"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)
