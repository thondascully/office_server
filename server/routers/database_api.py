"""
Database Management Router
Handles export/import operations
"""
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse
from datetime import datetime
import shutil
import config
from database import vector_db

router = APIRouter()

@router.get("/export/vectors")
async def export_vectors():
    """Export vector database"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"vectors_{timestamp}.json"
    filepath = config.EXPORTS_DIR / filename

    vector_db.export(filepath)

    return FileResponse(
        path=filepath,
        filename=filename,
        media_type="application/json"
    )

@router.post("/import/vectors")
async def import_vectors(file: UploadFile = File(...)):
    """Import vector database"""
    filepath = config.EXPORTS_DIR / file.filename

    with open(filepath, 'wb') as f:
        shutil.copyfileobj(file.file, f)

    vector_db.import_from(filepath)

    return {"status": "success", "vectors_imported": len(vector_db.vectors)}
