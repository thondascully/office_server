"""
Dashboard API Router
Handles dashboard UI and statistics
"""
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import config
from database import metadata_db, vector_db
from models import LabelRequest
from services.rpi_manager import rpi_manager

router = APIRouter()
templates = Jinja2Templates(directory=str(config.BASE_DIR / "templates"))

@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@router.get("/api/dashboard/stats")
async def get_stats():
    """Get dashboard statistics"""
    try:
        all_people = metadata_db.get_all()
        present = [p for p in all_people if p.state == "in"]
        unlabeled = metadata_db.get_unlabeled()
        active_rpis = rpi_manager.get_active_rpis()

        return {
            "total_people": len(all_people),
            "people_present": len(present),
            "unlabeled_count": len(unlabeled),
            "vector_count": len(vector_db.vectors),
            "connected_rpis": active_rpis
        }
    except Exception as e:
        import traceback
        print(f"Error in get_stats: {e}")
        traceback.print_exc()
        raise


@router.get("/api/dashboard/people")
async def get_all_people():
    """Get all people with their info"""
    people = metadata_db.get_all()

    result = []
    for person in people:
        thumbnail = None
        if person.image_paths:
            try:
                img_path = person.image_paths[-1]
                if img_path and Path(img_path).exists():
                    thumbnail = f"/static/images/{Path(img_path).name}"
            except Exception:
                pass  # Skip thumbnail if there's an error

        result.append({
            "person_id": person.person_id,
            "name": person.name,
            "state": person.state,
            "image_count": len(person.image_paths),
            "thumbnail": thumbnail,
            "last_seen": person.last_seen.isoformat(),
            "entered_at": person.entered_at.isoformat() if person.entered_at else None,
            "created_at": person.created_at.isoformat(),
            "last_exit": person.last_exit.isoformat() if person.last_exit else None,
            "last_similarity": person.last_similarity,
            "visit_count": person.visit_count,
            "is_labeled": person.name is not None
        })

    return result

@router.post("/api/dashboard/label")
async def label_person(req: LabelRequest):
    """Label an unlabeled person"""
    metadata_db.update_name(req.person_id, req.name)
    return {"status": "success", "person_id": req.person_id, "name": req.name}

@router.delete("/api/dashboard/person/{person_id}")
async def delete_person(person_id: str):
    """Delete a person from the system"""
    vector_db.delete(person_id)
    metadata_db.delete(person_id)
    return {"status": "success", "person_id": person_id}
