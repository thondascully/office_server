"""
RPi Communication Router
Handles communication between RPi devices and server
"""
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import Response
from services.rpi_manager import rpi_manager
from datetime import datetime

router = APIRouter()

@router.post("/heartbeat")
async def rpi_heartbeat(request: Request):
    """RPi sends periodic heartbeat"""
    data = await request.json()
    rpi_id = data.get('rpi_id', 'default')
    status = data.get('status', 'unknown')
    uptime = data.get('uptime', 0)

    rpi_manager.update_heartbeat(rpi_id, status, uptime)

    return {"status": "ok", "server_time": datetime.now().isoformat()}

@router.get("/commands/{rpi_id}")
async def get_rpi_commands(rpi_id: str):
    """RPi polls this to get commands from dashboard"""
    command, params = rpi_manager.get_command(rpi_id)
    return {"command": command, "params": params}

@router.get("/config/{rpi_id}")
async def get_rpi_config(rpi_id: str):
    """Get RPi configuration"""
    cfg = rpi_manager.load_config(rpi_id)
    return cfg

@router.post("/config/{rpi_id}")
async def update_rpi_config(rpi_id: str, request: Request):
    """Update RPi configuration"""
    data = await request.json()
    cfg = rpi_manager.load_config(rpi_id)

    # Update tripwires if provided
    if 'tripwires' in data:
        cfg['tripwires'] = data['tripwires']

    rpi_manager.save_config(rpi_id, cfg)
    return {"status": "success", "message": "Config updated and saved"}

@router.post("/stream/{rpi_id}")
async def receive_stream_frame(rpi_id: str, request: Request):
    """Receive stream frame from RPi"""
    image_bytes = await request.body()
    rpi_manager.update_stream_frame(rpi_id, image_bytes)
    return {"status": "ok"}

@router.get("/stream/{rpi_id}")
async def get_stream_frame(rpi_id: str):
    """Get latest stream frame for dashboard"""
    frame_data = rpi_manager.get_stream_frame(rpi_id)

    if frame_data is None:
        raise HTTPException(status_code=404, detail="No stream available or expired")

    return Response(content=frame_data['data'], media_type="image/jpeg")

@router.get("/status")
async def get_rpi_status():
    """Get status of all connected RPis"""
    result = {"rpis": []}

    for rpi_id in rpi_manager.connected_rpis.keys():
        status = rpi_manager.get_rpi_status(rpi_id)
        if status:
            result["rpis"].append(status)

    return result
