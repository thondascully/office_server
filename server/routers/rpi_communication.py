"""
RPi Communication Router
Handles communication between RPi devices and server
"""
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import Response
from services.rpi_manager import rpi_manager
from datetime import datetime, timezone

router = APIRouter()

@router.post("/heartbeat")
async def rpi_heartbeat(request: Request):
    """RPi sends periodic heartbeat"""
    data = await request.json()
    rpi_id = data.get('rpi_id', 'default')
    status = data.get('status', 'unknown')
    uptime = data.get('uptime', 0)

    rpi_manager.update_heartbeat(rpi_id, status, uptime)

    return {"status": "ok", "server_time": datetime.now(timezone.utc).isoformat()}

@router.get("/commands/{rpi_id}")
async def get_rpi_commands(rpi_id: str):
    """RPi polls this to get commands from dashboard"""
    command, params = rpi_manager.get_command(rpi_id)
    if command:
        print(f"[RPi Comm] RPi '{rpi_id}' received command: {command} (params: {params})")
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
    try:
        # Set a timeout to prevent hanging on large images
        image_bytes = await request.body()
        # Limit image size to prevent memory issues (10MB max)
        if len(image_bytes) > 10 * 1024 * 1024:
            print(f"[RPi Stream] WARNING: Frame from {rpi_id} too large: {len(image_bytes)} bytes")
            return {"status": "error", "message": "Image too large"}
        rpi_manager.update_stream_frame(rpi_id, image_bytes)
        # Log first frame to confirm streaming started
        if not hasattr(receive_stream_frame, '_logged'):
            receive_stream_frame._logged = set()
        if rpi_id not in receive_stream_frame._logged:
            print(f"[RPi Stream] First frame received from {rpi_id} ({len(image_bytes)} bytes) - streaming active")
            receive_stream_frame._logged.add(rpi_id)
        return {"status": "ok"}
    except Exception as e:
        # Don't let stream errors crash the server
        print(f"[RPi Stream] ERROR: Frame error from {rpi_id}: {e}")
        return {"status": "error", "message": str(e)}

@router.get("/stream/{rpi_id}")
async def get_stream_frame(rpi_id: str):
    """Get latest stream frame for dashboard"""
    frame_data = rpi_manager.get_stream_frame(rpi_id)

    if frame_data is None:
        # Check if RPi is connected but not streaming
        if rpi_id in rpi_manager.connected_rpis:
            # RPi is connected but no recent frames - it may not have started streaming yet
            raise HTTPException(
                status_code=404, 
                detail=f"RPi '{rpi_id}' connected but not sending stream frames. Check RPi logs."
            )
        else:
            raise HTTPException(
                status_code=404, 
                detail=f"RPi '{rpi_id}' not connected or stream expired"
            )

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
