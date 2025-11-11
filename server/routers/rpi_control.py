"""
RPi Control Router
Handles dashboard → RPi command triggers
EASY TO EXTEND: Add new commands here + update dashboard buttons
"""
from fastapi import APIRouter, Request
from services.rpi_manager import rpi_manager

router = APIRouter()

@router.post("/register")
async def trigger_registration(request: Request):
    """Trigger registration on RPi"""
    data = await request.json()
    rpi_id = data.get('rpi_id', 'default')

    rpi_manager.set_command(rpi_id, "register")
    return {"status": "success", "message": f"Registration triggered for {rpi_id}"}

@router.post("/start-stream")
async def start_stream(request: Request):
    """Start camera stream"""
    data = await request.json()
    rpi_id = data.get('rpi_id', 'default')

    rpi_manager.set_command(rpi_id, "start_stream")
    return {"status": "success", "message": f"Stream started for {rpi_id}"}

@router.post("/stop-stream")
async def stop_stream(request: Request):
    """Stop camera stream"""
    data = await request.json()
    rpi_id = data.get('rpi_id', 'default')

    rpi_manager.set_command(rpi_id, "stop_stream")
    rpi_manager.clear_stream(rpi_id)

    return {"status": "success", "message": f"Stream stopped for {rpi_id}"}

@router.post("/calibrate")
async def trigger_calibration(request: Request):
    """Trigger calibration on RPi"""
    data = await request.json()
    rpi_id = data.get('rpi_id', 'default')

    rpi_manager.set_command(rpi_id, "calibrate")
    return {"status": "success", "message": f"Calibration mode activated for {rpi_id}"}

@router.post("/stop-calibrate")
async def stop_calibration(request: Request):
    """Stop calibration"""
    data = await request.json()
    rpi_id = data.get('rpi_id', 'default')

    rpi_manager.set_command(rpi_id, "stop_calibrate")
    rpi_manager.clear_stream(rpi_id)

    return {"status": "success", "message": f"Calibration stopped for {rpi_id}"}

@router.post("/system-toggle")
async def toggle_system(request: Request):
    """Enable/disable the RPi system (to save battery)"""
    data = await request.json()
    rpi_id = data.get('rpi_id', 'default')
    enabled = data.get('enabled', True)

    # Store the system state in the command system
    # The RPi will check this via /api/rpi/commands/{rpi_id}
    rpi_manager.set_command(rpi_id, "system_toggle", {"enabled": enabled})
    
    status = "enabled" if enabled else "disabled"
    return {
        "status": "success", 
        "message": f"System {status} for {rpi_id}",
        "enabled": enabled
    }
