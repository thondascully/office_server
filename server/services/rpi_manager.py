"""
RPi State Management Service
Centralized management of connected RPi devices
"""
from datetime import datetime
from typing import Dict, Optional
import yaml
from pathlib import Path
import config

class RPiManager:
    """Manages RPi device state and configuration"""

    def __init__(self):
        self.connected_rpis: Dict[str, dict] = {}
        self.stream_frames: Dict[str, dict] = {}
        self.configs_dir = config.DATA_DIR / "rpi_configs"
        self.configs_dir.mkdir(exist_ok=True)

    def update_heartbeat(self, rpi_id: str, status: str, uptime: int):
        """Update RPi heartbeat"""
        if rpi_id not in self.connected_rpis:
            self.connected_rpis[rpi_id] = {}

        self.connected_rpis[rpi_id]['last_seen'] = datetime.now()
        self.connected_rpis[rpi_id]['status'] = status
        self.connected_rpis[rpi_id]['uptime'] = uptime

    def set_command(self, rpi_id: str, command: str, params: dict = None):
        """Set a command for RPi to execute"""
        if rpi_id not in self.connected_rpis:
            self.connected_rpis[rpi_id] = {}

        self.connected_rpis[rpi_id]["command"] = command
        self.connected_rpis[rpi_id]["params"] = params or {}

    def get_command(self, rpi_id: str) -> tuple[Optional[str], dict]:
        """Get and clear command for RPi"""
        if rpi_id not in self.connected_rpis:
            self.connected_rpis[rpi_id] = {"command": None}
            return None, {}

        command = self.connected_rpis[rpi_id].get("command")
        params = self.connected_rpis[rpi_id].get("params", {})
        self.connected_rpis[rpi_id]["command"] = None
        return command, params

    def update_stream_frame(self, rpi_id: str, image_bytes: bytes):
        """Update stream frame for RPi"""
        self.stream_frames[rpi_id] = {
            'data': image_bytes,
            'timestamp': datetime.now()
        }

    def get_stream_frame(self, rpi_id: str) -> Optional[dict]:
        """Get latest stream frame"""
        if rpi_id not in self.stream_frames:
            return None

        frame_data = self.stream_frames[rpi_id]
        # Check if frame is recent (within 2 seconds)
        if (datetime.now() - frame_data['timestamp']).total_seconds() > 2:
            return None

        return frame_data

    def clear_stream(self, rpi_id: str):
        """Clear stream data for RPi"""
        if rpi_id in self.stream_frames:
            del self.stream_frames[rpi_id]

    def get_active_rpis(self, timeout_seconds: int = 60) -> list:
        """Get list of active RPis"""
        active = []
        current_time = datetime.now()
        for rpi_id, info in self.connected_rpis.items():
            last_seen = info.get('last_seen')
            if last_seen and (current_time - last_seen).total_seconds() < timeout_seconds:
                active.append(rpi_id)
        return active

    def get_rpi_status(self, rpi_id: str) -> Optional[dict]:
        """Get status of specific RPi"""
        if rpi_id not in self.connected_rpis:
            return None

        info = self.connected_rpis[rpi_id]
        last_seen = info.get('last_seen')

        if not last_seen:
            return None

        current_time = datetime.now()
        is_streaming = rpi_id in self.stream_frames and \
                      (current_time - self.stream_frames[rpi_id]['timestamp']).total_seconds() < 2

        return {
            "rpi_id": rpi_id,
            "status": info.get('status', 'unknown'),
            "last_seen": last_seen.isoformat(),
            "uptime": info.get('uptime', 0),
            "current_mode": info.get('status', 'idle'),
            "is_streaming": is_streaming
        }

    def load_config(self, rpi_id: str) -> dict:
        """Load RPi configuration"""
        config_file = self.configs_dir / f"{rpi_id}.yaml"
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        # Default config
        return {
            'tripwires': {'outer_x': 800, 'inner_x': 1800},
            'detection_interval': 1.0,
            'burst_size': 10
        }

    def save_config(self, rpi_id: str, cfg: dict):
        """Save RPi configuration"""
        config_file = self.configs_dir / f"{rpi_id}.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(cfg, f)

# Global instance
rpi_manager = RPiManager()
