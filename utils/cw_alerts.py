"""Alert helpers for SO101 edge scripts (teleoperate, remoteoperate)."""

import logging
import sys
import threading
import time
from typing import Dict, List

from cyberwave import Twin

logger = logging.getLogger(__name__)

# Throttle: minimum seconds between creating the same alert type
_TEMP_ALERT_THROTTLE = 10.0  # 10 seconds per joint
_MQTT_ALERT_THROTTLE = 60.0  # 1 minute
_ERROR_ALERT_THROTTLE = 60.0  # 1 minute (calibration, etc.)
_MOTOR_ALERT_THROTTLE = 10.0  # 10 seconds (fast detection for motor issues)
_MQTT_ERROR_ALERT_THROTTLE = 60.0  # 1 minute for MQTT error rate

_last_alert_times: Dict[str, float] = {}
_alert_lock = threading.Lock()


def _should_create_alert(alert_key: str, throttle_seconds: float) -> bool:
    """Return True if we should create this alert (not throttled)."""
    with _alert_lock:
        now = time.time()
        last = _last_alert_times.get(alert_key, 0.0)
        if now - last < throttle_seconds:
            return False
        _last_alert_times[alert_key] = now
        return True


def _log_alert_failure(alert_type: str, exc: Exception) -> None:
    """Log alert creation failure. Uses stderr so it shows even when logging is disabled."""
    msg = f"Alert creation failed ({alert_type}): {exc}"
    logger.warning(msg)
    try:
        print(msg, file=sys.stderr, flush=True)
    except OSError:
        pass


def create_temperature_alert(
    twin: Twin,
    joint_name: str,
    device: str,
    temperature: float,
    *,
    warning_threshold: float = 42.0,
    critical_threshold: float = 50.0,
) -> bool:
    """
    Create a motor overheating alert if temperature exceeds thresholds.

    Args:
        twin: Twin instance with alerts (twin.alerts)
        joint_name: Joint/motor name (e.g. shoulder_pan)
        device: "leader" or "follower"
        temperature: Current temperature in °C
        warning_threshold: Create warning alert above this (default 42)
        critical_threshold: Create critical alert above this (default 50)

    Returns:
        True if alert was created, False if throttled or below threshold
    """
    if temperature < warning_threshold:
        return False

    severity = "critical" if temperature >= critical_threshold else "warning"
    alert_key = f"temp_{device}_{joint_name}"

    if not _should_create_alert(alert_key, _TEMP_ALERT_THROTTLE):
        return False

    try:
        twin.alerts.create(
            name=f"Motor overheating: {joint_name} ({device})",
            description=f"{joint_name} on {device} is at {temperature:.0f}°C",
            alert_type="motor_overheating",
            severity=severity,
            source_type="edge",
        )
        logger.info(f"Created {severity} alert: {joint_name} ({device}) at {temperature:.0f}°C")
        return True
    except Exception as e:
        _log_alert_failure("temperature", e)
        return False


def create_mqtt_disconnected_alert(twin: Twin) -> bool:
    """
    Create an alert when MQTT is disconnected.

    Args:
        twin: Twin instance with alerts

    Returns:
        True if alert was created, False if throttled
    """
    alert_key = "mqtt_disconnected"
    if not _should_create_alert(alert_key, _MQTT_ALERT_THROTTLE):
        return False

    try:
        twin.alerts.create(
            name="MQTT disconnected",
            description="Lost connection to Cyberwave MQTT broker",
            alert_type="mqtt_disconnected",
            severity="error",
            source_type="edge",
        )
        logger.info("Created MQTT disconnected alert")
        return True
    except Exception as e:
        _log_alert_failure("mqtt_disconnected", e)
        return False


def create_calibration_upload_failed_alert(
    twin: Twin,
    device: str,
    error: Exception,
) -> bool:
    """
    Create an alert when calibration upload to the twin fails.

    Args:
        twin: Twin instance with alerts
        device: "leader" or "follower"
        error: The exception that caused the failure

    Returns:
        True if alert was created
    """
    alert_key = f"calibration_upload_failed_{device}"
    if not _should_create_alert(alert_key, _ERROR_ALERT_THROTTLE):
        return False

    try:
        twin.alerts.create(
            name=f"{device.capitalize()} calibration upload failed",
            description=str(error),
            alert_type="calibration_upload_failed",
            severity="error",
            source_type="edge",
        )
        logger.info(f"Created calibration upload failed alert for {device}: {error}")
        return True
    except Exception as e:
        _log_alert_failure("calibration_upload_failed", e)
        return False


def create_calibration_needed_alert(
    twin: Twin,
    device: str,
    *,
    description: str = "",
) -> bool:
    """
    Create a calibration needed alert for leader or follower.

    Args:
        twin: Twin instance with alerts
        device: "leader" or "follower"
        description: Optional details

    Returns:
        True if alert was created
    """
    try:
        twin.alerts.create(
            name=f"{device.capitalize()} calibration needed",
            description=description or f"The {device} device requires calibration.",
            alert_type="calibration_needed",
            severity="warning",
            source_type="edge",
        )
        logger.info(f"Created calibration needed alert for {device}")
        return True
    except Exception as e:
        _log_alert_failure("calibration_needed", e)
        return False


def create_camera_default_device_alert(
    twin: Twin,
    default_cameras: List[dict],
) -> bool:
    """
    Create an info alert when cameras use default device assignment (no sensors_devices).

    Args:
        twin: Twin instance with alerts (typically the robot twin)
        default_cameras: List of camera dicts with used_default=True, each containing
            setup_name, twin_uuid, video_device (or camera_id)

    Returns:
        True if alert was created, False if empty list
    """
    if not default_cameras:
        return False

    parts = []
    for c in default_cameras:
        setup_name = c.get("setup_name", "camera")
        dev = c.get("video_device") or c.get("camera_id", "?")
        if str(dev).startswith("/dev/"):
            parts.append(f"{setup_name}={dev}")
        else:
            parts.append(f"{setup_name}=/dev/video{dev}")
    description = (
        "Cameras using default device assignment (no sensors_devices or video_device in metadata): "
        + ", ".join(parts)
        + ". Configure sensors_devices in Sensor Settings for stable mapping."
    )

    try:
        twin.alerts.create(
            name="Cameras using default device assignment",
            description=description,
            alert_type="camera_default_device",
            severity="info",
            source_type="edge",
        )
        logger.info(
            "Created camera default device alert: %d camera(s) using pool assignment",
            len(default_cameras),
        )
        return True
    except Exception as e:
        _log_alert_failure("camera_default_device", e)
        return False


def create_motor_error_alert(
    twin: Twin,
    error_count: int,
    *,
    threshold: int = 10,
) -> bool:
    """
    Create an alert when motor/serial error count exceeds threshold.

    Motor failures need fast detection (low threshold, short throttle).

    Args:
        twin: Twin instance with alerts
        error_count: Current motor error count from status tracker
        threshold: Create alert when errors exceed this (default 10)

    Returns:
        True if alert was created, False if throttled or below threshold
    """
    if error_count < threshold:
        return False

    alert_key = "motor_error_rate"
    if not _should_create_alert(alert_key, _MOTOR_ALERT_THROTTLE):
        return False

    try:
        twin.alerts.create(
            name="Motor communication errors",
            description=f"Edge script reported {error_count} motor/serial errors",
            alert_type="motor_error_rate",
            severity="warning",
            source_type="edge",
        )
        logger.info(f"Created motor error alert: {error_count} motor errors")
        return True
    except Exception as e:
        _log_alert_failure("motor_error_rate", e)
        return False


def create_mqtt_error_alert(
    twin: Twin,
    error_count: int,
    *,
    threshold: int = 100,
) -> bool:
    """
    Create an alert when MQTT/queue error count exceeds threshold.

    MQTT errors can tolerate more (100 per 60 sec) before alerting.

    Args:
        twin: Twin instance with alerts
        error_count: Current MQTT error count from status tracker
        threshold: Create alert when errors exceed this (default 100)

    Returns:
        True if alert was created, False if throttled or below threshold
    """
    if error_count < threshold:
        return False

    alert_key = "mqtt_error_rate"
    if not _should_create_alert(alert_key, _MQTT_ERROR_ALERT_THROTTLE):
        return False

    try:
        twin.alerts.create(
            name="MQTT/queue error rate",
            description=f"Edge script reported {error_count} MQTT/queue errors",
            alert_type="mqtt_error_rate",
            severity="warning",
            source_type="edge",
        )
        logger.info(f"Created MQTT error alert: {error_count} MQTT errors")
        return True
    except Exception as e:
        _log_alert_failure("mqtt_error_rate", e)
        return False
