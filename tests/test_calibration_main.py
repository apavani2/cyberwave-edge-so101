"""Unit tests for main.py calibration-related functionality."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root for imports (conftest mocks cyberwave before this)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def mock_client():
    """Create a mock Cyberwave client with MQTT and alert update chain."""
    client = MagicMock()
    client.mqtt.publish_command_message = MagicMock()
    mock_alert = MagicMock()
    mock_alert.metadata = {}
    mock_robot = MagicMock()
    mock_robot.alerts.get.return_value = mock_alert
    client.twin.return_value = mock_robot
    return client


@pytest.fixture
def valid_calibration_data():
    """Valid calibration start payload."""
    return {
        "step": "start_calibration",
        "type": "follower",
        "follower_port": "/dev/ttyACM0",
        "follower_id": "follower1",
        "alert_uuid": "alert-uuid-123",
    }


class TestCalibrationConcurrencyGuard:
    """Tests for calibration concurrency guard in _handle_calibration_start."""

    def test_rejects_concurrent_calibration_when_proc_running(
        self, mock_client, valid_calibration_data
    ):
        """When _calibration_proc is running, start_calibration should be rejected."""
        import main as main_module

        fake_proc = MagicMock()
        fake_proc.poll.return_value = None  # None means still running

        with patch.object(main_module, "_calibration_proc", fake_proc):
            with patch.object(main_module, "threading") as mock_threading:
                mock_threading.Thread = MagicMock()

                main_module._handle_calibration_start(
                    mock_client,
                    "twin-uuid-456",
                    valid_calibration_data,
                )

        mock_client.mqtt.publish_command_message.assert_called_once_with(
            "twin-uuid-456",
            {"status": "error", "reason": "calibration_already_running"},
        )
        mock_threading.Thread.return_value.start.assert_not_called()

    def test_allows_calibration_when_no_proc_running(
        self, mock_client, valid_calibration_data
    ):
        """When no calibration is running, start_calibration should proceed."""
        import main as main_module

        with patch.object(main_module, "_calibration_proc", None):
            with patch.object(main_module, "threading") as mock_threading:
                mock_thread = MagicMock()
                mock_threading.Thread.return_value = mock_thread

                main_module._handle_calibration_start(
                    mock_client,
                    "twin-uuid-456",
                    valid_calibration_data,
                )

        mock_threading.Thread.assert_called_once()
        call_kwargs = mock_threading.Thread.call_args.kwargs
        assert call_kwargs["kwargs"]["alert_uuid"] == "alert-uuid-123"
        mock_thread.start.assert_called_once()

    def test_allows_calibration_when_proc_finished(
        self, mock_client, valid_calibration_data
    ):
        """When _calibration_proc exists but poll() returns non-None (finished), allow new start."""
        import main as main_module

        fake_proc = MagicMock()
        fake_proc.poll.return_value = 0  # Process finished

        with patch.object(main_module, "_calibration_proc", fake_proc):
            with patch.object(main_module, "threading") as mock_threading:
                mock_thread = MagicMock()
                mock_threading.Thread.return_value = mock_thread

                main_module._handle_calibration_start(
                    mock_client,
                    "twin-uuid-456",
                    valid_calibration_data,
                )

        mock_threading.Thread.assert_called_once()
        mock_thread.start.assert_called_once()


class TestCalibrationCommandPassthrough:
    """Tests that calibration commands pass alert_uuid and twin_uuid to subprocess."""

    def test_run_calibration_with_advance_includes_alert_and_twin_uuid(self):
        """_run_calibration_with_advance builds cmd with --alert-uuid and --twin-uuid."""
        import main as main_module

        mock_client = MagicMock()
        mock_client.mqtt.publish_command_message = MagicMock()

        with patch.object(main_module, "subprocess") as mock_subprocess:
            mock_popen = MagicMock()
            mock_popen.wait.return_value = 0
            mock_popen.stdin = None
            mock_subprocess.Popen.return_value = mock_popen

            main_module._run_calibration_with_advance(
                mock_client,
                twin_uuid="twin-123",
                device_type="follower",
                port="/dev/ttyACM0",
                device_id="follower1",
                alert_uuid="alert-456",
            )

        call_args = mock_subprocess.Popen.call_args[0][0]
        assert "--alert-uuid" in call_args
        assert "alert-456" in call_args
        assert "--twin-uuid" in call_args
        assert "twin-123" in call_args

    def test_run_calibration_without_alert_uuid_omits_alert_args(self):
        """When alert_uuid is None, --alert-uuid and --twin-uuid are not passed."""
        import main as main_module

        mock_client = MagicMock()
        mock_client.mqtt.publish_command_message = MagicMock()

        with patch.object(main_module, "subprocess") as mock_subprocess:
            mock_popen = MagicMock()
            mock_popen.wait.return_value = 0
            mock_popen.stdin = None
            mock_subprocess.Popen.return_value = mock_popen

            main_module._run_calibration_with_advance(
                mock_client,
                twin_uuid="twin-123",
                device_type="follower",
                port="/dev/ttyACM0",
                device_id="follower1",
                alert_uuid=None,
            )

        call_args = mock_subprocess.Popen.call_args[0][0]
        assert "--alert-uuid" not in call_args
        assert "--twin-uuid" not in call_args


class TestHandleCommandCalibrate:
    """Tests for handle_command with calibrate command."""

    def test_calibrate_start_dispatches_to_handle_calibration_start(
        self, mock_client, valid_calibration_data
    ):
        """handle_command with calibrate/start_calibration calls _handle_calibration_start."""
        import main as main_module

        with patch.object(
            main_module, "_handle_calibration_start", MagicMock()
        ) as mock_start:
            with patch.object(main_module, "_calibration_proc", None):
                with patch.object(main_module, "threading") as mock_threading:
                    mock_threading.Thread.return_value = MagicMock()

                    main_module.handle_command(
                        mock_client,
                        "twin-uuid-456",
                        "calibrate",
                        {"data": valid_calibration_data},
                    )

        mock_start.assert_called_once_with(
            mock_client, "twin-uuid-456", valid_calibration_data
        )


class TestStopCurrentOperationGracefulShutdown:
    """Tests for _stop_current_operation graceful shutdown behavior."""

    def test_sets_stop_event_before_force_disconnect(self):
        """_stop_current_operation sets stop_event first, then waits before force disconnect."""
        import main as main_module

        stop_event = __import__("threading").Event()
        mock_follower = MagicMock()
        mock_thread = MagicMock()
        mock_thread.is_alive.side_effect = [True, True, False]

        with patch.object(main_module, "_current_thread", mock_thread):
            with patch.object(main_module, "_current_follower", mock_follower):
                with patch.object(main_module, "_operation_stop_event", stop_event):
                    with patch.object(main_module, "_calibration_proc", None):
                        main_module._stop_current_operation()

        assert stop_event.is_set()
        mock_thread.join.assert_called()
        calls = mock_thread.join.call_args_list
        assert len(calls) >= 1
        assert calls[0][1]["timeout"] == main_module.GRACEFUL_JOIN_TIMEOUT

    def test_does_not_force_disconnect_when_thread_exits_gracefully(self):
        """When thread exits after stop_event, no force disconnect is needed."""
        import main as main_module

        stop_event = __import__("threading").Event()
        mock_follower = MagicMock()
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = False

        with patch.object(main_module, "_current_thread", mock_thread):
            with patch.object(main_module, "_current_follower", mock_follower):
                with patch.object(main_module, "_operation_stop_event", stop_event):
                    with patch.object(main_module, "_calibration_proc", None):
                        main_module._stop_current_operation()

        mock_follower.disconnect.assert_not_called()

    def test_force_disconnects_when_thread_does_not_exit_gracefully(self):
        """When thread does not exit within graceful timeout, force disconnect is called."""
        import main as main_module

        stop_event = __import__("threading").Event()
        mock_follower = MagicMock()
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True  # Thread never exits

        with patch.object(main_module, "_current_thread", mock_thread):
            with patch.object(main_module, "_current_follower", mock_follower):
                with patch.object(main_module, "_operation_stop_event", stop_event):
                    with patch.object(main_module, "_calibration_proc", None):
                        main_module._stop_current_operation()

        mock_follower.disconnect.assert_called_once()

    def test_stops_calibration_subprocess_gracefully(self):
        """_stop_current_operation terminates calibration first, then kill if needed."""
        import main as main_module

        mock_proc = MagicMock()
        mock_proc.wait.return_value = None
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = False

        with patch.object(main_module, "_current_thread", mock_thread):
            with patch.object(main_module, "_current_follower", None):
                with patch.object(main_module, "_operation_stop_event", None):
                    with patch.object(main_module, "_calibration_proc", mock_proc):
                        main_module._stop_current_operation()

        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called_once_with(timeout=5)
        mock_proc.kill.assert_not_called()


class TestSingleScriptAtATime:
    """Tests for single-script-at-a-time concurrency guards."""

    def test_run_script_command_calls_stop_before_running(self):
        """_run_script_command calls _stop_current_operation before running script."""
        import main as main_module

        mock_client = MagicMock()
        mock_client.mqtt.publish_command_message = MagicMock()

        with patch.object(main_module, "_stop_current_operation", MagicMock()) as mock_stop:
            with patch.object(main_module, "subprocess") as mock_subprocess:
                mock_subprocess.run.return_value = MagicMock(returncode=0)

                main_module._run_script_command(
                    mock_client,
                    twin_uuid="twin-123",
                    script_name="find_port",
                    data={},
                )

        mock_stop.assert_called_once()

    def test_handle_calibration_start_calls_stop_when_not_rejecting(self):
        """_handle_calibration_start calls _stop_current_operation before starting."""
        import main as main_module

        valid_data = {
            "type": "follower",
            "follower_port": "/dev/ttyACM0",
            "follower_id": "follower1",
        }

        with patch.object(main_module, "_calibration_proc", None):
            with patch.object(main_module, "_stop_current_operation", MagicMock()) as mock_stop:
                with patch.object(main_module, "threading") as mock_threading:
                    mock_threading.Thread.return_value = MagicMock()
                    mock_client = MagicMock()
                    mock_client.mqtt.publish_command_message = MagicMock()
                    mock_robot = MagicMock()
                    mock_alert = MagicMock()
                    mock_alert.metadata = {}
                    mock_robot.alerts.get.return_value = mock_alert
                    mock_client.twin.return_value = mock_robot

                    main_module._handle_calibration_start(
                        mock_client,
                        "twin-123",
                        valid_data,
                    )

        mock_stop.assert_called_once()
