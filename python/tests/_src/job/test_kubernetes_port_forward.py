# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import signal
import socket
import subprocess
import sys
import threading
from unittest.mock import patch

import pytest
from monarch._src.job import (
    _kubernetes_port_forward as pf,
    job_sidecar as js,
    process_guard as pg,
)
from monarch._src.job.process_guard import _Shutdown, _wait_for_socket, ProcessGuard


@pytest.fixture
def kubectl(tmp_path, monkeypatch):
    """A real child process with kubectl's readiness/connection output protocol."""
    executable = tmp_path / "kubectl"
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")

    def install(body):
        executable.write_text(f"#!{sys.executable}\n" + body)
        executable.chmod(0o755)

    install("""
import socket
server = socket.socket()
server.bind(('127.0.0.1', 0))
server.listen()
print(f'Forwarding from 127.0.0.1:{server.getsockname()[1]} -> 26600', flush=True)
while True:
    conn, _ = server.accept()
    with conn:
        command = conn.recv(8)
        if command == b'exit':
            break
        # More than a pipe buffer: a long-lived tunnel must drain stdout.
        for _ in range(8192):
            print('Handling connection for 26600', flush=True)
        conn.sendall(b'ok')
""")
    return install


def gateway():
    return pf.KubernetesPortForward("test", "worker-0", 26600, None)


def test_tunnel_drains_output_and_closes_on_release(kubectl):
    forward = gateway()
    with forward.open() as address:
        process = forward._process
        port = int(address.rsplit(":", 1)[1])
        for _ in range(3):
            with socket.create_connection(("127.0.0.1", port), timeout=5) as conn:
                conn.sendall(b"ping")
                assert conn.recv(2) == b"ok"
        forward.check_alive()
    assert process.poll() is not None
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=1)


def test_dead_forward_is_reported_instead_of_reused(kubectl):
    forward = gateway()
    with forward.open() as address:
        with socket.create_connection(
            ("127.0.0.1", int(address.rsplit(":", 1)[1]))
        ) as conn:
            conn.sendall(b"exit")
        forward._process.wait(timeout=5)
        with pytest.raises(RuntimeError, match="allocation port-forward.*has exited"):
            forward.check_alive()


def test_startup_warning_does_not_hide_readiness(kubectl):
    kubectl("""
import sys, time
print('Warning: deprecated API', file=sys.stderr, flush=True)
print('Forwarding from 127.0.0.1:12345 -> 26600', flush=True)
time.sleep(60)
""")
    forward = gateway()
    with forward.open() as address:
        assert address == "tcp://127.0.0.1:12345"
        forward.check_alive()


def test_startup_error_cleans_up_child(kubectl):
    kubectl("import sys\nprint('permission denied', file=sys.stderr, flush=True)\n")
    forward = gateway()
    with pytest.raises(RuntimeError, match="permission denied"):
        with forward.open():
            pytest.fail("unexpected readiness")
    assert forward._process is None


@pytest.mark.parametrize("output", ["", "partial line"])
def test_startup_timeout_is_bounded(kubectl, monkeypatch, output):
    kubectl(f"import time\nprint({output!r}, end='', flush=True)\ntime.sleep(60)\n")
    monkeypatch.setattr(pf, "_START_TIMEOUT", 0.1)
    forward = gateway()
    with pytest.raises(RuntimeError, match="did not start within"):
        with forward.open():
            pytest.fail("unexpected readiness")
    assert forward._process is None


def test_release_kills_unresponsive_child(kubectl, monkeypatch):
    kubectl("""
import signal, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
print('Forwarding from 127.0.0.1:12345 -> 26600', flush=True)
time.sleep(60)
""")
    monkeypatch.setattr(pf, "_STOP_TIMEOUT", 0.2)
    forward = gateway()
    with forward.open():
        process = forward._process
    assert process.returncode == -signal.SIGKILL


def test_allocation_gateway_survives_client_connections(kubectl, tmp_path):
    """Separate IPC clients reuse one owner; only allocation release closes it."""
    socket_path = str(tmp_path / "sidecar.sock")
    thread = threading.Thread(
        target=js._run_job_sidecar, args=(socket_path,), daemon=True
    )
    with (
        patch.object(js, "attach") as attach,
        patch.object(js, "shutdown_context") as shutdown,
        patch.object(
            js, "create_job_sidecar", side_effect=lambda _: ProcessGuard(socket_path, 0)
        ),
    ):
        thread.start()
        _wait_for_socket(socket_path, timeout=5)
        try:
            addresses = [
                js.ensure_job_gateway("allocation", gateway()) for _ in range(3)
            ]
            assert len(set(addresses)) == 1
            attach.assert_called_once_with(addresses[0])
            port = int(addresses[0].rsplit(":", 1)[1])
            with socket.create_connection(("127.0.0.1", port), timeout=5) as conn:
                conn.sendall(b"ping")
                assert conn.recv(2) == b"ok"
        finally:
            ProcessGuard(socket_path, 0).send(_Shutdown())
            thread.join(timeout=10)
        assert not thread.is_alive()
        shutdown.return_value.get.assert_called_once_with(timeout=5)
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=1)


def test_gateway_failure_propagates_over_sidecar_protocol(kubectl, tmp_path):
    kubectl("print('permission denied', flush=True)\n")
    socket_path = str(tmp_path / "sidecar.sock")
    thread = threading.Thread(
        target=js._run_job_sidecar, args=(socket_path,), daemon=True
    )
    with patch.object(
        js, "create_job_sidecar", side_effect=lambda _: ProcessGuard(socket_path, 0)
    ):
        thread.start()
        _wait_for_socket(socket_path, timeout=5)
        try:
            with pytest.raises(RuntimeError, match="permission denied"):
                js.ensure_job_gateway("allocation", gateway())
        finally:
            ProcessGuard(socket_path, 0).send(_Shutdown())
            thread.join(timeout=10)
        assert not thread.is_alive()


def test_failed_attachment_releases_gateway_and_allows_retry(kubectl):
    state = js._JobSidecarState()
    forward = gateway()
    with patch.object(js, "attach", side_effect=RuntimeError("attach failed")):
        with pytest.raises(RuntimeError, match="attach failed"):
            state.handle_gateway(js.GatewayRequest(forward))
    assert forward._process is None
    with patch.object(js, "attach"), patch.object(js, "shutdown_context"):
        state.handle_gateway(js.GatewayRequest(forward))
        forward.check_alive()
        state.shutdown()
    assert forward._process is None


@pytest.mark.parametrize("drain_fails", [False, True])
def test_gateway_closes_after_runtime_drain_even_on_failure(kubectl, drain_fails):
    state = js._JobSidecarState()
    forward = gateway()

    def drain(*, timeout):
        forward.check_alive()
        if drain_fails:
            raise RuntimeError("drain failed")

    with patch.object(js, "attach"), patch.object(js, "shutdown_context") as shutdown:
        shutdown.return_value.get.side_effect = drain
        state.handle_gateway(js.GatewayRequest(forward))
        process = forward._process
        if drain_fails:
            with pytest.raises(RuntimeError, match="drain failed"):
                state.shutdown()
        else:
            state.shutdown()
        assert process.poll() is not None


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
@pytest.mark.parametrize("connected", [False, True])
def test_sidecar_signal_reaps_tunnel(kubectl, tmp_path, signum, connected):
    """Signals unwind cleanup while blocked in accept or a client's read."""
    worker = tmp_path / "sidecar.py"
    worker.write_text("""
import sys
from unittest.mock import patch
from monarch._src.job import job_sidecar as js
with patch.object(js, 'attach'), patch.object(js, 'shutdown_context'):
    js._run_job_sidecar(sys.argv[1])
""")
    socket_path = str(tmp_path / "sidecar.sock")
    process = subprocess.Popen(
        [sys.executable, str(worker), socket_path], start_new_session=True
    )
    guard = ProcessGuard(socket_path, process.pid)
    try:
        _wait_for_socket(socket_path, pid=process.pid, timeout=5)
        address = guard.send(js.GatewayRequest(gateway())).get()
        port = int(address.rsplit(":", 1)[1])
        if not connected:
            guard._file.close()
            guard._conn.close()
        process.send_signal(signum)
        assert process.wait(timeout=10) == 128 + signum
        assert not os.path.exists(socket_path)
        with pytest.raises(OSError):
            socket.create_connection(("127.0.0.1", port), timeout=1)
    finally:
        if guard._file is not None:
            guard._file.close()
        if guard._conn is not None:
            guard._conn.close()
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def test_forced_sidecar_shutdown_also_kills_its_tunnel(kubectl, tmp_path, monkeypatch):
    worker = tmp_path / "worker.py"
    address_file = tmp_path / "address"
    worker.write_text("""
import socket, subprocess, sys, time
forward = subprocess.Popen(['kubectl'], stdout=subprocess.PIPE, text=True)
with open(sys.argv[1], 'w') as output:
    output.write(forward.stdout.readline())
server = socket.socket(socket.AF_UNIX)
server.bind(sys.argv[2])
server.listen()
# Deliberately ignore shutdown requests; ProcessGuard must kill the group.
while True:
    conn, _ = server.accept()
    conn.close()
""")
    guard = ProcessGuard.create(
        str(tmp_path / "lock"),
        "allocation",
        [sys.executable, str(worker), str(address_file)],
    )
    port = int(address_file.read_text().split(":")[1].split()[0])
    wait = pg._wait_for_pid_exit
    monkeypatch.setattr(pg, "_wait_for_pid_exit", lambda pid: wait(pid, timeout=0.1))
    try:
        assert os.getsid(guard._pid) == guard._pid
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            pass
        # The dummy server may close before the shutdown message is sent.
        try:
            guard.shutdown()
        except BrokenPipeError:
            wait(guard._pid, timeout=0.1)
        with pytest.raises(OSError):
            socket.create_connection(("127.0.0.1", port), timeout=1)
    finally:
        try:
            os.killpg(guard._pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
