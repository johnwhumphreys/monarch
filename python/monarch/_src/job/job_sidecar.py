# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Per-job sidecar command server.

The sidecar process is keyed by a job ``apply_id`` and kept alive by ``ProcessGuard``.
"""

import os
import pickle
import socket
import sys
import threading
import time
import traceback
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
from typing import Any, Protocol

from monarch._rust_bindings.monarch_hyperactor.channel import BindSpec
from monarch._src.job.process_guard import find_process, ProcessGuard
from monarch.actor import attach, enable_transport, HostMesh, shutdown_context
from monarch.config import get_runtime_config

_JOB_SIDECAR_WORKER_MODULE = "monarch._src.job._job_sidecar_worker"

try:
    from __manifest__ import fbmake  # noqa

    _IN_PAR: bool = bool(fbmake.get("par_style"))
except ImportError:
    _IN_PAR = False


def job_sidecar_lock_path(apply_id: str) -> str:
    """Return the lock path for the per-job sidecar process."""
    return f"/tmp/monarch_job_sidecar_{apply_id}.lock"


def spawn_module(
    lock_path: str,
    config_key: object,
    module_name: str,
    runtime_transport: str | None = None,
    attach_to: str | None = None,
    process_name: str | None = None,
) -> ProcessGuard:
    """Launch a Python module as a ``ProcessGuard``-managed background process."""
    env = (
        {"HYPERACTOR_PROCESS_NAME": process_name} if process_name is not None else None
    )
    if _IN_PAR:
        command = [sys.argv[0]]
        env = {**(env or {}), "PAR_MAIN_OVERRIDE": module_name}
    else:
        if not sys.executable:
            raise RuntimeError("no python executable available")
        command = [sys.executable, "-m", module_name]
    if runtime_transport is not None:
        command.extend(["--runtime-transport", runtime_transport])
    if attach_to is not None:
        command.extend(["--attach-to", attach_to])
    return ProcessGuard.create(lock_path, config_key, command, env=env)


def create_job_sidecar(
    apply_id: str,
    attach_to: str | None = None,
) -> ProcessGuard:
    """Ensure the per-job sidecar process is running and return its guard."""
    runtime_transport = sidecar_transport_from_runtime()
    return spawn_module(
        job_sidecar_lock_path(apply_id),
        apply_id,
        _JOB_SIDECAR_WORKER_MODULE,
        runtime_transport=runtime_transport,
        attach_to=attach_to,
        process_name="job_sidecar",
    )


def find_job_sidecar(apply_id: str) -> ProcessGuard | None:
    """Return the per-job sidecar process guard if it exists."""
    return find_process(job_sidecar_lock_path(apply_id))


def stop_job_sidecar(apply_id: str) -> None:
    """Shut down the per-job sidecar process for an apply id if it exists."""
    guard = find_job_sidecar(apply_id)
    if guard is not None:
        guard.shutdown()


class Gateway(Protocol):
    """Scheduler-specific connection held open for an allocation's lifetime."""

    def open(self) -> AbstractContextManager[str]: ...

    def check_alive(self) -> None: ...


@dataclass
class GatewayRequest:
    gateway: Gateway


def ensure_job_gateway(apply_id: str, gateway: Gateway) -> str:
    """Acquire the allocation's gateway without transferring ownership to the caller."""
    response = create_job_sidecar(apply_id).send(GatewayRequest(gateway)).get()
    if isinstance(response, dict) and "error" in response:
        raise RuntimeError(response["error"])
    if not isinstance(response, str) or response == "ok":
        raise RuntimeError(f"unexpected gateway response: {response!r}")
    return response


def sidecar_transport_from_runtime() -> str | None:
    """Return the Runtime-layer default transport for sidecar startup."""
    transport = get_runtime_config().get("default_transport")
    if transport is None:
        return None
    if isinstance(transport, str):
        return transport
    return str(BindSpec(transport))


def configure_sidecar_transport(transport: str | None) -> None:
    """Enable sidecar startup transport before actor context bootstrap."""
    if transport is not None:
        enable_transport(transport)


@dataclass
class ClearMountsRequest:
    """Clear the job sidecar's mount state."""


@dataclass
class MountsRequest:
    """Refresh the job sidecar's mount state."""

    mounts: Any
    host_meshes: dict[str, HostMesh]


@dataclass
class TelemetryRequest:
    """Open or refresh the job sidecar's telemetry state."""

    apply_id: str
    config: dict[str, object]
    host_meshes: dict[str, HostMesh]
    spawn_worker_collectors: bool = True


@dataclass
class AdminUrlRequest:
    """Set the mesh-admin URL in the job sidecar process."""

    apply_id: str
    admin_url: str


class _JobSidecarState:
    """State owned by the background job sidecar process."""

    def __init__(self) -> None:
        self._mounts_handle: Any | None = None
        self._mounts_key: bytes | None = None
        self._telemetry_handle: Any | None = None
        self._apply_id: str | None = None
        self._gateway: Gateway | None = None
        self._gateway_address: str | None = None
        self._gateway_stack = ExitStack()

    def handle_gateway(self, request: GatewayRequest) -> str:
        if self._gateway is None:
            # Roll back the tunnel if attachment fails. Once acquired, later
            # clients reuse it; their connection settings cannot replace a live
            # sidecar's process-global actor attachment.
            with ExitStack() as stack:
                address = stack.enter_context(request.gateway.open())
                attach(address)
                self._gateway = request.gateway
                self._gateway_address = address
                self._gateway_stack = stack.pop_all()
        self._gateway.check_alive()
        assert self._gateway_address is not None
        return self._gateway_address

    def _ensure_apply_id(self, apply_id: str) -> None:
        if self._apply_id is None:
            self._apply_id = apply_id
        elif self._apply_id != apply_id:
            raise RuntimeError(f"job sidecar already owns apply id {self._apply_id}")

    def handle_mounts(self, request: MountsRequest) -> str:
        # The request carries declarative mount config. Compare that serialized
        # config so edited mounts replace the live handles, while unchanged
        # config refreshes existing remote mounts in-place.
        # @lint-ignore PYTHONPICKLEISBAD
        mounts_key = pickle.dumps(request.mounts)
        t0 = time.time()
        if self._mounts_handle is None:
            _dbg("initialising mounts")
            self._mounts_handle = request.mounts.open(request.host_meshes)
            self._mounts_key = mounts_key
            _dbg(f"mounts opened in {time.time() - t0:.2f}s")
            return "ok"

        if mounts_key != self._mounts_key:
            _dbg("replacing mounts")
            self._mounts_handle.close()
            self._mounts_handle = request.mounts.open(request.host_meshes)
            self._mounts_key = mounts_key
            _dbg(f"mounts replaced in {time.time() - t0:.2f}s")
            return "ok"

        _dbg("refreshing mounts")
        self._mounts_handle.refresh()
        _dbg(f"refresh complete in {time.time() - t0:.2f}s")
        return "ok"

    def clear_mounts(self) -> str:
        """Close any live mounts and reset mount state."""
        if self._mounts_handle is not None:
            self._mounts_handle.close()
            self._mounts_handle = None
            self._mounts_key = None
        return "ok"

    def handle_telemetry(self, request: TelemetryRequest) -> object:
        from monarch._src.job.telemetry_config import _TelemetryHandle

        self._ensure_apply_id(request.apply_id)
        if self._telemetry_handle is None:
            self._telemetry_handle = _TelemetryHandle(request.apply_id)

        return self._telemetry_handle.open_or_refresh(
            request.host_meshes,
            request.config,
            spawn_worker_collectors=request.spawn_worker_collectors,
        )

    def handle_admin_url(self, request: AdminUrlRequest) -> str:
        self._ensure_apply_id(request.apply_id)
        os.environ["MONARCH_ADMIN_URL"] = request.admin_url
        return "ok"

    def shutdown(self) -> None:
        try:
            self.clear_mounts()
            if self._telemetry_handle is not None:
                self._telemetry_handle.shutdown()
                self._telemetry_handle = None
            if self._gateway is not None:
                # Worker teardown and callbacks still need the tunnel.
                shutdown_context().get(timeout=5)
        finally:
            self._gateway_stack.close()
            self._gateway = None
            self._gateway_address = None
            self._apply_id = None


def _dbg(msg: str) -> None:
    print(f"[job_sidecar pid={os.getpid()}] {msg}", file=sys.stderr, flush=True)


def _run_job_sidecar(
    socket_path: str,
    runtime_transport: str | None = None,
    attach_to: str | None = None,
) -> None:
    """Run in the child process: bind socket then serve refresh/shutdown requests."""
    import signal as _signal

    def terminate(signum, _frame):
        # Unwind through the server's finally block so allocation-owned
        # subprocesses are reaped. Further signals must not interrupt cleanup;
        # ProcessGuard can still force termination with SIGKILL if it stalls.
        _signal.signal(_signal.SIGINT, _signal.SIG_IGN)
        _signal.signal(_signal.SIGTERM, _signal.SIG_IGN)
        raise SystemExit(128 + signum)

    # Signal handlers are process-global and only legal from the main thread.
    if threading.current_thread() is threading.main_thread():
        _signal.signal(_signal.SIGINT, terminate)
        _signal.signal(_signal.SIGTERM, terminate)

    configure_sidecar_transport(runtime_transport)
    if attach_to is not None:
        attach(attach_to)

    from monarch._src.job.process_guard import _Shutdown

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(5)
    _dbg(f"listening on {socket_path}")

    state = _JobSidecarState()

    try:
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                break
            try:
                f = conn.makefile("rb")
                while True:
                    try:
                        # @lint-ignore PYTHONPICKLEISBAD
                        msg = pickle.load(f)
                    except EOFError:
                        break  # client disconnected (e.g. readiness probe)
                    if isinstance(msg, _Shutdown):
                        _dbg("received shutdown")
                        return
                    try:
                        if isinstance(msg, GatewayRequest):
                            response = state.handle_gateway(msg)
                        elif isinstance(msg, MountsRequest):
                            response = state.handle_mounts(msg)
                        elif isinstance(msg, ClearMountsRequest):
                            response = state.clear_mounts()
                        elif isinstance(msg, TelemetryRequest):
                            try:
                                response = state.handle_telemetry(msg)
                            except Exception:
                                # TODO: Centralize sidecar error.
                                response = {"error": traceback.format_exc()}
                        elif isinstance(msg, AdminUrlRequest):
                            try:
                                response = state.handle_admin_url(msg)
                            except Exception:
                                # TODO: Centralize sidecar error.
                                response = {"error": traceback.format_exc()}
                        else:
                            raise RuntimeError(
                                f"unexpected job sidecar request: {msg!r}"
                            )
                    except Exception:
                        _dbg(
                            "ERROR during job sidecar operation:\n"
                            + traceback.format_exc()
                        )
                        response = (
                            {"error": traceback.format_exc()}
                            if isinstance(msg, GatewayRequest)
                            else "ok"
                        )
                    # @lint-ignore PYTHONPICKLEISBAD
                    conn.sendall(pickle.dumps(response))
            except Exception:
                _dbg("ERROR handling connection:\n" + traceback.format_exc())
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

    finally:
        try:
            state.shutdown()
        finally:
            server.close()
            try:
                os.unlink(socket_path)
            except FileNotFoundError:
                pass
            _dbg("exiting")
