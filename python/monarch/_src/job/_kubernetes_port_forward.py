# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A kubectl tunnel owned by the allocation sidecar, not a client process."""

import contextlib
import re
import shutil
import subprocess
import threading
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field

_START_TIMEOUT = 30
_STOP_TIMEOUT = 1


@dataclass
class KubernetesPortForward:
    namespace: str
    pod: str
    port: int
    kubeconfig: str | None
    _process: subprocess.Popen[str] | None = field(default=None, init=False, repr=False)
    _output: deque[str] = field(
        default_factory=lambda: deque(maxlen=20), init=False, repr=False
    )

    def check_alive(self) -> None:
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError(
                f"allocation port-forward to pod/{self.pod} has exited; "
                "release and recreate the allocation to reconnect its runtime. "
                + "".join(self._output)
            )

    @contextlib.contextmanager
    def open(self) -> Iterator[str]:
        if shutil.which("kubectl") is None:
            raise RuntimeError("kubectl is required for out-of-cluster port forwarding")
        cmd = [
            "kubectl",
            "port-forward",
            "--address",
            "127.0.0.1",
            "--namespace",
            self.namespace,
            f"pod/{self.pod}",
            f":{self.port}",
        ]
        if self.kubeconfig is not None:
            cmd.extend(["--kubeconfig", self.kubeconfig])
        process = self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        reader = None
        try:
            ready = threading.Event()
            address = None

            # Read through startup warnings and keep draining for the allocation
            # lifetime. Waiting on the event also bounds partial-line output.
            def drain() -> None:
                nonlocal address
                assert process.stdout is not None
                try:
                    for line in process.stdout:
                        self._output.append(line)
                        if address is None:
                            match = re.search(
                                r"Forwarding from 127\.0\.0\.1:(\d+) ->", line
                            )
                            if match is not None:
                                address = f"tcp://127.0.0.1:{match.group(1)}"
                                ready.set()
                finally:
                    ready.set()  # EOF wakes startup too, with diagnostics.

            reader = threading.Thread(target=drain, daemon=True)
            reader.start()
            if not ready.wait(_START_TIMEOUT):
                raise RuntimeError(
                    f"kubectl port-forward to pod/{self.pod} did not start within {_START_TIMEOUT}s"
                )
            if address is None:
                raise RuntimeError(
                    "kubectl port-forward failed to start: "
                    + ("".join(self._output) or "no output")
                )
            yield address
        finally:
            try:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=_STOP_TIMEOUT)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=_STOP_TIMEOUT)
            finally:
                if reader is not None:
                    reader.join(timeout=_STOP_TIMEOUT)
                if process.stdout is not None:
                    process.stdout.close()
                self._process = None
