"""Single native model-job lane with process groups, durable status and restart recovery."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from uuid import uuid4


class Jobs:
    def __init__(self, store, launcher=None):
        self.store = store
        self.launcher = launcher or subprocess.Popen
        self.process = None
        self.current = None
        self.log = None
        self.stopping = threading.Event()
        self.mutex = threading.RLock()
        self.thread = None
        self.lockfile = None

    def start(self):
        self.lockfile = (self.store.root / "scheduler.lock").open("w")
        try:
            fcntl.flock(self.lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lockfile.close()
            raise RuntimeError("A workbench already owns this workspace") from None
        for job in self.store.list("job"):
            if job["status"] in ("running", "stopping"):
                # The worker owns a separate GPU lock. Do not signal a possibly reused PID.
                self.store.update_job(
                    job["id"],
                    status="interrupted",
                    error="Dashboard restarted; no automatic resume",
                )
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def enqueue(self, spec):
        identity = uuid4().hex
        output = self.store.root / "runs" / identity
        output.mkdir(parents=True)
        frozen = {
            **spec,
            "output": str(output.resolve()),
            "workspace": str(self.store.root.resolve()),
        }
        (output / "spec.json").write_text(json.dumps(frozen, indent=2))
        return self.store.add("job", {"spec": frozen, "status": "queued"}, identity)

    def tick(self):
        with self.mutex:
            if self.process:
                code = self.process.poll()
                if code is None:
                    return
                state = self.store.get("job", self.current)["status"]
                self.store.update_job(
                    self.current,
                    status="cancelled"
                    if state == "stopping"
                    else ("completed" if code == 0 else "failed"),
                    exit_code=code,
                )
                self.log.close()
                self.process = None
                self.current = None
            if self.stopping.is_set():
                return
            queued = next((j for j in self.store.list("job") if j["status"] == "queued"), None)
            if not queued:
                return
            output = Path(queued["spec"]["output"])
            self.log = (output / "job.log").open("ab")
            try:
                self.process = self.launcher(
                    [
                        sys.executable,
                        "-m",
                        "credit_risk.workbench.worker",
                        str(output / "spec.json"),
                    ],
                    stdout=self.log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    cwd=str(Path(__file__).resolve().parents[3]),
                )
                self.current = queued["id"]
                self.store.update_job(self.current, status="running", pid=self.process.pid)
            except OSError as exc:
                self.log.close()
                self.process = None
                self.store.update_job(queued["id"], status="failed", error=type(exc).__name__)

    def stop(self, identity):
        with self.mutex:
            job = self.store.get("job", identity)
            if job["status"] == "queued":
                return self.store.update_job(identity, status="cancelled")
            if self.current == identity and self.process:
                self.store.update_job(identity, status="stopping")
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
            return self.store.get("job", identity)

    def loop(self):
        while not self.stopping.wait(0.3):
            try:
                self.tick()
            except Exception as exc:  # noqa: BLE001 — background supervisor must persist failures.
                if self.current:
                    self.store.update_job(self.current, status="failed", error=type(exc).__name__)

    def close(self):
        self.stopping.set()
        if self.thread:
            self.thread.join(timeout=2)
        if self.current:
            self.stop(self.current)
            self.tick()
        if self.lockfile:
            self.lockfile.close()
