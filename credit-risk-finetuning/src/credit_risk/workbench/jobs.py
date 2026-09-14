"""Single native model-job lane with process groups, durable status and restart recovery."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


def now():
    return datetime.now(UTC).isoformat()


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
                    finished_at=now(),
                )
            elif job["status"] == "queued":
                # No model job starts without an explicit Start in this dashboard session.
                self.store.update_job(
                    job["id"],
                    status="interrupted",
                    error="Dashboard restarted before this job started; no automatic start",
                    finished_at=now(),
                )
        for item in self.store.list("session_item"):
            if item["status"] in ("queued", "running"):
                self.store.update("session_item", item["id"], status="interrupted", finished_at=now())
                self.fail_question(item, "Dashboard restarted before this step finished; ask again")
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
                finished = self.store.update_job(
                    self.current,
                    status="cancelled"
                    if state == "stopping"
                    else ("completed" if code == 0 else "failed"),
                    exit_code=code,
                    finished_at=now(),
                )
                self.cancel_partners(finished)
                if finished["spec"].get("kind") == "session":
                    self.session_ended(finished)
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
                self.store.update_job(
                    self.current, status="running", pid=self.process.pid, started_at=now()
                )
            except OSError as exc:
                self.log.close()
                self.process = None
                self.cancel_partners(
                    self.store.update_job(
                        queued["id"], status="failed", error=type(exc).__name__, finished_at=now()
                    )
                )

    def fail_question(self, item, error):
        try:
            question = self.store.get("question", item["question_id"])
        except ValueError:
            return
        if question["status"] not in ("answered", "failed"):
            self.store.update("question", question["id"], status="failed", error=error)

    def session_ended(self, job):
        """Settle items of an ended session: fail in-flight work, continue or cancel the rest."""
        key = job["spec"]["session_key"]
        items = [i for i in self.store.list("session_item") if i["session_key"] == key]
        for item in items:
            if item["status"] == "running" and item.get("session_job_id") == job["id"]:
                error = f"Model session {job['status']} while this step was running"
                self.store.update("session_item", item["id"], status="failed", error=error, finished_at=now())
                self.fail_question(item, error)
        queued = [i for i in items if i["status"] == "queued"]
        if not queued:
            return
        if job["status"] == "cancelled":
            for item in queued:
                self.store.update("session_item", item["id"], status="cancelled", finished_at=now())
                self.fail_question(item, "Model session was stopped")
            return
        active = {"queued", "running", "stopping"}
        if not any(
            other["status"] in active and other["spec"].get("session_key") == key
            for other in self.store.list("job")
        ):
            spec = {k: v for k, v in job["spec"].items() if k not in ("output", "workspace")}
            self.enqueue(spec)

    def cancel_partners(self, job):
        """A paired comparison is meaningless once one side cannot complete."""
        comparison = job.get("spec", {}).get("comparison_id")
        if not comparison or job["status"] == "completed":
            return
        for other in self.store.list("job"):
            if (
                other["id"] != job["id"]
                and other["status"] == "queued"
                and other["spec"].get("comparison_id") == comparison
            ):
                self.store.update_job(
                    other["id"],
                    status="cancelled",
                    error=f"Paired {job['spec'].get('comparison_role', 'evaluation')} job ended as {job['status']}",
                    finished_at=now(),
                )

    def stop(self, identity):
        with self.mutex:
            job = self.store.get("job", identity)
            if job["status"] == "queued":
                cancelled = self.store.update_job(identity, status="cancelled", finished_at=now())
                self.cancel_partners(cancelled)
                return cancelled
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
                    self.cancel_partners(
                        self.store.update_job(
                            self.current, status="failed", error=type(exc).__name__, finished_at=now()
                        )
                    )

    def close(self):
        self.stopping.set()
        if self.thread:
            self.thread.join(timeout=2)
        if self.current:
            self.stop(self.current)
            self.tick()
        if self.lockfile:
            self.lockfile.close()
