"""Engine manager — each TTS engine runs as an isolated subprocess ("worker")
inside its own uv-managed venv. The server talks to workers over line-delimited
JSON on stdin/stdout; worker logs stream on stderr into a ring buffer.

Why subprocesses: engines pin conflicting deps (chatterbox: torch 2.6 /
transformers 5.2; qwen-tts: its own stack), and killing a process is the only
reliable way to free VRAM on an 8 GB card shared with other apps.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = ROOT / "engines"
ENVS_DIR = ROOT / "envs"
MODELS_DIR = ROOT / "models"  # HF_HOME lives here so the project is self-contained

LOAD_TIMEOUT = 3600   # first load may download several GB of weights
TTS_TIMEOUT = 1800    # long texts on a mid GPU


class EngineError(RuntimeError):
    pass


def _env_python(engine_id: str) -> Path:
    return ENVS_DIR / engine_id / "Scripts" / "python.exe"


class EngineSpec:
    def __init__(self, engine_dir: Path):
        self.dir = engine_dir
        self.id = engine_dir.name
        with open(engine_dir / "engine.json", "r", encoding="utf-8") as f:
            self.meta = json.load(f)

    @property
    def installed(self) -> bool:
        return _env_python(self.id).exists()

    def default_model(self) -> str:
        for m in self.meta.get("models", []):
            if m.get("default"):
                return m["id"]
        return self.meta["models"][0]["id"]

    def public(self) -> dict:
        return {**self.meta, "installed": self.installed}


class Worker:
    """One live engine subprocess."""

    def __init__(self, spec: EngineSpec):
        self.spec = spec
        self.loaded_model: str | None = None
        self.last_used = time.time()
        self.busy = False  # in-flight request — idle watchdog must not kill us
        self.cancelled = False  # killed on purpose via /api/tts/cancel
        self.log: deque[str] = deque(maxlen=400)
        self._req_lock = threading.Lock()
        self._next_id = 1
        self._responses: queue.Queue = queue.Queue()

        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
            "HF_HOME": str(MODELS_DIR),
        }
        self.proc = subprocess.Popen(
            [str(_env_python(spec.id)), str(spec.dir / "worker.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(spec.dir),
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        threading.Thread(target=self._pump_stderr, daemon=True).start()

    # -- pumps ------------------------------------------------------------
    def _pump_stdout(self):
        for line in self.proc.stdout:  # type: ignore[union-attr]
            line = line.strip()
            if not line:
                continue
            try:
                self._responses.put(json.loads(line))
            except json.JSONDecodeError:
                self.log.append(f"[stdout?] {line}")
        self._responses.put(None)  # EOF sentinel

    def _pump_stderr(self):
        for line in self.proc.stderr:  # type: ignore[union-attr]
            self.log.append(line.rstrip())

    # -- rpc ---------------------------------------------------------------
    @property
    def alive(self) -> bool:
        return self.proc.poll() is None

    def request(self, cmd: str, timeout: float, **kw) -> dict:
        with self._req_lock:
            if not self.alive:
                raise EngineError(f"{self.spec.id} worker is not running")
            self.busy = True
            rid = self._next_id
            self._next_id += 1
            msg = {"id": rid, "cmd": cmd, **kw}
            try:
                self.proc.stdin.write(json.dumps(msg) + "\n")  # type: ignore[union-attr]
                self.proc.stdin.flush()  # type: ignore[union-attr]
            except OSError as e:
                raise EngineError(f"{self.spec.id} worker pipe broken: {e}")

            deadline = time.time() + timeout
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    self.kill()
                    raise EngineError(f"{self.spec.id}:{cmd} timed out after {timeout:.0f}s")
                try:
                    resp = self._responses.get(timeout=min(remaining, 5.0))
                except queue.Empty:
                    if not self.alive:
                        if self.cancelled:
                            raise EngineError("generation cancelled")
                        tail = "\n".join(list(self.log)[-15:])
                        raise EngineError(
                            f"{self.spec.id} worker died during {cmd!r}.\nLast log:\n{tail}"
                        )
                    continue
                if resp is None:
                    if self.cancelled:
                        raise EngineError("generation cancelled")
                    tail = "\n".join(list(self.log)[-15:])
                    raise EngineError(
                        f"{self.spec.id} worker exited during {cmd!r}.\nLast log:\n{tail}"
                    )
                if resp.get("id") != rid:
                    continue  # stale response from a timed-out request
                self.last_used = time.time()
                self.busy = False
                if not resp.get("ok"):
                    raise EngineError(resp.get("error", "unknown worker error"))
                return resp

    def kill(self):
        try:
            if self.alive:
                self.proc.kill()
        except OSError:
            pass


class EngineManager:
    def __init__(self, get_config):
        """get_config: zero-arg callable returning the live config dict."""
        self.get_config = get_config
        self.specs: dict[str, EngineSpec] = {}
        for d in sorted(ENGINES_DIR.iterdir()):
            if (d / "engine.json").exists():
                self.specs[d.name] = EngineSpec(d)
        self.workers: dict[str, Worker] = {}
        self.current_worker: Worker | None = None  # set while a tts runs
        self.gen_lock = threading.Lock()   # one GPU → one generation at a time
        self._state_lock = threading.Lock()
        self.installs: dict[str, dict] = {}  # engine_id -> {status, log: deque}
        threading.Thread(target=self._idle_watchdog, daemon=True).start()

    # -- lifecycle ----------------------------------------------------------
    def ensure(self, engine_id: str, model_id: str | None = None) -> Worker:
        spec = self.specs.get(engine_id)
        if spec is None:
            raise EngineError(f"unknown engine {engine_id!r}")
        if not spec.installed:
            raise EngineError(
                f"engine {engine_id!r} is not installed — install it from the UI "
                f"(or POST /api/engines/{engine_id}/install)"
            )
        model_id = model_id or spec.default_model()

        with self._state_lock:
            if self.get_config().get("exclusive_gpu", True):
                for other_id, w in list(self.workers.items()):
                    if other_id != engine_id:
                        w.kill()
                        del self.workers[other_id]

            w = self.workers.get(engine_id)
            if w is not None and not w.alive:
                del self.workers[engine_id]
                w = None
            if w is None:
                w = Worker(spec)
                self.workers[engine_id] = w

        if w.loaded_model != model_id:
            w.request("load", LOAD_TIMEOUT, model=model_id)
            w.loaded_model = model_id
        return w

    def tts(self, engine_id: str, model_id: str | None, payload: dict) -> dict:
        with self.gen_lock:
            w = self.ensure(engine_id, model_id)
            self.current_worker = w
            try:
                return w.request("tts", TTS_TIMEOUT, **payload)
            finally:
                self.current_worker = None

    def cancel_current(self) -> bool:
        """Kill the worker that is generating right now (the only reliable way
        to reclaim the GPU mid-generation). The model reloads on next use."""
        w = self.current_worker
        if w is None or not w.busy or not w.alive:
            return False
        w.cancelled = True
        w.kill()
        with self._state_lock:
            for eid, ww in list(self.workers.items()):
                if ww is w:
                    del self.workers[eid]
        return True

    def unload(self, engine_id: str | None = None):
        with self._state_lock:
            targets = [engine_id] if engine_id else list(self.workers)
            for eid in targets:
                w = self.workers.pop(eid, None)
                if w:
                    w.kill()

    def status(self) -> dict:
        out = {}
        for eid, spec in self.specs.items():
            w = self.workers.get(eid)
            inst = self.installs.get(eid)
            out[eid] = {
                "installed": spec.installed,
                "running": bool(w and w.alive),
                "loaded_model": w.loaded_model if (w and w.alive) else None,
                "idle_seconds": round(time.time() - w.last_used) if (w and w.alive) else None,
                "installing": bool(inst and inst["status"] == "running"),
                "install_status": inst["status"] if inst else None,
            }
        return out

    def worker_log(self, engine_id: str) -> list[str]:
        w = self.workers.get(engine_id)
        return list(w.log) if w else []

    def _idle_watchdog(self):
        while True:
            time.sleep(30)
            try:
                mins = float(self.get_config().get("idle_unload_minutes", 15) or 0)
            except (TypeError, ValueError):
                mins = 15
            if mins <= 0:
                continue
            cutoff = time.time() - mins * 60
            with self._state_lock:
                for eid, w in list(self.workers.items()):
                    if w.alive and not w.busy and w.last_used < cutoff:
                        w.kill()
                        del self.workers[eid]

    # -- installation ---------------------------------------------------------
    def install(self, engine_id: str):
        spec = self.specs.get(engine_id)
        if spec is None:
            raise EngineError(f"unknown engine {engine_id!r}")
        inst = self.installs.get(engine_id)
        if inst and inst["status"] == "running":
            return
        state = {"status": "running", "log": deque(maxlen=2000)}
        self.installs[engine_id] = state
        threading.Thread(target=self._install_thread, args=(spec, state), daemon=True).start()

    def _install_thread(self, spec: EngineSpec, state: dict):
        log = state["log"]
        logfile = spec.dir / "install.log"
        py_version = spec.meta.get("python", "3.12")
        env_dir = ENVS_DIR / spec.id

        def run(args: list[str]) -> int:
            log.append("$ " + " ".join(args))
            p = subprocess.Popen(
                args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", cwd=str(ROOT),
            )
            with open(logfile, "a", encoding="utf-8") as lf:
                lf.write("$ " + " ".join(args) + "\n")
                for line in p.stdout:  # type: ignore[union-attr]
                    log.append(line.rstrip())
                    lf.write(line)
            return p.wait()

        try:
            steps = [
                ["uv", "venv", str(env_dir), "--python", py_version, "--allow-existing"],
                ["uv", "pip", "install",
                 "-r", str(spec.dir / "requirements.txt"),
                 # engines mix the PyTorch wheel index with PyPI; without this
                 # uv takes the first index that has a package (stale torch-side
                 # tqdm/etc.) and resolution fails. best-match spans both.
                 "--index-strategy", "unsafe-best-match",
                 "--python", str(_env_python(spec.id))],
            ]
            for args in steps:
                rc = run(args)
                if rc != 0:
                    state["status"] = f"failed (exit {rc})"
                    return
            # optional per-engine post-install (weight/checkpoint download, repo
            # checkout, patches) — runs inside the engine's own venv so it has
            # its deps (huggingface_hub, git, etc.). The modular seam for
            # git-based engines that aren't a plain pip install.
            setup = spec.dir / "setup.py"
            if setup.exists():
                log.append("running post-install setup.py …")
                rc = run([str(_env_python(spec.id)), str(setup)])
                if rc != 0:
                    state["status"] = f"failed (setup.py exit {rc})"
                    return
            state["status"] = "done"
            log.append(f"engine {spec.id} installed OK")
        except Exception as e:  # surface everything to the UI log
            state["status"] = f"failed ({e})"
            log.append(f"install crashed: {e}")

    def install_log(self, engine_id: str, pos: int = 0) -> dict:
        inst = self.installs.get(engine_id)
        if not inst:
            return {"status": None, "lines": [], "pos": 0}
        lines = list(inst["log"])
        return {"status": inst["status"], "lines": lines[pos:], "pos": len(lines)}
