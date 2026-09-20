"""Portable continuous-build core (M-10): the quality gate in one place, so the
Windows pwsh wrapper and any POSIX shell/CI runner execute identical logic.

Runs ruff (advisory), the config-schema import check, pytest (the gate), an
optional Sphinx build, and an optional pip-audit; records results to
outputs/ci/last_build.json + build_history.csv and refreshes the vault status
block. Exit 0 = green, 1 = failed build.

Usage from any platform:
    python -m scripts.build_core [--profile quick|full|docs]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CI_DIR = ROOT / "outputs" / "ci"
VAULT_DIR = ROOT / "vault" / "50 Build-Ops"
LOCK_FILE = CI_DIR / "build.lock"
LOCK_STALE_MIN = 30
COVERAGE_MIN = 80  # H-03: first documented coverage gate (line + branch)


def _run(cmd: list[str], **kw) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, **kw)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _lock_ok() -> bool:
    if not LOCK_FILE.is_file():
        return True
    age_min = (time.time() - LOCK_FILE.stat().st_mtime) / 60
    if age_min < LOCK_STALE_MIN:
        print("[skip] another build is running")
        return False
    return True


def _acquire_lock() -> None:
    CI_DIR.mkdir(parents=True, exist_ok=True)
    LOCK_FILE.write_text(datetime.now().astimezone().isoformat())


def _release_lock() -> None:
    LOCK_FILE.unlink(missing_ok=True)


def _parse_counts(out: str) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "skipped": 0, "errored": 0}
    for key in counts:
        m = re.search(rf"(\d+) {key}", out)
        if m:
            counts[key] = int(m.group(1))
    return counts


def _coverage_pct() -> str:
    cov = CI_DIR / "coverage.json"
    if not cov.is_file():
        return ""
    try:
        data = json.loads(cov.read_text(encoding="utf-8"))
        return f"{round(data['total']['percent_covered'], 1):.1f}"
    except Exception:  # noqa: BLE001 — coverage summary is advisory
        return ""


def _git_commit() -> str:
    try:
        code, out = _run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT)
        return out.strip() if code == 0 else "no-git"
    except Exception:  # noqa: BLE001
        return "no-git"


def _record(
    profile: str,
    status: str,
    counts: dict[str, int],
    ruff_ok: bool,
    ruff_err: int,
    schema_ok: bool,
    cov_pct: str,
    audit_vulns: int,
    duration: float,
    log_tail: str,
) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %M:%S")
    commit = _git_commit()

    meta = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "profile": profile,
        "status": status,
        "tests_passed": counts["passed"],
        "tests_failed": counts["failed"],
        "tests_errored": counts["errored"],
        "tests_skipped": counts["skipped"],
        "ruff_ok": ruff_ok,
        "ruff_findings": ruff_err,
        "schema_ok": schema_ok,
        "coverage_pct": cov_pct,
        "audit_vulns": audit_vulns,
        "duration_sec": round(duration, 1),
        "commit": commit,
        "summary": log_tail,
    }
    (CI_DIR / "last_build.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    csv = CI_DIR / "build_history.csv"
    if not csv.is_file():
        csv.write_text(
            "timestamp,profile,status,passed,failed,errored,skipped,"
            "ruff_findings,duration_sec,commit\n",
            encoding="utf-8",
        )
    with open(csv, "a", encoding="utf-8") as f:
        f.write(
            f"{ts},{profile},{status},{counts['passed']},{counts['failed']},"
            f"{counts['errored']},{counts['skipped']},{ruff_err},"
            f"{round(duration,1)},{commit}\n"
        )

    _update_vault(
        ts,
        profile,
        status,
        counts,
        ruff_ok,
        ruff_err,
        schema_ok,
        cov_pct,
        audit_vulns,
        duration,
        commit,
        log_tail,
    )


def _update_vault(
    ts,
    profile,
    status,
    counts,
    ruff_ok,
    ruff_err,
    schema_ok,
    cov_pct,
    audit_vulns,
    duration,
    commit,
    log_tail,
) -> None:
    status_path = VAULT_DIR / "Build Status.md"
    if not status_path.is_file():
        return
    callout = "success" if status == "PASS" else "error"
    audit_txt = (
        "not run (quick profile)"
        if audit_vulns < 0
        else "clean"
        if audit_vulns == 0
        else f"{audit_vulns} vulnerabilities — advisory"
    )
    block = f"""<!-- BEGIN AUTO-BUILD-STATUS -->
> [{callout}] Last build: {ts} — {status} (profile: {profile})

- **Status**: `{status}`
- **Profile**: {profile}
- **Tests**: {counts['passed']} passed, {counts['failed']} failed, {counts['errored']} errored, {counts['skipped']} skipped
- **Ruff**: {'clean' if ruff_ok else f"{ruff_err} findings (advisory)"}
- **Schema import**: {'ok' if schema_ok else 'FAILED'}
- **Coverage**: {cov_pct + '%' if cov_pct else 'not measured'}
- **pip-audit**: {audit_txt}
- **Duration**: {round(duration,1)}s
- **Commit**: {commit}
- **Summary**: {log_tail}
- **History**: [[Build Log]] · machine-readable: `outputs/ci/last_build.json`
<!-- END AUTO-BUILD-STATUS -->"""
    content = status_path.read_text(encoding="utf-8")
    pattern = r"<!-- BEGIN AUTO-BUILD-STATUS -->.*?<!-- END AUTO-BUILD-STATUS -->"
    if re.search(pattern, content, re.DOTALL):
        status_path.write_text(
            re.sub(pattern, lambda _: block, content, flags=re.DOTALL), encoding="utf-8"
        )

    log_path = VAULT_DIR / "Build Log.md"
    if log_path.is_file():
        row = (
            f"| {ts} | {profile} | {status} | "
            f"{counts['passed']}/{sum(counts.values())} | {ruff_err} | "
            f"{round(duration,1)}s | {commit} |"
        )
        lines = log_path.read_text(encoding="utf-8").splitlines()
        out, inserted = [], False
        for line in lines:
            if not inserted and line.startswith("|---"):
                out.append(line)
                out.append(row)
                inserted = True
                continue
            out.append(line)
        if inserted:
            log_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def run(profile: str = "full", python: str = "") -> int:
    if not python:
        venv = ROOT / ".venv" / "Scripts" / "python.exe"
        python = str(venv) if venv.is_file() else sys.executable

    if not _lock_ok():
        return 0
    _acquire_lock()
    start = time.time()

    try:
        # 1. ruff (advisory)
        code, ruff_out = _run([python, "-m", "ruff", "check", "src", "scripts", "tests"])
        ruff_ok = code == 0
        m = re.search(r"Found (\d+) error", ruff_out)
        ruff_err = int(m.group(1)) if m else 0
        print(f"[ruff] {'clean' if ruff_ok else 'findings'} ({ruff_err} findings)")

        # 2. config-schema import sanity
        schema_ok = False
        code, _ = _run([python, "-c", "import quobo.config_schema as c; c.QuoboConfig"])
        if code == 0:
            schema_ok = True
        else:
            code, _ = _run(
                [
                    python,
                    "-c",
                    "import sys; sys.path.insert(0,'src'); "
                    "import quobo.config_schema as c; c.QuoboConfig",
                ]
            )
            schema_ok = code == 0
        print(f"[schema] {'ok' if schema_ok else 'IMPORT FAILED'}")

        # 3. pytest (the gate)
        args = [
            python,
            "-m",
            "pytest",
            "tests/",
            "-q",
            "--tb=short",
            "--cov=quobo",
            "--cov-report=term-missing",
            "--cov-report=json:outputs/ci/coverage.json",
            f"--cov-fail-under={COVERAGE_MIN}",
        ]
        if profile == "quick":
            args += ["-m", "not integration"]
        code, pytest_out = _run(args, cwd=ROOT)
        pytest_ok = code == 0 and schema_ok
        counts = _parse_counts(pytest_out)
        print(
            f"[pytest] {'PASS' if pytest_ok else 'FAIL'} "
            f"passed={counts['passed']} failed={counts['failed']} "
            f"errored={counts['errored']} skipped={counts['skipped']}"
        )

        cov_pct = _coverage_pct()
        print(
            f"[coverage] {cov_pct}% (gate >= {COVERAGE_MIN}%)"
            if cov_pct
            else f"[coverage] not measured (gate >= {COVERAGE_MIN}%)"
        )

        # 4. docs (optional)
        docs_ok = True
        if profile == "docs":
            code, _ = _run(
                [python, "-m", "sphinx", "-b", "html", "docs", "docs/_build/html"], cwd=ROOT
            )
            docs_ok = code == 0
            print(f"[docs] {'PASS' if docs_ok else 'FAIL'}")
            if not docs_ok:
                pytest_ok = False

        # 4b. pip-audit (advisory, full/docs only)
        audit_vulns = -1
        if profile != "quick":
            code, audit_out = _run(
                [python, "-m", "pip_audit", "--progress-spinner", "off", "--skip-editable"]
            )
            m = re.search(r"Found (\d+) known vulnerabilit", audit_out)
            if m:
                audit_vulns = int(m.group(1))
            elif "No known vulnerabilities" in audit_out:
                audit_vulns = 0
            print(
                f"[audit] {'clean' if audit_vulns == 0 else 'unavailable (offline?)' if audit_vulns < 0 else f'{audit_vulns} vulnerabilities (advisory)'}"
            )

        duration = time.time() - start
        status = "PASS" if pytest_ok else "FAIL"
        tail = [ln for ln in pytest_out.splitlines() if re.search(r"passed|failed|error", ln)]
        log_tail = tail[-1] if tail else ""

        _record(
            profile,
            status,
            counts,
            ruff_ok,
            ruff_err,
            schema_ok,
            cov_pct,
            audit_vulns,
            duration,
            log_tail,
        )

        print(f"[done] {status} in {round(duration,1)}s (json+csv+vault updated)")
        return 0 if pytest_ok else 1
    finally:
        _release_lock()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", choices=["quick", "full", "docs"], default="full")
    ap.add_argument("--python", default="")
    return run(ap.parse_args().profile, ap.parse_args().python)


if __name__ == "__main__":
    sys.exit(main())
