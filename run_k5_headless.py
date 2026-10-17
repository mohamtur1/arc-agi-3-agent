#!/usr/bin/env python3
"""run_k5_headless.py — Headless driver for Project k5.

Verifies credentials, executes offline self-test, runs the push stage,
and prints the mobile prompt card for manual app submission.
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def verify_credentials() -> None:
    """Ensure Kaggle credentials are set via environment or kaggle.json."""
    user = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    token = os.environ.get("KAGGLE_API_TOKEN")
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"

    if (user and key) or token or kaggle_json.is_file():
        return

    sys.stderr.write(
        "ERROR: Kaggle credentials not found.\n"
        "Set KAGGLE_USERNAME and KAGGLE_KEY, or KAGGLE_API_TOKEN, or place ~/.kaggle/kaggle.json.\n"
    )
    sys.exit(1)


def describe_secret(value: str) -> str:
    """Shape of a credential, never its value."""
    if not value:
        return "empty"
    if re.fullmatch(r"[0-9a-fA-F]{32}", value):
        return "%d chars, classic-32hex" % len(value)
    if value.startswith("KGAT_"):
        return "%d chars, KGAT access token" % len(value)
    if value.startswith("kaggle"):
        return "%d chars, unusual (starts with 'kaggle')" % len(value)
    return "%d chars, unrecognised shape" % len(value)


def normalize_credentials() -> None:
    """Put each credential where the Kaggle CLI actually looks for it.

    Kaggle now hands out KGAT_... access tokens, and CLI 2.x reads those from
    KAGGLE_API_TOKEN -- not from KAGGLE_KEY, which stays a legacy 32-hex pair.
    A KGAT token saved in KAGGLE_KEY is present, well-formed, and rejected at
    the API call, which reads like a broken key rather than a misplaced one.
    """
    key = os.environ.get("KAGGLE_KEY", "").strip()
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if key.startswith("KGAT_") and not token:
        os.environ["KAGGLE_API_TOKEN"] = key
        print("  KAGGLE_KEY holds a KGAT access token -> using it as "
              "KAGGLE_API_TOKEN (CLI 2.x reads that name)")
    elif token.startswith("KGAT_") and not key:
        print("  KAGGLE_API_TOKEN holds the KGAT access token (as intended)")
    elif key and not key.startswith("KGAT_") and not token:
        print("  using the legacy KAGGLE_USERNAME/KAGGLE_KEY pair")


def probe_auth() -> None:
    """Prove the Kaggle CLI accepts these credentials, not just that they exist.

    A present-but-rejected key is the difference between 'no credentials' and
    'wrong credentials', and the fix is different. Read-only: lists the user's
    own kernels, page size 1.
    """
    print("=== Step 0: Kaggle Auth Probe ===")
    print("  KAGGLE_USERNAME   : %s"
          % describe_secret(os.environ.get("KAGGLE_USERNAME", "")))
    print("  KAGGLE_KEY        : %s"
          % describe_secret(os.environ.get("KAGGLE_KEY", "")))
    print("  KAGGLE_API_TOKEN  : %s"
          % describe_secret(os.environ.get("KAGGLE_API_TOKEN", "")))

    binary = shutil.which("kaggle")
    base = [binary] if binary else [sys.executable, "-m", "kaggle"]
    cmd = base + ["kernels", "list", "--mine", "--page-size", "1"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode == 0:
        print("  -> Kaggle accepted the credentials (read-only probe OK)")
        return

    if "Authentication required" in out or "401" in out or "Unauthorized" in out:
        sys.stderr.write(
            "\nAUTH PROBE FAILED: the credentials exist but Kaggle REJECTED them.\n"
            "  Present-but-invalid is not the same as absent, and only you can\n"
            "  fix it:\n"
            "    1. Open https://www.kaggle.com/settings/api\n"
            "    2. 'Generate New Token' -> copy the KGAT_... value\n"
            "    3. Repo Settings -> Secrets -> Actions -> New secret named\n"
            "       KAGGLE_API_TOKEN, pasted verbatim\n"
            "    4. Delete the old KAGGLE_KEY (legacy keys are being retired;\n"
            "       a stale one makes every run fail here)\n"
            "  The workflow already passes KAGGLE_API_TOKEN through.\n"
        )
    else:
        sys.stderr.write("\nAUTH PROBE FAILED (unexpected):\n%s\n" % out[-1200:])
    sys.exit(1)


def run_command(cmd: list[str], cwd: Path) -> str:
    """Execute command and stream output, failing loudly on error."""
    res = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(f"Command failed: {' '.join(cmd)}\n")
        sys.stderr.write(res.stderr or res.stdout)
        sys.exit(res.returncode)
    return res.stdout


def main() -> None:
    verify_credentials()
    normalize_credentials()
    probe_auth()
    repo_root = Path(__file__).resolve().parent
    k5_dir = repo_root / "kaggle5"
    variant = os.environ.get("K5_VARIANT", "lever3").strip() or "lever3"
    if variant not in ("lever3", "kernel"):
        sys.stderr.write("ERROR: K5_VARIANT must be 'lever3' or 'kernel', got %r\n"
                         % variant)
        sys.exit(1)

    print("=== Step 1: Pre-flight Self-Test ===")
    self_test_out = run_command([sys.executable, "kaggle5.py", "--self-test"], cwd=k5_dir)
    print(self_test_out.strip())

    allow_same = os.environ.get("K5_ALLOW_SAME", "").strip().lower() in ("1", "true", "yes")
    cmd = [sys.executable, "kaggle5.py", "--stage", "push", "--variant", variant]
    if allow_same:
        cmd.append("--allow-same")
        print("\nnote: allow_same=true — pushing even though the injected notebook")
        print("      matches the current kernel (e.g. after a hardware change).")
    print("\n=== Step 2: Executing Headless Push Stage (variant=%s) ===" % variant)
    push_out = run_command(cmd, cwd=k5_dir)
    print(push_out.strip())

    print("\n" + "=" * 66)
    print("PUSH COMPLETE — MACHINE HALTED. THE MACHINE NEVER AUTO-SUBMITS.")
    print("variant pushed: %s" % variant)
    print("=" * 66)
    print("Mobile Finalization Instructions:")
    print("1. Open the Kaggle Mobile App.")
    print("2. Navigate to: https://www.kaggle.com/code/mohamedmalikturay/arc-agi-3-milestone-2-solution-3e9adb")
    print("3. Wait for the new version to finish execution.")
    print("4. Tap '...' -> 'Submit to Competition' -> Select the new version.")
    print("=" * 66)


if __name__ == "__main__":
    main()
