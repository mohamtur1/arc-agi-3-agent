#!/usr/bin/env python3
"""run_k5_headless.py — Headless driver for Project k5.

Verifies credentials, executes offline self-test, runs the push stage,
and prints the mobile prompt card for manual app submission.
"""
import os
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
    repo_root = Path(__file__).resolve().parent
    k5_dir = repo_root / "kaggle5"

    print("=== Step 1: Pre-flight Self-Test ===")
    self_test_out = run_command([sys.executable, "kaggle5.py", "--self-test"], cwd=k5_dir)
    print(self_test_out.strip())

    print("\n=== Step 2: Executing Headless Push Stage ===")
    push_out = run_command([sys.executable, "kaggle5.py", "--stage", "push"], cwd=k5_dir)
    print(push_out.strip())

    print("\n" + "=" * 66)
    print("PUSH COMPLETE — MACHINE HALTED. THE MACHINE NEVER AUTO-SUBMITS.")
    print("=" * 66)
    print("Mobile Finalization Instructions:")
    print("1. Open the Kaggle Mobile App.")
    print("2. Navigate to: https://www.kaggle.com/code/mohamedmalikturay/arc-agi-3-milestone-2-solution-3e9adb")
    print("3. Wait for the new version to finish execution.")
    print("4. Tap '...' -> 'Submit to Competition' -> Select the new version.")
    print("=" * 66)


if __name__ == "__main__":
    main()
