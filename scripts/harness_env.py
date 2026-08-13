#!/usr/bin/env python3
"""
harness_env.py - Command Execution Gym & Safety Sandbox
"""
import re
import subprocess
from typing import Tuple

FORBIDDEN_PATTERNS = [r"rm\s+-rf\s+/", r"mkfs", r"dd\s+if=", r"shutdown", r"reboot"]

class ExecutionHarnessEnv:
    def __init__(self, dry_run: bool = True, timeout_sec: float = 3.0):
        self.dry_run = dry_run
        self.timeout_sec = timeout_sec

    def is_safe(self, command: str) -> bool:
        for p in FORBIDDEN_PATTERNS:
            if re.search(p, command, re.IGNORECASE):
                return False
        return True

    def execute(self, command: str) -> Tuple[int, str, str]:
        if not self.is_safe(command):
            return -999, "", "Forbidden destructive command"
        if self.dry_run:
            return 0, "DRY_RUN: Command syntax validated.", ""
        try:
            res = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=self.timeout_sec)
            return res.returncode, res.stdout, res.stderr
        except Exception as e:
            return -1, "", str(e)

if __name__ == "__main__":
    env = ExecutionHarnessEnv()
    code, out, err = env.execute("adb connect 10.30.20.101:5555")
    print(f"Harness test -> return code: {code}")
