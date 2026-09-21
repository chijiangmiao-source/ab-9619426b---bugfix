"""Container health-check probe: queries the local /health endpoint."""

import json
import os
import sys
import urllib.request


def main() -> int:
    port = os.environ.get("API_PORT", "8080")
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=2
        ) as resp:
            ok = resp.status == 200 and json.load(resp).get("status") == "ok"
    except Exception:
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
