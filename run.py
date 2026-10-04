"""Launch the Dr.Assistant API and open its local workstation."""

import os
import socket
import time
import webbrowser
import threading


def open_browser_when_ready(host: str, port: int, url: str) -> None:
    """Open the UI only after the local server accepts connections."""
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                print(f"\n[+] Dr.Assistant is ready at {url}\n")
                if not webbrowser.open(url):
                    print("[!] Browser did not open automatically; use the URL above.")
                return
        except OSError:
            time.sleep(0.25)
    print(f"[!] Server did not become ready in time. Open {url} manually if it is running.")


if __name__ == "__main__":
    host = os.environ.get("DR_ASSISTANT_HOST", "127.0.0.1")
    try:
        port = int(os.environ.get("DR_ASSISTANT_PORT", "8000"))
        if not 1 <= port <= 65535:
            raise ValueError
    except ValueError:
        raise SystemExit("DR_ASSISTANT_PORT must be an integer between 1 and 65535.")

    browser_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url = f"http://{browser_host}:{port}"
    print("=" * 72)
    print("  Dr.Assistant - Medical Imaging & Clinical Decision Support")
    print(f"  Starting local server on {url}...")
    print("=" * 72)

    threading.Thread(
        target=open_browser_when_ready,
        args=(browser_host, port, url),
        daemon=True,
    ).start()

    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "Application dependencies are missing. Install them with "
            "'python -m pip install -r requirements.txt'."
        ) from exc

    uvicorn.run("app.main:app", host=host, port=port, reload=False, log_level="info")
