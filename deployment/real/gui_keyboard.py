"""Keyboard control of the same episode shown in the StarVLA manual GUI."""
import argparse
import json
import time
from urllib.request import ProxyHandler, Request, build_opener
from urllib.error import HTTPError

from deployment.real.manual_client import discard_queued_terminal_input


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui-url", default="http://127.0.0.1:8042")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--collection-id", default="real_train_v1")
    parser.add_argument("--seed-namespace", default="real_train_v1")
    parser.add_argument("--max-joint-speed", type=float, default=0.3)
    args = parser.parse_args()
    opener = build_opener(ProxyHandler({}))
    def request(path, body=None, method="GET"):
        payload = None if body is None else json.dumps(body).encode()
        req = Request(args.gui_url.rstrip("/") + path, data=payload, method=method,
                      headers={"Content-Type": "application/json"})
        try:
            with opener.open(req, timeout=310) as response:
                return json.load(response)
        except HTTPError as exc:
            raise RuntimeError(json.load(exc).get("detail", str(exc))) from exc
    print(request("/api/real/status"))
    print("n + Enter: execute ONE horizon; e: end episode and return to its start; "
          "y/n/drop: label AFTER ending; q: detach keyboard (GUI stays open).")
    try:
        while True:
            status = request("/api/real/status")
            command = input(f"[{status['phase']}]> ").strip().lower()
            if command == "q":
                return
            try:
                if status["phase"] == "await_label" and command in ("y", "n", "drop"):
                    print(request("/api/real/label", {"episode_id": status["episode_id"], "decision": command}, "POST"))
                elif command == "n":
                    print(request("/api/real/step", {"host": args.host, "port": args.port,
                          "prompt": args.prompt, "collection_id": args.collection_id,
                          "seed_namespace": args.seed_namespace, "max_joint_speed": args.max_joint_speed}, "POST"))
                elif command == "e":
                    print(request("/api/real/end", method="POST"))
                    while request("/api/real/status")["phase"] == "stopping":
                        time.sleep(0.5)
                    print(request("/api/real/status"))
                else:
                    print("Use n to execute; e to end; then y/n/drop to label, or use the GUI.")
            except Exception as exc:
                print(f"Request failed: {exc}")
            finally:
                discard_queued_terminal_input()
    except (KeyboardInterrupt, EOFError):
        print("\nRequesting E-STOP; no automatic return motion.")
        print(request("/api/estop", method="POST"))


if __name__ == "__main__":
    main()
